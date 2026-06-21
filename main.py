"""
Discord bot that watches Instagram accounts for ban / unban status.

Commands (work as both slash commands and !prefix commands):
  checkbanned <username>    - notify when the account becomes banned
  checkunbanned <username>  - notify when the account comes back online
  checkstatus <username>    - check the live status right now
  list                      - list accounts currently being watched
  stopall                   - stop all active watches in this server

A background task runs on a short interval (see CHECK_INTERVAL_SECONDS
below), checks every active watch, and posts an embed with a rendered
profile card the moment a watched condition is met. Once a watch fires, it
is automatically deactivated (the bot stops checking that specific watch
after it notifies). "Time taken" is measured from when the watch was added
(created_at) to the moment the condition is detected.

NOTE on CHECK_INTERVAL_SECONDS: each check is now a single RapidAPI
call (instagram-looter2.p.rapidapi.com) — no headless browser, no
scraping Instagram directly. Checks within one loop tick run
concurrently (CHECK_CONCURRENCY in flight at once), so a 200-account
sweep finishes in tens of seconds, not minutes. The 10-second
interval is fine for that. If you have a paid RapidAPI plan with a
higher per-minute quota, bump CHECK_CONCURRENCY accordingly.
"""

import asyncio
import os
import ssl
import time

import aiohttp
import discord
from discord.ext import commands, tasks
from dotenv import load_dotenv

# Force Python to use certifi's CA bundle for all HTTPS calls (Discord, IG).
# Works around outdated system OpenSSL on macOS / minimal Linux images where
# `pip-system-certs` is missing and the bundled `ssl` can't verify Discord.
try:
    import certifi

    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except Exception as _cert_err:
    print(f"certifi setup warning: {_cert_err!r}")
    _SSL_CTX = None

from card_renderer import render_profile_card
from instagram_checker import check_instagram_account
from storage import WatchStore

ESC = str.maketrans({"_": r"\_"})


def _esc(username: str) -> str:
    return username.translate(ESC)


load_dotenv()
TOKEN = os.getenv("DISCORD_TOKEN")

CHECK_INTERVAL_SECONDS = 10
# No inter-account delay: we're now going through the RapidAPI endpoint
# (instagram-looter2.p.rapidapi.com), not scraping Instagram directly, so
# there's no anti-bot-rate-limit concern. Concurrency is bounded by
# CHECK_CONCURRENCY below instead. The previous 5-second sleep made a
# 200-account sweep take ~17 minutes and starved the event loop, so
# slash commands stopped responding. Setting to 0 fixes that.
INTER_ACCOUNT_DELAY_SECONDS = 0
# How many Instagram checks run in parallel inside one periodic tick.
# 8 is a safe ceiling for RapidAPI's free / Basic plans; bump higher
# if you're on a paid plan with a larger per-minute quota.
CHECK_CONCURRENCY = 8

# Every N iterations (~N*10s), post a one-line "still watching" message to
# the channel for each active watch. Gives the user visible proof the bot is
# alive and checking, and surfaces the current status of the watched account.
HEARTBEAT_EVERY_N_ITERATIONS = 30  # ~5 minutes at 10s interval

_loop_iteration = 0
_loop_stats = {"runs": 0, "checks": 0, "errors": 0, "fired": 0}

CARD_DIR = "cards"
os.makedirs(CARD_DIR, exist_ok=True)

intents = discord.Intents.default()
intents.message_content = True  # needed for classic !prefix commands

bot = commands.Bot(command_prefix="!", intents=intents)
store = WatchStore("watchlist.json")


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours} hours, {minutes} minutes, {secs} seconds"


def build_stats_description(info: dict, elapsed_seconds: float | None = None) -> str:
    lines = []
    if info.get("full_name"):
        lines.append(f"*{info.get('full_name')}*")
    parts = []
    if info.get("posts") is not None:
        parts.append(f"Posts: {info['posts']:,}")
    if info.get("followers") is not None:
        parts.append(f"Followers: {info['followers']:,}")
    if info.get("following") is not None:
        parts.append(f"Following: {info['following']:,}")
    if parts:
        lines.append(" | ".join(parts))
    if elapsed_seconds is not None:
        lines.append(f"⏱️ *Time taken: {format_duration(elapsed_seconds)}*")
    return "\n".join(lines)


async def _parse_usernames(
    ctx: commands.Context,
    username_str: str | None,
    attachment: discord.Attachment | None = None,
) -> list[str]:
    """Parse usernames from command argument (comma-separated) or attached .txt file.
    Returns a list of clean, case-insensitively deduplicated usernames."""
    usernames = []

    if username_str:
        parts = [u.strip().lstrip("@").strip() for u in username_str.split(",")]
        usernames = [u for u in parts if u]

    # Gather attachments: explicit parameter (slash command) or message attachments (prefix)
    attachments: list[discord.Attachment] = []
    if attachment is not None:
        attachments.append(attachment)
    message = getattr(ctx, "message", None)
    if message and message.attachments:
        attachments.extend(message.attachments)

    if not usernames:
        for att in attachments:
            if att.filename.lower().endswith(".txt"):
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(att.url) as resp:
                            if resp.status == 200:
                                content = await resp.text()
                                lines = content.replace(",", "\n").split("\n")
                                usernames = [
                                    u.strip().lstrip("@").strip()
                                    for u in lines
                                    if u.strip()
                                ]
                    break
                except Exception as e:
                    print(f"Failed to read attachment {att.filename}: {e!r}")

    seen = set()
    unique = []
    for u in usernames:
        if u.lower() not in seen:
            seen.add(u.lower())
            unique.append(u)
    return unique


async def _resolve_channel(channel_id: int, guild_id: int | None = None):
    """Resolve a channel by id. Tries the cache first, then falls back to the
    Discord API. Returns None only if the channel truly doesn't exist or the
    bot has no access. Works for guild channels, threads, and DMs."""
    if channel_id is None:
        return None
    # Fast path: in-memory cache (works for guild channels the bot has loaded)
    ch = bot.get_channel(channel_id)
    if ch is not None:
        return ch
    # Medium path: search guild channels explicitly. If we know the guild_id,
    # try that guild first; otherwise scan all guilds.
    if guild_id is not None:
        guild = bot.get_guild(guild_id)
        if guild is not None:
            ch = guild.get_channel(channel_id)
            if ch is not None:
                return ch
    for guild in bot.guilds:
        ch = guild.get_channel(channel_id)
        if ch is not None:
            return ch
    # Slow path: if we know the guild_id but it isn't cached, try fetching
    # the guild first to force-load it, then look up the channel within it.
    # This covers the gap where the bot joined a guild after the gateway
    # READY event and the guild was never cached (slash-command interactions
    # don't populate bot.guilds).
    if guild_id is not None:
        try:
            guild = await bot.fetch_guild(guild_id)
            if guild is not None:
                ch = guild.get_channel(channel_id) or await guild.fetch_channel(
                    channel_id
                )
                if ch is not None:
                    return ch
        except discord.Forbidden as e:
            print(
                f"_resolve_channel({channel_id}) fetch_guild({guild_id}) "
                f"Forbidden: {e!r} (bot may be missing 'bot' OAuth scope)"
            )
        except (discord.NotFound, discord.HTTPException) as e:
            print(
                f"_resolve_channel({channel_id}) fetch_guild({guild_id}) {type(e).__name__}: {e!r}"
            )
    # Last resort: ask Discord directly. Works for DMs / threads.
    try:
        ch = await bot.fetch_channel(channel_id)
        if ch is not None:
            return ch
    except discord.NotFound as e:
        print(f"_resolve_channel({channel_id}) NotFound: {e!r}")
    except discord.Forbidden as e:
        print(f"_resolve_channel({channel_id}) Forbidden: {e!r}")
    except Exception as e:
        print(f"_resolve_channel({channel_id}) failed: {e!r}")
    # All paths failed -- log detailed diagnostics
    print(
        f"_resolve_channel({channel_id}) exhausted all paths. "
        f"bot.get_channel={bot.get_channel(channel_id)!r}, "
        f"guilds={[g.name for g in bot.guilds]}"
    )
    return None


async def _can_post_in(channel) -> bool:
    """Check whether the bot can send a message in the given channel.
    Uses the channel's own permissions object if available (guild channel),
    and falls back to a no-op send for DM channels."""
    if channel is None:
        return False
    perms = None
    try:
        me = channel.guild.me if channel.guild else None
        if me is not None:
            perms = channel.permissions_for(me)
    except Exception:
        perms = None
    if perms is not None:
        return perms.send_messages
    # DM or unknown: try to resolve the channel object; if it works, we can post
    ch = await _resolve_channel(channel.id)
    return ch is not None


def _explain_no_permission(channel) -> str:
    """Return a one-line explanation of why the bot can't post in `channel`."""
    try:
        if channel.guild is not None:
            me = channel.guild.me
            if me is None:
                return "I'm not a member of that server."
            perms = channel.permissions_for(me)
            if not perms.view_channel:
                return "I can't even see this channel."
            if not perms.send_messages:
                return "I can see the channel but can't send messages here."
    except Exception:
        pass
    return "Make sure I have the `Send Messages` permission in this channel."


def build_embed(
    title: str, username: str, info: dict, elapsed_seconds: float | None = None
) -> discord.Embed:
    return discord.Embed(
        title=_esc(title),
        url=f"https://www.instagram.com/{username}/",
        description=build_stats_description(info, elapsed_seconds),
    )


def attach_card_image(embed: discord.Embed, info: dict):
    """Returns a discord.File to send alongside the embed, or None."""
    if info.get("image") and os.path.exists(info["image"]):
        filename = os.path.basename(info["image"])
        file = discord.File(info["image"], filename=filename)
        embed.set_image(url=f"attachment://{filename}")
        return file
    return None


# ---------------------------------------------------------------------------
# Bot lifecycle
# ---------------------------------------------------------------------------


@bot.event
async def on_ready():
    print(f"Logged in as {bot.user} (id: {bot.user.id})")
    # Log every guild + text channel the bot can see, so a future 'channel
    # not found' is easy to diagnose -- you'll know which channels the bot
    # actually has cached.
    for guild in bot.guilds:
        names = [c.name for c in guild.text_channels[:5]]
        print(
            f"  Guild '{guild.name}' ({guild.id}) -- "
            f"{len(guild.text_channels)} text channels, sample: {names}"
        )
    # One-time cleanup: drop any watch whose channel the bot can't reach.
    try:
        watches = await store.get_active_watches()
        dead = []
        for w in watches:
            ch = await _resolve_channel(w["channel_id"], w.get("guild_id"))
            if ch is None:
                dead.append(w["channel_id"])
        if dead:
            removed = await store.cleanup_dead_watches(set(dead))
            print(
                f"Startup cleanup: removed {removed} watch(es) pointing "
                f"to unreachable channels: {dead}"
            )
        else:
            print(f"Startup cleanup: all {len(watches)} active watch(es) reachable.")
    except Exception as e:
        print(f"Startup cleanup failed: {e!r}")

    try:
        await bot.tree.sync()
    except Exception as e:
        print(f"Slash command sync failed: {e}")
    if not periodic_check.is_running():
        periodic_check.start()
        print(f"Started periodic_check loop (every {CHECK_INTERVAL_SECONDS}s)")
    else:
        print("periodic_check loop already running")


def _guild_id(ctx):
    return ctx.guild.id if ctx.guild else 0


async def _defer(ctx):
    """Defer a slash command's response so we have time to do async work
    (file I/O, network calls) without hitting the 3-second interaction timeout.
    No-op for prefix commands."""
    if hasattr(ctx, "interaction") and ctx.interaction is not None:
        interaction = ctx.interaction
        if not interaction.response.is_done():
            try:
                await interaction.response.defer()
            except (discord.NotFound, discord.HTTPException):
                pass


async def _reply(ctx, content: str | None = None, **kwargs):
    """Reply to a command, working in both prefix and slash (hybrid) contexts.
    If the interaction has been deferred, uses followup. Otherwise uses
    response.send_message. Falls back to ctx.send for prefix commands."""
    if hasattr(ctx, "interaction") and ctx.interaction is not None:
        interaction = ctx.interaction
        if interaction.response.is_done():
            return await interaction.followup.send(content=content, **kwargs)
        try:
            return await interaction.response.send_message(content=content, **kwargs)
        except (discord.NotFound, discord.HTTPException):
            # interaction expired; try followup as a last resort
            try:
                return await interaction.followup.send(content=content, **kwargs)
            except Exception:
                return None
    return await ctx.send(content=content, **kwargs)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@bot.hybrid_command(
    name="checkbanned",
    description="Watch IG accounts and post to #banned when banned. Use commas or attach a .txt file.",
)
async def checkbanned(
    ctx: commands.Context, file: discord.Attachment = None, *, username: str = None
):
    usernames = await _parse_usernames(ctx, username, attachment=file)
    if not usernames:
        await _reply(
            ctx,
            "❌ No valid usernames provided. Usage: `/checkbanned user1,user2` or attach a .txt file.",
        )
        return

    guild = ctx.guild
    if guild is None:
        await _reply(ctx, "❌ This command only works in a server.")
        return

    target_ch = discord.utils.get(guild.text_channels, name="banned")
    if target_ch is None:
        await _reply(
            ctx,
            "❌ This server needs a **#banned** text channel. Create one first, then try again.",
        )
        return

    perms = target_ch.permissions_for(guild.me)
    if not perms or not perms.send_messages:
        await _reply(
            ctx,
            f"❌ I can't send messages in #{target_ch.name}. {_explain_no_permission(target_ch)}",
        )
        return

    await _defer(ctx)

    added = []
    skipped = []
    for uname in usernames:
        was_added = await store.add_watch(
            _guild_id(ctx), target_ch.id, ctx.author.id, uname, "banned"
        )
        if was_added:
            added.append(uname)
        else:
            skipped.append(uname)

    parts = []
    if added:
        if len(added) <= 5:
            names = ", ".join("@" + _esc(u) for u in added)
            parts.append(
                f"✅ Added {len(added)} watch(es): {names} — notifications go to #{target_ch.name}"
            )
        else:
            parts.append(
                f"✅ Added {len(added)} watches — notifications go to #{target_ch.name}. "
                f"({', '.join('@' + _esc(u) for u in added[:3])} +{len(added) - 3} more)"
            )
    if skipped:
        parts.append(f"ℹ️ Skipped {len(skipped)} (already watching).")
    await _reply(ctx, "\n".join(parts))


@bot.hybrid_command(
    name="checkunbanned",
    description="Watch Instagram account(s) and notify #unban when they recover. Use commas or attach a .txt file.",
)
async def checkunbanned(
    ctx: commands.Context, file: discord.Attachment = None, *, username: str = None
):
    usernames = await _parse_usernames(ctx, username, attachment=file)
    if not usernames:
        await _reply(
            ctx,
            "❌ No valid usernames provided. Usage: `/checkunbanned user1,user2` or attach a .txt file.",
        )
        return

    guild = ctx.guild
    if guild is None:
        await _reply(ctx, "❌ This command only works in a server.")
        return

    target_ch = discord.utils.get(guild.text_channels, name="unban")
    if target_ch is None:
        await _reply(
            ctx,
            "❌ This server needs an **#unban** text channel. Create one first, then try again.",
        )
        return

    perms = target_ch.permissions_for(guild.me)
    if not perms or not perms.send_messages:
        await _reply(
            ctx,
            f"❌ I can't send messages in #{target_ch.name}. {_explain_no_permission(target_ch)}",
        )
        return

    await _defer(ctx)

    added = []
    skipped = []
    for uname in usernames:
        was_added = await store.add_watch(
            _guild_id(ctx), target_ch.id, ctx.author.id, uname, "unbanned"
        )
        if was_added:
            added.append(uname)
        else:
            skipped.append(uname)

    parts = []
    if added:
        if len(added) <= 5:
            names = ", ".join("@" + _esc(u) for u in added)
            parts.append(
                f"✅ Added {len(added)} watch(es): {names} — notifications go to #{target_ch.name}"
            )
        else:
            parts.append(
                f"✅ Added {len(added)} watches — notifications go to #{target_ch.name}. "
                f"({', '.join('@' + _esc(u) for u in added[:3])} +{len(added) - 3} more)"
            )
    if skipped:
        parts.append(f"ℹ️ Skipped {len(skipped)} (already watching).")
    await _reply(ctx, "\n".join(parts))


@bot.hybrid_command(
    name="checkstatus",
    description="Check Instagram account(s) current status. Use commas or attach a .txt file.",
)
async def checkstatus(
    ctx: commands.Context, file: discord.Attachment = None, *, username: str = None
):
    await _defer(ctx)
    usernames = await _parse_usernames(ctx, username, attachment=file)
    if not usernames:
        await _reply(
            ctx,
            "❌ No valid usernames provided. Usage: `/checkstatus user1,user2` or attach a .txt file.",
        )
        return

    for uname in usernames:
        path = os.path.join(CARD_DIR, f"status_{uname}.png")
        try:
            info = await check_instagram_account(uname, path)
        except Exception as e:
            await _reply(ctx, f"❌ Error checking @{_esc(uname)}: {e!r}")
            continue

        title_map = {
            "active": f"Account Active | @{uname} ✅",
            "banned": f"Account Banned | @{uname} ❌",
            "unknown": f"Status Unknown | @{uname} ⚠️",
        }
        embed = build_embed(title_map[info["status"]], uname, info)
        file = attach_card_image(embed, info)

        if file:
            await _reply(ctx, embed=embed, file=file)
        else:
            await _reply(ctx, embed=embed)

        if len(usernames) > 1:
            await asyncio.sleep(1)

    if len(usernames) > 1:
        await _reply(ctx, f"✅ Checked {len(usernames)} accounts.")


@bot.hybrid_command(
    name="fakeban",
    description="Generate a fake ban notification card for a username.",
)
async def fakeban(ctx: commands.Context, username: str):
    await _defer(ctx)
    username = username.lstrip("@").strip()
    path = os.path.join(CARD_DIR, f"fakeban_{username}.png")
    render_profile_card(
        username=username,
        output_path=path,
        followers=None,
        following=None,
        posts=None,
        avatar_bytes=None,
    )
    title = f"Account Banned | @{username} 🔒🚫"
    embed = build_embed(title, username, {})
    embed.set_image(url=f"attachment://{os.path.basename(path)}")
    file = discord.File(path, filename=os.path.basename(path))
    await _reply(ctx, embed=embed, file=file)


@bot.hybrid_command(
    name="list", description="List every account currently being watched and what for."
)
async def list_cmd(ctx: commands.Context):
    await _defer(ctx)
    watches = await store.list_watches(_guild_id(ctx))
    if not watches:
        await _reply(ctx, "No accounts are currently being watched.")
        return
    lines = []
    for w in watches:
        waiting_for = (
            "getting banned" if w["watch_type"] == "banned" else "coming back online"
        )
        lines.append(f"• **@{_esc(w['username'])}** — waiting for it to {waiting_for}")
    await _reply(ctx, "**Currently watched accounts:**\n" + "\n".join(lines))


@bot.hybrid_command(
    name="stopall", description="Stop all active watches in this server."
)
async def stopall(ctx: commands.Context):
    await _defer(ctx)
    stopped = await store.stop_all(_guild_id(ctx))
    await _reply(
        ctx,
        f"Stopped {stopped} active watch(es). No more checks will run for this server.",
    )


@bot.hybrid_command(
    name="botstatus",
    description="Show internal bot stats: loop runs, errors, last fire time.",
)
async def botstatus(ctx: commands.Context):
    await _defer(ctx)
    is_running = periodic_check.is_running()
    next_iter = periodic_check.next_iteration
    if next_iter is not None:
        next_str = f"<t:{int(next_iter.timestamp())}:R>"
    else:
        next_str = "n/a"
    stats = _loop_stats
    msg = (
        f"**Bot diagnostics**\n"
        f"• Loop running: `{is_running}`\n"
        f"• Check interval: `{CHECK_INTERVAL_SECONDS}s`\n"
        f"• Next iteration: {next_str}\n"
        f"• Total loop runs: `{stats['runs']}`\n"
        f"• Successful checks: `{stats['checks']}`\n"
        f"• Errors: `{stats['errors']}`\n"
        f"• Notifications fired: `{stats['fired']}`"
    )
    await _reply(ctx, msg)


@bot.hybrid_command(
    name="botperms",
    description="Show what permissions the bot has in this channel (debug).",
)
async def botperms(ctx: commands.Context):
    ch = ctx.channel
    guild_name = ch.guild.name if ch.guild else "<DM or unknown>"
    channel_id = ch.id
    # Try to find the actual channel object via resolve (works for partial
    # channels that arrived via slash command in uncached guilds)
    fresh = await _resolve_channel(channel_id)
    if fresh is not None and fresh.guild is not None:
        guild_name = fresh.guild.name
        me = fresh.guild.me
        p = fresh.permissions_for(me) if me else None
    else:
        me = ch.guild.me if ch.guild else None
        p = ch.permissions_for(me) if me else None
    if p is None:
        await _reply(
            ctx,
            f"**Permissions in #{ch.name}** (server: `{guild_name}`)\n"
            f"Channel ID: `{channel_id}`\n\n"
            f"❌ Could not compute permissions. The bot may not be a "
            f"member of the server, or the channel may be a DM.",
        )
        return
    flags = [
        ("View Channel", p.view_channel),
        ("Send Messages", p.send_messages),
        ("Send Messages in Threads", p.send_messages_in_threads),
        ("Embed Links", p.embed_links),
        ("Attach Files", p.attach_files),
        ("Read Message History", p.read_message_history),
        ("Use External Emojis", p.external_emojis),
    ]
    lines = [
        f"**Permissions in #{ch.name}** (server: `{guild_name}`)",
        f"Channel ID: `{channel_id}`",
        "",
    ]
    for label, allowed in flags:
        mark = "✅" if allowed else "❌"
        lines.append(f"{mark} {label}")
    if p.administrator:
        lines.insert(3, "✅ Administrator (all permissions)")
    await _reply(ctx, "\n".join(lines))


@bot.hybrid_command(
    name="forcecheck", description="Run one check cycle right now (admin debug)."
)
async def forcecheck(ctx: commands.Context):
    await _defer(ctx)
    await _reply(ctx, "Running one check cycle...")
    try:
        await periodic_check()
        await _reply(ctx, "Done.")
    except Exception as e:
        await _reply(ctx, f"Error: {e!r}")


# ---------------------------------------------------------------------------
# Background periodic check
# ---------------------------------------------------------------------------


@tasks.loop(seconds=CHECK_INTERVAL_SECONDS)
async def periodic_check():
    global _loop_iteration
    _loop_iteration += 1
    _loop_stats["runs"] += 1
    iter_id = _loop_iteration

    try:
        watches = await store.get_active_watches()
    except Exception as e:
        print(f"[iter {iter_id}] failed to load watches: {e!r}")
        _loop_stats["errors"] += 1
        return

    if not watches:
        if iter_id == 1 or iter_id % 30 == 0:
            print(f"[iter {iter_id}] no active watches")
        return

    print(
        f"[iter {iter_id}] checking {len(watches)} watch(es) "
        f"(concurrency={CHECK_CONCURRENCY})"
    )

    # Run all checks concurrently, bounded by a semaphore so we don't
    # hammer the API. This is the fix for "bot doesn't respond in Discord":
    # the previous sequential + 5s-sleep design held the event loop for
    # ~17 minutes on a 200-account sweep, so slash commands never got a
    # chance to respond. With CHECK_CONCURRENCY in flight at once, the
    # same sweep finishes in ~30s and the event loop is free in between.
    sem = asyncio.Semaphore(CHECK_CONCURRENCY)

    async def _check_one(w: dict) -> None:
        async with sem:
            await _check_one_watch(w, iter_id)

    await asyncio.gather(*[_check_one(w) for w in watches], return_exceptions=True)


async def _check_one_watch(w: dict, iter_id: int) -> None:
    """Run a single check for one watch: fetch status, optionally heartbeat,
    fire the notification if the watched condition is met, deactivate."""
    username = w["username"]
    path = os.path.join(CARD_DIR, f"{username}_{w['id']}.png")
    try:
        info = await check_instagram_account(username, path)
        _loop_stats["checks"] += 1
    except Exception as e:
        print(f"[iter {iter_id}] error checking @{username}: {e!r}")
        _loop_stats["errors"] += 1
        return

    print(
        f"[iter {iter_id}] @{username} -> status={info['status']} "
        f"followers={info.get('followers')} watch_type={w['watch_type']}"
    )

    # Heartbeat: every N iterations, post a one-line "still watching"
    # message to the channel so the user can see the bot is alive.
    if iter_id % HEARTBEAT_EVERY_N_ITERATIONS == 0:
        channel = await _resolve_channel(w["channel_id"], w.get("guild_id"))
        if channel is None:
            print(
                f"[iter {iter_id}] heartbeat: channel {w['channel_id']} "
                f"unreachable for @{username}; deleting watch"
            )
            await store.delete_watch(w["id"])
            return
        try:
            emoji = {
                "active": "✅",
                "banned": "❌",
                "unknown": "⚠️",
            }.get(info["status"], "❔")
            note = {
                "active": "still active — I'll post when it changes",
                "banned": "currently banned — I'll post when it recovers",
                "unknown": "couldn't read its status this cycle",
            }.get(info["status"], "status unknown")
            await channel.send(
                f"💓 Still watching **@{_esc(username)}** — {note} {emoji}"
            )
        except Exception:
            pass

    wants_banned = w["watch_type"] == "banned" and info["status"] == "banned"
    wants_unbanned = w["watch_type"] == "unbanned" and info["status"] == "active"

    if wants_banned or wants_unbanned:
        if wants_unbanned:
            title = f"Account Recovered | @{username} 🏆✅"
        else:
            title = f"Account Banned | @{username} 🔒🚫"

        created_at = w.get("created_at", time.time())
        elapsed = time.time() - created_at
        embed = build_embed(title, username, info, elapsed_seconds=elapsed)
        file = attach_card_image(embed, info)

        # Notify the target channel (the #banned or #unban channel)
        channel = await _resolve_channel(w["channel_id"], w.get("guild_id"))
        if channel is None:
            print(
                f"[iter {iter_id}] channel {w['channel_id']} no longer "
                f"accessible for @{username}; deleting watch. "
                f"User must re-run the command in a valid channel."
            )
            await store.delete_watch(w["id"])
            return

        try:
            if file:
                await channel.send(content=f"<@{w['user_id']}>", embed=embed, file=file)
            else:
                await channel.send(content=f"<@{w['user_id']}>", embed=embed)
            _loop_stats["fired"] += 1
            print(
                f"[iter {iter_id}] notified for @{username} "
                f"({w['watch_type']}) in channel {w['channel_id']}"
            )
        except discord.Forbidden as e:
            print(
                f"[iter {iter_id}] permission denied sending to "
                f"channel {w['channel_id']} for @{username}: {e!r}; "
                f"deleting watch"
            )
            await store.delete_watch(w["id"])
        except discord.NotFound as e:
            print(
                f"[iter {iter_id}] channel {w['channel_id']} deleted "
                f"for @{username}: {e!r}; deleting watch"
            )
            await store.delete_watch(w["id"])
        except Exception as e:
            print(
                f"[iter {iter_id}] failed to send notification for "
                f"@{username}: {e!r}; deleting watch"
            )
            await store.delete_watch(w["id"])
        else:
            # Notification sent successfully -- deactivate so it
            # doesn't fire again.
            await store.deactivate(w["id"])


@periodic_check.before_loop
async def before_periodic_check():
    print("periodic_check: waiting for bot to be ready...")
    await bot.wait_until_ready()
    # Wait for the guild cache to be populated. discord.py fires on_ready
    # before all guild channels are loaded, which causes get_channel() to
    # return None for valid channels. Wait a few seconds for the cache to
    # settle, and also wait until we have at least one guild with channels.
    print("periodic_check: waiting for guild cache to populate...")
    for _ in range(30):  # up to 30 seconds
        if bot.guilds and any(g.channels for g in bot.guilds):
            break
        await asyncio.sleep(1)
    print(
        f"periodic_check: cache ready, {len(bot.guilds)} guild(s), "
        f"{sum(len(g.channels) for g in bot.guilds)} total channels. "
        f"Loop will start firing."
    )


@periodic_check.error
async def periodic_check_error(error):
    """If the loop itself crashes, log it loudly and restart it."""
    print(f"[FATAL] periodic_check crashed: {error!r}")
    import traceback

    traceback.print_exc()
    if not periodic_check.is_running():
        try:
            periodic_check.restart()
            print("periodic_check restarted after crash.")
        except Exception as e:
            print(f"Failed to restart periodic_check: {e!r}")


if __name__ == "__main__":
    if not TOKEN:
        raise SystemExit(
            "DISCORD_TOKEN is not set. Copy .env.example to .env and fill it in."
        )
    bot.run(TOKEN)
