"""
Discord bot that watches Instagram accounts for ban / unban status.

Commands (work as both slash commands and !prefix commands):
  checkbanned <username>    - notify when the account becomes banned
  checkunbanned <username>  - notify when the account comes back online
  checkstatus <username>    - check the live status right now
  list                      - list accounts currently being watched
  stopall                   - stop all active watches in this server
  botstatus                 - show cache stats and loop diagnostics

Polling architecture:
  The loop ticks every CHECK_INTERVAL_SECONDS (60s), but individual accounts
  are only checked based on their last known status (tiered frequency):
    - Active accounts  → every 5 min  (bans are rare, no need to poll hard)
    - Banned accounts  → every 1 min  (user wants fast unban detection)
    - Unknown accounts → every 2 min  (retry cadence)

  This cuts API calls by ~87% vs checking all accounts every tick.

  Each check calls Instagram's own GraphQL API
  (i.instagram.com/api/v1/users/web_profile_info/) — no RapidAPI, no API
  key, no monthly quota. This is the same endpoint Instagram's web frontend
  uses, and it's free and unauthenticated.

  For each account:
    - HTTP 200 + JSON → status="active", full profile data available
    - HTTP 404       → status="banned" (account gone)
    - HTTP 429/401   → status="rate_limited" (deferred with backoff)

   A persistent status_cache.json tracks the last status + profile signature
   per username. If both are unchanged, the card PNG render is skipped.
   Notifications only fire on actual status transitions (active↔banned).

   If Instagram rate-limits a request, the account is deferred with
   exponential backoff (60s → 15 min max) so it retries automatically
   without blocking the sweep. Set PROXY_URLS in .env to rotate across
   egress IPs and reduce rate-limiting.
"""

import asyncio
import os
import re
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
except Exception:
    _SSL_CTX = None

from card_renderer import render_profile_card
from instagram_checker import check_instagram_account
from status_cache import StatusCache
from storage import WatchStore, _sanitize_username

ESC = str.maketrans({"_": r"\_"})


def _esc(username: str) -> str:
    return username.translate(ESC)


load_dotenv()
TOKEN = os.getenv("DISCORD_TOKEN")

# Polling interval (seconds). The loop ticks every 60s, but individual
# accounts are only checked based on their tier interval below (not every
# tick). This dramatically reduces API calls — active accounts are only
# checked every 5 min, banned every 1 min, etc.
CHECK_INTERVAL_SECONDS = 120

# How many accounts to fetch in parallel inside one tick.
# 8 is safe from one IP without triggering Instagram's soft rate limit.
# If you consistently get rate_limited, lower this to 4 or add proxies.
CHECK_CONCURRENCY = 3

# Status-based check frequency (seconds since last_checked).
# Active accounts are checked less often (bans are rare events).
# Banned accounts are checked more often (user wants fast unban detection).
# Unknown accounts retry at a medium cadence.
TIER_INTERVALS = {
    "active": 300,  # 5 minutes
    "banned": 60,  # 1 minute
    "unknown": 120,  # 2 minutes
    None: 60,  # default for never-checked accounts
}

# Optional: channel ID for rate-limit alerts.
# Falls back to the first guild's system channel.
ALERT_CHANNEL_ID = os.getenv("ALERT_CHANNEL_ID")

_loop_iteration = 0
_loop_stats = {
    "runs": 0,
    "checks": 0,
    "errors": 0,
    "fired": 0,
    "cache_hits": 0,
    "cache_misses": 0,
    "last_sweep_duration": 0.0,
}

CARD_DIR = "cards"
os.makedirs(CARD_DIR, exist_ok=True)

intents = discord.Intents.default()
intents.message_content = True  # needed for classic !prefix commands

bot = commands.Bot(command_prefix="!", intents=intents)
store = WatchStore("watchlist.json")
status_cache = StatusCache("status_cache.json")


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
    username_str: str | None = None,
    file: discord.Attachment | None = None,
) -> list[str]:
    """Parse usernames from command argument (comma/space/newline-separated),
    slash-command file upload, or prefix-command message attachment .txt file.
    Returns a list of clean, case-insensitively deduplicated usernames."""
    usernames = []

    def _normalize_one(raw: str) -> str:
        return raw.strip().lstrip("@").strip()

    def _split_any(s: str) -> list[str]:
        # Split on commas, newlines, tabs, and runs of whitespace. Allows users
        # to paste usernames separated by any of those (e.g. copy-pasted from a
        # chat) without producing malformed compound usernames like
        # "camping_lovee     @frog_ins".
        for sep in (",", "\r\n", "\n", "\t"):
            s = s.replace(sep, " ")
        return [tok.strip() for tok in s.split(" ") if tok.strip()]

    if username_str:
        usernames = [
            _normalize_one(tok)
            for tok in _split_any(username_str)
            if _normalize_one(tok)
        ]

    # Gather attachments: explicit slash-command parameter, then message attachments (prefix)
    attachments: list[discord.Attachment] = []
    if file:
        attachments.append(file)
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
                                usernames = [
                                    _normalize_one(tok)
                                    for tok in _split_any(content)
                                    if _normalize_one(tok)
                                ]
                    break
                except Exception:
                    pass

    seen = set()
    unique = []
    for u in usernames:
        # Sanitize username using the utility function
        sanitized = _sanitize_username(u)
        for u_clean in sanitized:
            if u_clean.lower() not in seen:
                seen.add(u_clean.lower())
                unique.append(u_clean)
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
        except discord.Forbidden:
            pass
        except (discord.NotFound, discord.HTTPException):
            pass
    # Last resort: ask Discord directly. Works for DMs / threads.
    try:
        ch = await bot.fetch_channel(channel_id)
        if ch is not None:
            return ch
    except discord.NotFound:
        pass
    except discord.Forbidden:
        pass
    except Exception:
        pass
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


async def _resolve_alert_channel():
    """Find the channel for alerts + aggregate heartbeat.
    Tries ALERT_CHANNEL_ID from env, then falls back to the first guild's
    system channel."""
    if ALERT_CHANNEL_ID:
        try:
            ch = await _resolve_channel(int(ALERT_CHANNEL_ID))
            if ch is not None:
                return ch
        except Exception:
            pass
    if bot.guilds:
        guild = bot.guilds[0]
        if guild.system_channel:
            return guild.system_channel
    return None


async def _send_alert(message: str) -> None:
    """Log error/warning alerts to terminal only. Never sends Discord notifications."""
    import sys
    from datetime import datetime

    timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    log_message = f"{timestamp} [ALERT] {message}"
    print(log_message, file=sys.stderr)
    sys.stderr.flush()


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
    # One-time cleanup: drop any watch whose channel the bot can't reach.
    try:
        watches = await store.get_active_watches()
        dead = []
        for w in watches:
            ch = await _resolve_channel(w["channel_id"], w.get("guild_id"))
            if ch is None:
                dead.append(w["channel_id"])
        if dead:
            await store.cleanup_dead_watches(set(dead))
    except Exception:
        pass

    try:
        await bot.tree.sync()
    except Exception:
        pass

    if not periodic_check.is_running():
        periodic_check.start()

    # Prime cache at startup
    try:
        await status_cache.get_all()
    except Exception:
        pass


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
# Paginator
# ---------------------------------------------------------------------------


class WatchPaginator(discord.ui.View):
    """Paginated embed view for the /list command."""

    def __init__(self, lines: list[str], page_size: int = 15):
        super().__init__(timeout=120)
        self.lines = lines
        self.page_size = page_size
        self.current_page = 0
        self.total_pages = max(1, (len(lines) + page_size - 1) // page_size)
        self._update_buttons()

    def _get_page_content(self) -> str:
        start = self.current_page * self.page_size
        end = start + self.page_size
        return "\n".join(self.lines[start:end])

    def _build_embed(self) -> discord.Embed:
        embed = discord.Embed(
            title="Currently watched accounts",
            description=self._get_page_content(),
        )
        embed.set_footer(
            text=f"Page {self.current_page + 1}/{self.total_pages}  •  {len(self.lines)} total"
        )
        return embed

    def _update_buttons(self):
        self.previous.disabled = self.current_page == 0
        self.next.disabled = self.current_page >= self.total_pages - 1

    @discord.ui.button(label="◀ Previous", style=discord.ButtonStyle.secondary)
    async def previous(
        self, interaction: discord.Interaction, button: discord.ui.Button
    ):
        self.current_page -= 1
        self._update_buttons()
        await interaction.response.edit_message(embed=self._build_embed(), view=self)

    @discord.ui.button(label="Next ▶", style=discord.ButtonStyle.secondary)
    async def next(self, interaction: discord.Interaction, button: discord.ui.Button):
        self.current_page += 1
        self._update_buttons()
        await interaction.response.edit_message(embed=self._build_embed(), view=self)

    async def on_timeout(self):
        for child in self.children:
            child.disabled = True
        if hasattr(self, "_message") and self._message:
            try:
                await self._message.edit(view=self)
            except Exception:
                pass


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
    usernames = await _parse_usernames(ctx, username, file=file)
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
    usernames = await _parse_usernames(ctx, username, file=file)
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
    usernames = await _parse_usernames(ctx, username, file=file)
    if not usernames:
        await _reply(
            ctx,
            "❌ No valid usernames provided. Usage: `/checkstatus user1,user2` or attach a .txt file.",
        )
        return

    title_map = {
        "active": "Account Active | @{u} ✅",
        "banned": "Account Banned | @{u} ❌",
    }

    sem = asyncio.Semaphore(CHECK_CONCURRENCY)

    async def _check_one_status(uname: str):
        async with sem:
            path = os.path.join(CARD_DIR, f"status_{uname}.png")
            try:
                return await check_instagram_account(uname, path)
            except Exception as e:
                return {"status": "unknown", "_error": repr(e), "image": None}

    results = await asyncio.gather(*[_check_one_status(u) for u in usernames])

    successful = 0
    for uname, info in zip(usernames, results):
        if "_error" in info:
            print(f"[checkstatus] {uname}: ERROR {info['_error']}", flush=True)
            continue
        if info["status"] not in ("active", "banned"):
            print(f"[checkstatus] {uname}: skipped ({info['status']})", flush=True)
            continue
        successful += 1
        title_tmpl = title_map[info["status"]]
        embed = build_embed(title_tmpl.format(u=uname), uname, info)
        file = attach_card_image(embed, info)
        if file:
            await _reply(ctx, embed=embed, file=file)
        else:
            await _reply(ctx, embed=embed)

    if len(usernames) > 1:
        await _reply(
            ctx, f"✅ {successful}/{len(usernames)} accounts fetched successfully."
        )


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

    view = WatchPaginator(lines)
    view._message = await _reply(ctx, embed=view._build_embed(), view=view)


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
    description="Show cache stats and loop diagnostics.",
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
    cache_all = await status_cache.get_all()

    # Count accounts by tier
    tier_counts = {"active": 0, "banned": 0, "unknown": 0, "never": 0}
    for entry in cache_all.values():
        status = entry.get("confirmed")
        if status in tier_counts:
            tier_counts[status] += 1
        else:
            tier_counts["unknown"] += 1

    total = stats["cache_hits"] + stats["cache_misses"]
    hit_rate = f"{stats['cache_hits'] / total * 100:.1f}%" if total > 0 else "n/a"
    msg = (
        f"**Bot diagnostics**\n"
        f"• Loop running: `{is_running}`\n"
        f"• Loop tick: `{CHECK_INTERVAL_SECONDS}s`\n"
        f"• Concurrency: `{CHECK_CONCURRENCY}`\n"
        f"• Next iteration: {next_str}\n"
        f"• Last sweep duration: `{stats['last_sweep_duration']:.1f}s`\n"
        f"\n**Tier schedule**\n"
        f"• Active (5 min): `{tier_counts['active']}` account(s)\n"
        f"• Banned (1 min): `{tier_counts['banned']}` account(s)\n"
        f"• Unknown (2 min): `{tier_counts['unknown']}` account(s)\n"
        f"• Never checked: `{tier_counts['never']}` account(s)\n"
        f"\n**Cache**\n"
        f"• Cached accounts: `{len(cache_all)}`\n"
        f"• Card-render hit rate: `{hit_rate}` ({stats['cache_hits']} hits / {stats['cache_misses']} misses)\n"
        f"\n**Loop stats**\n"
        f"• Total runs: `{stats['runs']}`\n"
        f"• API checks: `{stats['checks']}`\n"
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
    """Fetch due accounts via Instagram's free GraphQL API.
    Accounts are checked on a tiered schedule based on their last known
    status (active=5min, banned=1min, unknown=2min). Skip card render for
    unchanged accounts. Notify on status transitions."""
    global _loop_iteration
    _loop_iteration += 1
    _loop_stats["runs"] += 1

    try:
        grouped = await store.get_active_watches_grouped_by_username()
    except Exception:
        _loop_stats["errors"] += 1
        return

    if not grouped:
        return

    all_usernames = list(grouped.keys())

    # Filter to only accounts that are due for their tier
    due_usernames = await status_cache.get_due_usernames(all_usernames, TIER_INTERVALS)

    if not due_usernames:
        return

    sweep_start = time.time()
    sem = asyncio.Semaphore(CHECK_CONCURRENCY)
    # results: username_lower -> (prev_status, new_status, info)
    results: dict[str, tuple[str | None, str | None, dict | None]] = {}
    iter_cache_hits = 0
    iter_cache_misses = 0

    async def _check_one(username_lower: str) -> None:
        nonlocal iter_cache_hits, iter_cache_misses
        orig = grouped[username_lower][0]["username"]
        cached = await status_cache.get(username_lower)
        cached_sig = cached.get("profile_sig") if cached else None
        prev_status = cached.get("confirmed") if cached else None

        async with sem:
            path = os.path.join(CARD_DIR, f"{orig}.png")
            try:
                info = await check_instagram_account(orig, path, cached_sig=cached_sig)
                _loop_stats["checks"] += 1
            except Exception:
                _loop_stats["errors"] += 1
                results[username_lower] = (prev_status, None, None)
                return

        status = info["status"]

        if status == "rate_limited":
            await status_cache.set_rate_limited(username_lower)
            results[username_lower] = (prev_status, None, None)
            return

        if status not in ("active", "banned"):
            results[username_lower] = (prev_status, None, None)
            return

        if (
            prev_status == status
            and info.get("profile_sig")
            and cached_sig
            and info["profile_sig"] == cached_sig
        ):
            iter_cache_hits += 1
        else:
            iter_cache_misses += 1

        await status_cache.clear_retry_backoff(username_lower)
        await status_cache.set(
            username_lower,
            {
                "confirmed": status,
                "last_checked": time.time(),
                "last_seen": time.time(),
                "profile_sig": info.get("profile_sig"),
                "avatar_url": info.get("avatar_url"),
            },
        )
        results[username_lower] = (prev_status, status, info)

    await asyncio.gather(
        *[_check_one(u) for u in due_usernames], return_exceptions=True
    )

    _loop_stats["last_sweep_duration"] = time.time() - sweep_start
    _loop_stats["cache_hits"] += iter_cache_hits
    _loop_stats["cache_misses"] += iter_cache_misses

    # Fan-out notifications: notify when desired status is met, then delete
    for username_lower, (prev_status, new_status, info) in results.items():
        if new_status is None or info is None:
            continue
        for w in grouped.get(username_lower, []):
            wants_banned = w["watch_type"] == "banned" and new_status == "banned"
            wants_unbanned = w["watch_type"] == "unbanned" and new_status == "active"
            if not (wants_banned or wants_unbanned):
                continue
            # Notify and delete from watchlist (one-shot notification)
            await _notify_watch(w, new_status, info)


async def _notify_watch(w: dict, confirmed: str, info: dict) -> None:
    """Send ban/unban notification to Discord channel. Only logs to terminal."""
    import sys
    from datetime import datetime

    username = w["username"]
    if confirmed == "active":
        title = f"Account Recovered | @{username} 🏆✅"
    else:
        title = f"Account Banned | @{username} 🔒🚫"

    created_at = w.get("created_at", time.time())
    elapsed = time.time() - created_at
    embed = build_embed(title, username, info, elapsed_seconds=elapsed)
    file = attach_card_image(embed, info)

    channel = await _resolve_channel(w["channel_id"], w.get("guild_id"))
    if channel is None:
        await store.delete_watch(w["id"])
        return

    try:
        if file:
            await channel.send(content=f"<@{w['user_id']}>", embed=embed, file=file)
        else:
            await channel.send(content=f"<@{w['user_id']}>", embed=embed)
        _loop_stats["fired"] += 1
    except discord.Forbidden:
        await store.delete_watch(w["id"])
    except discord.NotFound:
        await store.delete_watch(w["id"])
    except Exception:
        await store.delete_watch(w["id"])
    else:
        # Successfully notified - delete from watchlist (one-shot)
        await store.delete_watch(w["id"])


@periodic_check.before_loop
async def before_periodic_check():
    await bot.wait_until_ready()
    # Wait for the guild cache to be populated. discord.py fires on_ready
    # before all guild channels are loaded, which causes get_channel() to
    # return None for valid channels. Wait a few seconds for the cache to
    # settle, and also wait until we have at least one guild with channels.
    for _ in range(30):  # up to 30 seconds
        if bot.guilds and any(g.channels for g in bot.guilds):
            break
        await asyncio.sleep(1)


@periodic_check.error
async def periodic_check_error(error):
    """If the loop itself crashes, restart it."""
    if not periodic_check.is_running():
        try:
            periodic_check.restart()
        except Exception:
            pass


if __name__ == "__main__":
    if not TOKEN:
        raise SystemExit(
            "DISCORD_TOKEN is not set. Copy .env.example to .env and fill it in."
        )
    bot.run(TOKEN)
