"""
Checks an Instagram profile by fetching the public profile page HTML via aiohttp
and parsing og:meta tags and page content for status detection.

A cookie warmup (GET instagram.com) is done once per bot run to obtain session
cookies that reduce rate-limiting. The raw HTML approach avoids Instagram's
API rate-limiting which aggressively blocks datacenter IPs.

If even the HTML fetch fails (blank page, 403, redirect to login), the bot
falls back to "unknown" status.
"""

import asyncio
import re
import ssl
import time

import aiohttp
from aiohttp import CookieJar

try:
    import certifi

    _AIOHTTP_SSL = ssl.create_default_context(cafile=certifi.where())
except Exception:
    _AIOHTTP_SSL = None

from card_renderer import render_profile_card

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

HOMEPAGE_URL = "https://www.instagram.com/"

# Shared cookie jar across checks within a single bot run
_COOKIE_JAR: "CookieJar | None" = None
_COOKIES_WARMED = False
_WARMUP_LOCK = asyncio.Lock()

# Per-username cooldown
_last_check: dict[str, float] = {}
_MIN_INTERVAL = 5.0

# Profile-page markup markers that indicate a banned / unavailable account
NOT_FOUND_MARKERS = [
    "isn't available",
    "may be broken",
    "may have been removed",
    "page not found",
    "user not found",
    "sorry, this page isn't available",
]

# Regex to parse counts from og:description: "1,234 Followers, 567 Following, 89 Posts"
FOLLOWERS_RE = re.compile(r"([\d,]+(?:[.,]\d+)?[KkMm]?)\s+Followers", re.IGNORECASE)
FOLLOWING_RE = re.compile(r"([\d,]+)\s+Following", re.IGNORECASE)
POSTS_RE = re.compile(r"([\d,]+)\s+Posts", re.IGNORECASE)

# Parse full name from og:title: "Full Name (@username) • Instagram ..."
TITLE_NAME_RE = re.compile(r"^(.+?)\s*\(@")


def _parse_count(text: str) -> int | None:
    """Parses Instagram's abbreviated counts: '104M', '27.8K', '1,234', etc."""
    if not text:
        return None
    s = text.replace(",", "").strip()
    mult = 1
    if s and s[-1] in "Kk":
        mult = 1_000
        s = s[:-1]
    elif s and s[-1] in "Mm":
        mult = 1_000_000
        s = s[:-1]
    try:
        return int(float(s) * mult)
    except (ValueError, TypeError):
        return None


async def _warmup_cookies(ssl_context=None) -> None:
    """Visit Instagram homepage once to seed session cookies."""
    global _COOKIES_WARMED, _COOKIE_JAR
    if _COOKIES_WARMED:
        return
    async with _WARMUP_LOCK:
        if _COOKIES_WARMED:
            return
        if _COOKIE_JAR is None:
            _COOKIE_JAR = CookieJar()
        try:
            async with aiohttp.ClientSession(cookie_jar=_COOKIE_JAR) as session:
                async with session.get(
                    HOMEPAGE_URL,
                    headers={
                        "User-Agent": USER_AGENT,
                        "Accept": (
                            "text/html,application/xhtml+xml,"
                            "application/xml;q=0.9,*/*;q=0.8"
                        ),
                        "Accept-Language": "en-US,en;q=0.9",
                    },
                    timeout=aiohttp.ClientTimeout(total=15),
                    ssl=ssl_context,
                ) as resp:
                    await resp.read()
            _COOKIES_WARMED = True
            print("  [cookies] warmed up")
        except Exception as e:
            print(f"  [cookies] warmup failed: {e!r}")


def _extract_meta(html: str, prop: str) -> str | None:
    """Extract content from an og: meta tag in HTML."""
    # Match: <meta property="og:title" content="..." />
    pattern = rf'<meta\s+property=["\']og:{prop}["\']\s+content=["\']([^"\']*)["\']'
    m = re.search(pattern, html, re.IGNORECASE)
    if m:
        return m.group(1)
    # Try alternate attribute order: <meta content="..." property="og:title" />
    pattern2 = rf'<meta\s+content=["\']([^"\']*)["\']\s+property=["\']og:{prop}["\']'
    m = re.search(pattern2, html, re.IGNORECASE)
    if m:
        return m.group(1)
    return None


async def _download_avatar(url: str) -> bytes | None:
    if not url:
        return None
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url,
                timeout=aiohttp.ClientTimeout(total=10),
                headers={"User-Agent": USER_AGENT},
                ssl=_AIOHTTP_SSL,
            ) as resp:
                if resp.status == 200:
                    data = await resp.read()
                    if data and len(data) > 100:
                        return data
    except Exception:
        pass
    return None


async def check_instagram_account(username: str, output_path: str) -> dict:
    """
    Returns a dict:
      {
        "status": "active" | "banned" | "unknown",
        "image": path-or-None,
        "followers": int-or-None,
        "following": int-or-None,
        "posts": int-or-None,
        "full_name": str-or-None,
        "bio": str-or-None,
        "avatar_bytes": bytes-or-None,
      }
    """
    result = {
        "status": "unknown",
        "image": None,
        "followers": None,
        "following": None,
        "posts": None,
        "full_name": None,
        "bio": None,
        "avatar_bytes": None,
    }

    # Per-username cooldown
    now = time.time()
    last = _last_check.get(username, 0)
    if now - last < _MIN_INTERVAL:
        await asyncio.sleep(_MIN_INTERVAL - (now - last))
    _last_check[username] = time.time()

    # Ensure session cookies are seeded
    await _warmup_cookies(_AIOHTTP_SSL)

    url = f"https://www.instagram.com/{username}/"
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": (
            "text/html,application/xhtml+xml,"
            "application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.instagram.com/",
        "Sec-Fetch-Site": "same-origin",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Dest": "document",
    }

    for attempt in range(3):
        try:
            async with aiohttp.ClientSession(cookie_jar=_COOKIE_JAR) as session:
                async with session.get(
                    url,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=20),
                    ssl=_AIOHTTP_SSL,
                    allow_redirects=True,
                ) as resp:
                    if resp.status == 404:
                        result["status"] = "banned"
                        break

                    if resp.status == 200:
                        # Check if we were redirected to login or challenge page
                        final_url = str(resp.real_url)
                        if "accounts/login" in final_url or "challenge" in final_url:
                            print(
                                f"  [blocked] @{username}: redirected to "
                                f"login/challenge page"
                            )
                            result["status"] = "unknown"
                            break

                        html = await resp.text()
                        html_lower = html.lower()

                        # -- Detect banned / unavailable ----------------------------------
                        if any(m in html_lower for m in NOT_FOUND_MARKERS):
                            result["status"] = "banned"
                            break

                        # -- Extract og:meta tags ------------------------------------------
                        meta_title = _extract_meta(html, "title")
                        meta_desc = _extract_meta(html, "description")
                        og_image = _extract_meta(html, "image")

                        # -- Parse counts from og:description ----------------------------
                        if meta_desc:
                            f_m = FOLLOWERS_RE.search(meta_desc)
                            g_m = FOLLOWING_RE.search(meta_desc)
                            p_m = POSTS_RE.search(meta_desc)
                            if f_m:
                                result["followers"] = _parse_count(f_m.group(1))
                            if g_m:
                                result["following"] = _parse_count(g_m.group(1))
                            if p_m:
                                result["posts"] = _parse_count(p_m.group(1))

                        # -- Parse full_name from og:title -------------------------------
                        if meta_title:
                            m = TITLE_NAME_RE.match(meta_title)
                            if m:
                                candidate = m.group(1).strip()
                                if candidate.lower() != username.lower():
                                    result["full_name"] = candidate

                        # -- Classify status ----------------------------------------------
                        if result["followers"] is not None or (
                            meta_title and username.lower() in meta_title.lower()
                        ):
                            result["status"] = "active"
                        elif meta_title is None and meta_desc is None:
                            # Empty page — likely blocked
                            snippet = html[:200].replace("\n", " ")
                            print(
                                f"  [empty-page] @{username}: no meta tags, "
                                f"html_snippet='{snippet}'"
                            )
                            result["status"] = "unknown"
                        else:
                            # Has some meta tags but couldn't confirm active
                            result["status"] = "unknown"

                        # -- Download avatar --------------------------------------------
                        if og_image:
                            result["avatar_bytes"] = await _download_avatar(og_image)

                        break  # success — exit retry loop

                    elif resp.status == 429:
                        wait = 15 * (attempt + 1)
                        print(
                            f"  [rate-limit] @{username}: HTTP 429 "
                            f"(attempt {attempt + 1}/3) — waiting {wait}s"
                        )
                        await asyncio.sleep(wait)
                        # Re-warmup cookies in case they expired
                        global _COOKIES_WARMED
                        _COOKIES_WARMED = False
                        await _warmup_cookies(_AIOHTTP_SSL)

                    elif resp.status in (302, 301):
                        loc = resp.headers.get("Location", "")
                        if "login" in loc or "challenge" in loc:
                            print(f"  [blocked] @{username}: 302 redirect to login")
                        result["status"] = "unknown"
                        break

                    else:
                        text = await resp.text()
                        print(
                            f"  [http-error] @{username}: HTTP {resp.status} "
                            f"{text[:200]}"
                        )
                        break

        except aiohttp.ClientError as e:
            print(f"  [network-error] @{username}: {e!r}")
            if attempt < 2:
                await asyncio.sleep(5)
            else:
                break
        except Exception as e:
            print(f"  [unexpected-error] @{username}: {e!r}")
            break

    # Render the profile card
    try:
        render_profile_card(
            username=username,
            output_path=output_path,
            status=result["status"],
            full_name=result.get("full_name"),
            bio=result.get("bio"),
            posts=result.get("posts"),
            followers=result.get("followers"),
            following=result.get("following"),
            avatar_bytes=result.get("avatar_bytes"),
        )
        result["image"] = output_path
    except Exception as e:
        print(f"Card rendering failed for @{username}: {e}")
        result["image"] = None

    return result
