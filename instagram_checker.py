"""
Checks an Instagram profile using Instagram's public web_profile_info API.

Requires a session cookie warmup (visiting the Instagram homepage first) to get
a csrftoken. Without it, the API returns HTTP 429. Cookies persist across calls
via a module-level CookieJar so we only pay the warmup cost once per bot run.

Returns full profile data (followers, following, posts, full name, bio, avatar
URL) without a browser or login. Works reliably from any IP.
"""

import asyncio
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

INSTAGRAM_APP_ID = "9366197433924594"
API_URL = "https://www.instagram.com/api/v1/users/web_profile_info/"
HOMEPAGE_URL = "https://www.instagram.com/"

# Shared cookie jar so csrftoken persists across checks within a single bot run
_COOKIE_JAR = CookieJar()
_COOKIES_WARMED = False
_WARMUP_LOCK = asyncio.Lock()

# Per-username cooldown to avoid hammering the same account
_last_check: dict[str, float] = {}
_MIN_INTERVAL = 5.0


async def _warmup_cookies(ssl_context=None) -> None:
    """Visit Instagram homepage once to seed cookies (csrftoken, ig_did, mid)."""
    global _COOKIES_WARMED
    if _COOKIES_WARMED:
        return
    async with _WARMUP_LOCK:
        if _COOKIES_WARMED:
            return
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
                    await resp.read()  # consume body to populate cookies
            _COOKIES_WARMED = True
            csrf = _get_csrftoken()
            print(f"  [cookies] warmed up (csrftoken={'yes' if csrf else 'no'})")
        except Exception as e:
            print(f"  [cookies] warmup failed: {e!r}")


def _get_csrftoken() -> str | None:
    for cookie in _COOKIE_JAR:
        if cookie.key == "csrftoken":
            return cookie.value
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

    # Ensure we have session cookies before calling the API
    await _warmup_cookies(_AIOHTTP_SSL)

    csrftoken = _get_csrftoken()
    headers = {
        "User-Agent": USER_AGENT,
        "X-IG-App-ID": INSTAGRAM_APP_ID,
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": f"https://www.instagram.com/{username}/",
        "Origin": "https://www.instagram.com",
        "Sec-Fetch-Site": "same-origin",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Dest": "empty",
    }
    if csrftoken:
        headers["X-CSRFToken"] = csrftoken

    url = f"{API_URL}?username={username}"

    # Retry loop for transient rate-limiting
    for attempt in range(3):
        try:
            async with aiohttp.ClientSession(cookie_jar=_COOKIE_JAR) as session:
                async with session.get(
                    url,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=15),
                    ssl=_AIOHTTP_SSL,
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        user = data.get("data", {}).get("user")

                        if user is None:
                            result["status"] = "banned"
                        else:
                            result["status"] = "active"
                            result["followers"] = user.get("edge_followed_by", {}).get(
                                "count"
                            )
                            result["following"] = user.get("edge_follow", {}).get(
                                "count"
                            )
                            result["posts"] = user.get(
                                "edge_owner_to_timeline_media", {}
                            ).get("count")
                            result["full_name"] = user.get("full_name")
                            result["bio"] = user.get("biography")

                            avatar_url = user.get("profile_pic_url_hd") or user.get(
                                "profile_pic_url"
                            )
                            if avatar_url:
                                result["avatar_bytes"] = await _download_avatar(
                                    avatar_url
                                )
                        break  # success — exit retry loop

                    elif resp.status == 404:
                        result["status"] = "banned"
                        break

                    elif resp.status == 429:
                        wait = 15 * (attempt + 1)
                        print(
                            f"  [rate-limit] @{username}: HTTP 429 "
                            f"(attempt {attempt + 1}/3) — waiting {wait}s"
                        )
                        await asyncio.sleep(wait)
                        # Re-warmup cookies in case they expired
                        _COOKIES_WARMED = False
                        await _warmup_cookies(_AIOHTTP_SSL)
                        csrftoken = _get_csrftoken()
                        if csrftoken:
                            headers["X-CSRFToken"] = csrftoken

                    else:
                        text = await resp.text()
                        print(
                            f"  [api-error] @{username}: HTTP {resp.status} "
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
