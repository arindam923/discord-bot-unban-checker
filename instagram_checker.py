"""
Looks up Instagram profile data via the instagram-looter2.p.rapidapi.com API
and renders a styled profile card image.

This replaces the previous Playwright + residential-proxy approach. A single
REST call to RapidAPI returns everything we need (full_name, follower /
following / post counts, bio, profile picture URL) without launching a
browser, without scraping Instagram directly, and without burning proxy
bandwidth.

The avatar is then downloaded from the CDN URL using aiohttp and passed to
card_renderer.render_profile_card to produce the final PNG.

Status detection:
  - "active" -- API returned status=true with a real profile
  - "banned" -- API returned status=false (account doesn't exist / banned)
  - "unknown" -- network / 4xx / 5xx error (caller can log and retry later)
"""

import os
import ssl

import aiohttp
from dotenv import load_dotenv

from card_renderer import render_profile_card

load_dotenv()

RAPIDAPI_HOST = os.getenv("RAPIDAPI_HOST", "instagram-looter2.p.rapidapi.com")
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")
RAPIDAPI_URL = f"https://{RAPIDAPI_HOST}/profile"

try:
    import certifi

    _AIOHTTP_SSL = ssl.create_default_context(cafile=certifi.where())
except Exception:
    _AIOHTTP_SSL = None

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


async def _download_avatar(url: str) -> bytes | None:
    if not url:
        return None
    for ssl_ctx in (_AIOHTTP_SSL, None):
        try:
            kwargs = dict(
                timeout=aiohttp.ClientTimeout(total=10),
                headers={"User-Agent": USER_AGENT},
            )
            if ssl_ctx is not None:
                kwargs["ssl"] = ssl_ctx
            async with aiohttp.ClientSession() as session:
                async with session.get(url, **kwargs) as resp:
                    if resp.status == 200:
                        data = await resp.read()
                        if data and len(data) > 100:
                            return data
        except Exception:
            continue
    return None


async def _fetch_profile(username: str) -> dict:
    """Call the RapidAPI endpoint and return its raw JSON body."""
    if not RAPIDAPI_KEY:
        raise RuntimeError(
            "RAPIDAPI_KEY is not set. Copy .env.example to .env and fill it in."
        )
    headers = {
        "Content-Type": "application/json",
        "x-rapidapi-host": RAPIDAPI_HOST,
        "x-rapidapi-key": RAPIDAPI_KEY,
        "User-Agent": USER_AGENT,
    }
    params = {"username": username}
    last_err: Exception | None = None
    for ssl_ctx in (_AIOHTTP_SSL, None):
        try:
            kwargs = dict(
                timeout=aiohttp.ClientTimeout(total=15),
                headers=headers,
                params=params,
            )
            if ssl_ctx is not None:
                kwargs["ssl"] = ssl_ctx
            async with aiohttp.ClientSession() as session:
                async with session.get(RAPIDAPI_URL, **kwargs) as resp:
                    text = await resp.text()
                    if resp.status == 200:
                        import json

                        return json.loads(text)
                    # 4xx -- API key wrong, account not found, quota hit, etc.
                    # Treat as "banned" candidate (the API often returns 404
                    # for missing / banned accounts on this endpoint).
                    raise RuntimeError(
                        f"RapidAPI returned HTTP {resp.status}: {text[:200]}"
                    )
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"RapidAPI request failed: {last_err!r}")


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

    try:
        data = await _fetch_profile(username)
    except Exception as e:
        # Any HTTP / network error from the API endpoint is NOT evidence that
        # the Instagram account is banned. A 404 here is almost always a
        # quota / auth / network problem on our side; a 200 with status:false
        # is the real "banned / doesn't exist" signal (handled below).
        # So all exception paths stay as "unknown" so we don't fire false
        # "Recovered" / "Banned" notifications.
        print(f"  [api] error fetching @{username}: {e!r}")
        result["status"] = "unknown"
        try:
            render_profile_card(
                username=username,
                output_path=output_path,
                status=result["status"],
            )
            result["image"] = output_path
        except Exception:
            result["image"] = None
        return result

    # The API puts the "is this a real account?" flag at the top level.
    if not data.get("status"):
        result["status"] = "banned"
    else:
        result["status"] = "active"

    result["full_name"] = data.get("full_name") or None
    result["bio"] = data.get("biography") or None

    followers = (data.get("edge_followed_by") or {}).get("count")
    following = (data.get("edge_follow") or {}).get("count")
    posts = (data.get("edge_owner_to_timeline_media") or {}).get("count")
    result["followers"] = int(followers) if isinstance(followers, int) else None
    result["following"] = int(following) if isinstance(following, int) else None
    result["posts"] = int(posts) if isinstance(posts, int) else None

    avatar_url = data.get("profile_pic_url_hd") or data.get("profile_pic_url")
    if avatar_url:
        result["avatar_bytes"] = await _download_avatar(avatar_url)

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
