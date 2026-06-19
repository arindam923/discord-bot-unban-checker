"""
Checks an Instagram profile using Instagram's public web_profile_info API.

This endpoint returns JSON with full profile data (followers, following, posts,
full name, bio, avatar URL) without requiring a browser or login. It works
reliably from any IP — datacenter, residential, or otherwise — unlike browser
scraping which Instagram blocks for non-residential IPs.

The avatar image is downloaded separately via aiohttp and passed to
card_renderer.render_profile_card to produce the final PNG.
"""

import asyncio
import json
import os
import re
import ssl
import time

import aiohttp

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

# Rate-limit tracking: simple in-memory per-username cooldown
_last_check: dict[str, float] = {}
_MIN_INTERVAL = 3.0  # seconds between API calls for the same username


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

    # Rate-limit: avoid hammering the same username too fast
    now = time.time()
    last = _last_check.get(username, 0)
    if now - last < _MIN_INTERVAL:
        await asyncio.sleep(_MIN_INTERVAL - (now - last))
    _last_check[username] = time.time()

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

    url = f"{API_URL}?username={username}"

    try:
        async with aiohttp.ClientSession() as session:
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
                        # Null user → account banned, suspended, or deactivated
                        result["status"] = "banned"
                    else:
                        result["status"] = "active"
                        result["followers"] = user.get("edge_followed_by", {}).get(
                            "count"
                        )
                        result["following"] = user.get("edge_follow", {}).get("count")
                        result["posts"] = user.get(
                            "edge_owner_to_timeline_media", {}
                        ).get("count")
                        result["full_name"] = user.get("full_name")
                        result["bio"] = user.get("biography")

                        avatar_url = user.get("profile_pic_url_hd") or user.get(
                            "profile_pic_url"
                        )
                        if avatar_url:
                            result["avatar_bytes"] = await _download_avatar(avatar_url)
                elif resp.status == 404:
                    result["status"] = "banned"
                elif resp.status == 429:
                    print(f"  [rate-limit] @{username}: HTTP 429 — cooling down 30s")
                    await asyncio.sleep(30)
                else:
                    print(
                        f"  [api-error] @{username}: HTTP {resp.status} "
                        f"{await resp.text()[:200]}"
                    )
    except aiohttp.ClientError as e:
        print(f"  [network-error] @{username}: {e!r}")
    except Exception as e:
        print(f"  [unexpected-error] @{username}: {e!r}")

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
