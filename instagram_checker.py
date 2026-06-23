"""
Fetches Instagram profile data directly from Instagram's own GraphQL API.

NO RapidAPI. NO API key. NO quota. NO monthly limit.

Endpoint: www.instagram.com/api/v1/users/web_profile_info/?username=<u>
Headers:  X-IG-App-ID (Instagram's public web app ID), User-Agent, Referer

This is the same endpoint Instagram's own web frontend calls when you open
a profile page. It's free, unauthenticated, and returns full profile JSON
(followers, following, posts, bio, avatar URL) for active accounts, and
HTTP 404 for banned/deleted/nonexistent accounts.

Status detection:
  - "active"       -- HTTP 200 + JSON with user data
  - "banned"       -- HTTP 404 (account doesn't exist / banned / deleted)
  - "rate_limited" -- HTTP 429 (per-IP rate limit; needs proxies or slower interval)
  - "unknown"      -- network error, non-JSON response, parse failure

Anti rate-limit mitigation:
  A cookie warmup (GET https://www.instagram.com/) seeds the session with
  `mid`, `csrftoken`, `ig_nrcb` cookies that Instagram issues to real
  browsers. On a 429 or 401 require_login response we re-warm once and retry
  the request. This evades soft cookie-less blocks but will NOT help if the
  egress IP itself is hard-blocked by Instagram (typical for cloud/VPS IPs).

Proxy rotation:
  Set PROXY_URLS in .env (comma-separated). Two formats are accepted:
    1) "host:port:user:pass"               (Geonode / sticky-provider style)
    2) "http://user:pass@host:port"        (standard URL form)
  Each request routes through the next proxy in round-robin order. With
  rotating residential backends the same proxy URL hands out a fresh egress
  IP per call, so retrying a 429 / 401 require_login through the same URL
  usually re-rolls past the soft block. The cookie jar is wiped before
  every retry so a soft-blocked device fingerprint from a failed attempt
  cannot taint the next one.

  When PROXY_URLS is unset, all calls go direct (no proxy).

A shared aiohttp.ClientSession is used for connection pooling. In-memory
avatar cache keyed by CDN URL avoids re-downloading the same avatar. Card
render is skipped when the profile signature is unchanged from cache.
"""

import asyncio
import itertools
import json
import os
import ssl

import aiohttp
from dotenv import load_dotenv

from card_renderer import render_profile_card
from status_cache import compute_profile_sig

load_dotenv()

try:
    import certifi

    _AIOHTTP_SSL = ssl.create_default_context(cafile=certifi.where())
except Exception:
    _AIOHTTP_SSL = None

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

# Instagram's public web app ID. This is embedded in Instagram's own JS
# bundle and is used by their web frontend for unauthenticated GraphQL calls.
# It is NOT a secret and is not tied to any account or quota.
IG_APP_ID = "936619743392459"

GRAPHQL_URL = "https://www.instagram.com/api/v1/users/web_profile_info/"
WARMUP_URL = "https://www.instagram.com/"


def _parse_proxy_list(raw: str | None) -> list[str]:
    """Parse PROXY_URLS env var into a list of aiohttp-ready proxy URLs.
    Accepts both `host:port:user:pass` and `http://user:pass@host:port`."""
    if not raw:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        url = token if "://" in token else _colon_form_to_url(token)
        if url and url not in seen:
            seen.add(url)
            out.append(url)
    return out


def _colon_form_to_url(spec: str) -> str:
    """Convert 'host:port:user:pass' -> 'http://user:pass@host:port'.
    Also tolerates 'host:port' (no auth) and 'host:port:user'
    (user only, no pass)."""
    parts = spec.split(":")
    if len(parts) < 2:
        return ""
    host, port = parts[0], parts[1]
    if len(parts) == 2:
        return f"http://{host}:{port}"
    user = parts[2]
    if len(parts) >= 4:
        pwd = ":".join(parts[3:])  # in case password itself contains ':'
        return f"http://{user}:{pwd}@{host}:{port}"
    return f"http://{user}@{host}:{port}"


PROXIES: list[str] = _parse_proxy_list(os.getenv("PROXY_URLS"))
_proxy_cycle = itertools.cycle(PROXIES) if PROXIES else None

# Shared session for connection pooling (lazy-init, lives for bot lifetime)
_session: aiohttp.ClientSession | None = None

# In-memory avatar cache: url -> bytes (or None if download failed).
_avatar_url_cache: dict[str, bytes | None] = {}


def _next_proxy() -> str | None:
    """Return the next proxy URL in round-robin order, or None if no proxies."""
    if _proxy_cycle is None:
        return None
    return next(_proxy_cycle)


async def _get_session() -> aiohttp.ClientSession:
    global _session
    if _session is None or _session.closed:
        connector = aiohttp.TCPConnector(limit=0, limit_per_host=0, ssl=_AIOHTTP_SSL)
        # unsafe=True so the jar keeps cookies regardless of the host we hit
        # (warmup host vs GraphQL host both share the .instagram.com domain,
        # but unsafe guards against any future endpoint/host changes).
        jar = aiohttp.CookieJar(unsafe=True)
        _session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=15),
            connector=connector,
            cookie_jar=jar,
            headers={
                "User-Agent": USER_AGENT,
                "X-IG-App-ID": IG_APP_ID,
                "Accept": "*/*",
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": WARMUP_URL,
            },
        )
    return _session


async def _warmup(proxy: str | None, clear_cookies: bool = False) -> None:
    """Seed the session with Instagram's browser-issued cookies
    (mid / csrftoken / ig_nrcb) through the given proxy. Best-effort.

    When `clear_cookies=True`, wipe the jar first so retry attempts get a
    pristine cookie set issued by the current (rotated) egress IP, not a
    soft-blocked fingerprint left over from a prior failed attempt."""
    try:
        session = await _get_session()
        if clear_cookies and session.cookie_jar is not None:
            session.cookie_jar.clear()
        async with session.get(
            WARMUP_URL,
            proxy=proxy,
            timeout=aiohttp.ClientTimeout(total=15),
            allow_redirects=True,
        ):
            pass
    except Exception:
        pass


async def _download_avatar(url: str) -> bytes | None:
    if not url:
        return None
    if url in _avatar_url_cache:
        return _avatar_url_cache[url]
    try:
        session = await _get_session()
        async with session.get(
            url,
            proxy=_next_proxy(),
            headers={"User-Agent": USER_AGENT},
            timeout=aiohttp.ClientTimeout(total=10),
        ) as resp:
            if resp.status == 200:
                data = await resp.read()
                if data and len(data) > 100:
                    _avatar_url_cache[url] = data
                    return data
    except Exception:
        pass
    _avatar_url_cache[url] = None
    return None


async def _fetch_profile(
    username: str, proxy: str | None, clear_cookies: bool = False
) -> tuple[int | None, str | None, dict | None]:
    """One GraphQL attempt. Returns (status_code, response_text, parsed_json).
    On connection error returns (None, None, None)."""
    try:
        session = await _get_session()
        await _warmup(proxy, clear_cookies=clear_cookies)
        async with session.get(
            GRAPHQL_URL,
            params={"username": username},
            proxy=proxy,
            timeout=aiohttp.ClientTimeout(total=15),
        ) as resp:
            text = await resp.text()
            data = None
            if resp.status == 200:
                try:
                    data = json.loads(text)
                except Exception:
                    data = None
            return resp.status, text, data
    except Exception:
        return None, None, None


async def check_instagram_account(
    username: str, output_path: str, cached_sig: str | None = None
) -> dict:
    """
    Fetch profile status + data from Instagram's GraphQL API.

    If cached_sig matches the newly computed profile signature AND the PNG
    already exists, the card render is skipped (byte-identical output).

    Returns a dict:
      {
        "status": "active" | "banned" | "rate_limited" | "unknown",
        "image": path-or-None,
        "followers": int-or-None,
        "following": int-or-None,
        "posts": int-or-None,
        "full_name": str-or-None,
        "bio": str-or-None,
        "avatar_bytes": bytes-or-None,
        "avatar_url": str-or-None,
        "profile_sig": str-or-None,
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
        "avatar_url": None,
        "profile_sig": None,
    }

    # Attempt count: with rotating residential proxies the same proxy URL
    # hands out a fresh egress IP per request, so retrying across attempts
    # re-rolls the IP. Cap at 5 so a stuck proxy doesn't burn all bandwidth.
    if PROXIES:
        attempts = min(max(len(PROXIES), 3), 5)
    else:
        attempts = 1

    data: dict | None = None
    final_status_code: int | None = None
    final_text: str | None = None
    last_was_401_login = False

    for attempt in range(attempts):
        proxy = _next_proxy() if PROXIES else None
        code, text, parsed = await _fetch_profile(
            username, proxy, clear_cookies=attempt > 0
        )
        if code is None:
            # Connection error through this proxy → try next proxy.
            continue
        final_status_code = code
        final_text = text
        # Retryable rate-limit signals: 429, 401 require_login. Try next proxy.
        if code == 429 and attempt < attempts - 1:
            await asyncio.sleep(1)
            continue
        if code == 401 and "require_login" in (text or ""):
            last_was_401_login = True
            if attempt < attempts - 1:
                await asyncio.sleep(1)
                continue
        if code == 200 and parsed is not None:
            data = parsed
        break

    if data is None:
        if final_status_code == 404:
            result["status"] = "banned"
            new_sig = compute_profile_sig("banned", None, None, None, None, None, None)
            result["profile_sig"] = new_sig
            if cached_sig and cached_sig == new_sig and os.path.exists(output_path):
                result["image"] = output_path
            else:
                try:
                    render_profile_card(
                        username=username,
                        output_path=output_path,
                        status="banned",
                    )
                    result["image"] = output_path
                except Exception:
                    pass
            return result
        if final_status_code in (429, 401) and (
            final_status_code == 429
            or last_was_401_login
            or "require_login" in (final_text or "")
        ):
            result["status"] = "rate_limited"
            return result
        if final_status_code is not None:
            result["status"] = "unknown"
            return result
        # No proxy/connection succeeded at all
        result["status"] = "rate_limited"
        return result

    user = (data.get("data") or {}).get("user")
    if not user:
        result["status"] = "unknown"
        return result

    result["status"] = "active"
    result["full_name"] = user.get("full_name") or None
    result["bio"] = user.get("biography") or None

    followers = (user.get("edge_followed_by") or {}).get("count")
    following = (user.get("edge_follow") or {}).get("count")
    posts = (user.get("edge_owner_to_timeline_media") or {}).get("count")
    result["followers"] = int(followers) if isinstance(followers, int) else None
    result["following"] = int(following) if isinstance(following, int) else None
    result["posts"] = int(posts) if isinstance(posts, int) else None

    avatar_url = user.get("profile_pic_url_hd") or user.get("profile_pic_url")
    result["avatar_url"] = avatar_url

    new_sig = compute_profile_sig(
        result["status"],
        result["followers"],
        result["following"],
        result["posts"],
        result["full_name"],
        result["bio"],
        avatar_url,
    )
    result["profile_sig"] = new_sig

    # Card cache: skip render if profile inputs are unchanged and PNG exists
    if cached_sig and cached_sig == new_sig and os.path.exists(output_path):
        result["image"] = output_path
    else:
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
        except Exception:
            result["image"] = None

    return result


async def close_session() -> None:
    global _session
    if _session is not None and not _session.closed:
        await _session.close()
        _session = None
