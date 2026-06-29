"""
Fetches Instagram profile data via the socialyze.io API.

The bot used to call Instagram's own GraphQL endpoint directly with rotating
residential proxies, which got expensive. socialyze.io runs the scraper on
their side and exposes a small JSON API the bot can call directly (no proxy
needed, no per-IP rate limiting).

  Base URL:  https://socialyze.io/api/v1/trackers/instagram-followers
  Auth:      X-API-Key + X-API-Secret headers
  GET        -> {"data": [Tracker, ...]}     # all trackers on the account
  POST       -> {"success": true, "data": Tracker}  # add a tracker

Each Tracker payload:
  {
    "id":                "cmqyukttf014voqnbmrvxda3m",
    "username":          "nasa",
    "fullName":          "NASA" | null,
    "profilePicUrl":     "https://scontent-...cdninstagram.com/...jpg" | null,
    "isVerified":        bool,
    "isPrivate":         bool,
    "currentFollowers":  int,
    "currentFollowing":  int,
    "currentPosts":      int,
    "dailyChange":       int,
    "weeklyChange":      int,
    "monthlyChange":     int,
    "isTracking":        bool,
    "lastScrapedAt":     ISO8601 timestamp | null,   # null = never successfully scraped
    "snapshots":         [{...}, ...]
  }

Ban detection (heuristic — socialyze has no direct "is banned" flag):
  - "active"  -> lastScrapedAt is a non-null ISO timestamp
  - "banned"  -> lastScrapedAt is null AND it's been more than
                 SOCIALYZE_SCRAPE_GRACE_SECONDS since we POSTed the tracker
                 (banned/deleted accounts never get a successful scrape)
  - "unknown" -> lastScrapedAt is null but we're still inside the grace
                 window (initial scrape not done yet)
  Active -> banned transition: lastScrapedAt flipping from non-null to null
  on a tracker that we previously saw scraped = banned.

Avatar bytes are still fetched directly from the Instagram CDN (profilePicUrl).
The CDN is not behind Cloudflare 1010, so aiohttp works fine for that.

HTTP client:
  socialyze.io is behind Cloudflare with a browser-fingerprint check that
  blocks raw aiohttp / urllib (returns "error code: 1010"). We use
  curl_cffi's AsyncSession with `impersonate="chrome"` so the TLS handshake
  looks like a real browser. Avatars are fetched with aiohttp.
"""

import asyncio
import os
import time

import aiohttp
from curl_cffi.requests import AsyncSession
from dotenv import load_dotenv

from card_renderer import render_profile_card
from status_cache import compute_profile_sig

load_dotenv()

SOCIALYZE_BASE = "https://socialyze.io/api/v1/trackers/instagram-followers"
SOCIALYZE_API_KEY = os.getenv("SOCIALYZE_API_KEY", "")
SOCIALYZE_API_SECRET = os.getenv("SOCIALYZE_API_SECRET", "")

# Seconds to wait for socialyze's first scrape before declaring "banned" for
# a tracker that has never been successfully scraped. Default 5 min.
try:
    SOCIALYZE_SCRAPE_GRACE_SECONDS = float(
        os.getenv("SOCIALYZE_SCRAPE_GRACE_SECONDS", "300")
    )
except ValueError:
    SOCIALYZE_SCRAPE_GRACE_SECONDS = 300.0

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

# Shared sessions (lazy-init, live for bot lifetime)
_socialyze_session: AsyncSession | None = None
_avatar_session: aiohttp.ClientSession | None = None

# In-memory avatar cache: url -> bytes (or None if download failed).
_avatar_url_cache: dict[str, bytes | None] = {}


def _socialyze_headers() -> dict[str, str]:
    return {
        "X-API-Key": SOCIALYZE_API_KEY,
        "X-API-Secret": SOCIALYZE_API_SECRET,
        "Accept": "application/json",
        "Content-Type": "application/json",
    }


async def _get_socialyze_session() -> AsyncSession:
    global _socialyze_session
    if _socialyze_session is None:
        _socialyze_session = AsyncSession(
            impersonate="chrome",
            headers=_socialyze_headers(),
            timeout=20,
        )
    return _socialyze_session


async def _get_avatar_session() -> aiohttp.ClientSession:
    global _avatar_session
    if _avatar_session is None or _avatar_session.closed:
        _avatar_session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=15),
            headers={"User-Agent": USER_AGENT},
        )
    return _avatar_session


async def fetch_all_trackers() -> list[dict]:
    """GET the full list of trackers on this socialyze account.
    Returns a (possibly empty) list. Raises on transport errors so callers
    can surface them as transient 'unknown' rather than caching bad data."""
    session = await _get_socialyze_session()
    r = await session.get(SOCIALYZE_BASE)
    if r.status_code != 200:
        raise RuntimeError(
            f"socialyze list returned HTTP {r.status_code}: {r.text[:200]}"
        )
    payload = r.json()
    if not isinstance(payload, dict) or "data" not in payload:
        raise RuntimeError(f"socialyze list returned unexpected payload: {payload!r}")
    data = payload.get("data") or []
    if not isinstance(data, list):
        raise RuntimeError(f"socialyze list data is not a list: {data!r}")
    return data


def find_tracker_by_username(trackers: list[dict], username: str) -> dict | None:
    """Linear scan — list is small (tens of items per account). Case-insensitive."""
    target = username.strip().lstrip("@").lower()
    for t in trackers:
        if (t.get("username") or "").lower() == target:
            return t
    return None


async def add_tracker(username: str) -> dict:
    """POST a new tracker to socialyze. Returns the created Tracker dict.
    Raises on transport errors or non-2xx responses. Safe to call for an
    already-tracked username — socialyze returns the existing tracker and
    re-scrapes it (a small wasted-scrape cost, not a quota concern)."""
    clean = username.strip().lstrip("@")
    session = await _get_socialyze_session()
    r = await session.post(
        SOCIALYZE_BASE,
        json={"url": f"https://www.instagram.com/{clean}/"},
    )
    if r.status_code != 200:
        raise RuntimeError(
            f"socialyze add returned HTTP {r.status_code}: {r.text[:200]}"
        )
    payload = r.json()
    if not payload.get("success") or not isinstance(payload.get("data"), dict):
        raise RuntimeError(f"socialyze add returned unexpected payload: {payload!r}")
    return payload["data"]


async def _download_avatar(url: str) -> bytes | None:
    if not url:
        return None
    if url in _avatar_url_cache:
        return _avatar_url_cache[url]
    try:
        session = await _get_avatar_session()
        async with session.get(url) as resp:
            if resp.status == 200:
                data = await resp.read()
                if data and len(data) > 100:
                    _avatar_url_cache[url] = data
                    return data
    except Exception:
        pass
    _avatar_url_cache[url] = None
    return None


def _is_scraped(tracker: dict) -> bool:
    """Has this tracker been successfully scraped at least once?"""
    return bool(tracker.get("lastScrapedAt"))


def _looks_banned(tracker: dict) -> bool:
    """Soft-banned heuristic for the case where socialyze successfully
    scrapes a tracker but the underlying Instagram profile is gone.

    socialyze returns a non-null lastScrapedAt even for nonexistent users,
    but the profile fields come back empty/default:
        - fullName == username
        - profilePicUrl is null
        - currentFollowers / currentFollowing / currentPosts are all 0
    A real account — even a brand new one — will have a profile pic and
    a full name that differs from the handle. The (fullName == username)
    check is the strongest of these, but requiring all four together
    keeps the false-positive rate near zero.
    """
    if not tracker:
        return False
    username = (tracker.get("username") or "").strip().lower()
    full_name = (tracker.get("fullName") or "").strip().lower()
    if not username or full_name != username:
        return False
    if tracker.get("profilePicUrl"):
        return False
    if (tracker.get("currentFollowers") or 0) != 0:
        return False
    if (tracker.get("currentFollowing") or 0) != 0:
        return False
    if (tracker.get("currentPosts") or 0) != 0:
        return False
    return True


def _derive_status(
    tracker: dict | None,
    socialyze_added_at: float | None,
    prev_last_scraped_at: str | None,
) -> str:
    """Map a tracker + local state to one of active/banned/unknown."""
    if tracker is None:
        # No tracker yet. If we've been waiting past the grace period, this
        # is a banned/deleted account that socialyze can't even create for.
        # Inside the grace window: unknown.
        if socialyze_added_at is None:
            return "unknown"
        elapsed = time.time() - socialyze_added_at
        if elapsed >= SOCIALYZE_SCRAPE_GRACE_SECONDS:
            return "banned"
        return "unknown"

    # Signal 1: profile data looks empty/default → soft-banned.
    if _looks_banned(tracker):
        return "banned"

    last_scraped = tracker.get("lastScrapedAt")
    if last_scraped:
        return "active"

    # lastScrapedAt is null. Was it previously scraped (active -> banned)?
    if prev_last_scraped_at:
        return "banned"
    # Never scraped since we added it — inside grace = unknown, past = banned.
    if socialyze_added_at is None:
        return "unknown"
    elapsed = time.time() - socialyze_added_at
    if elapsed >= SOCIALYZE_SCRAPE_GRACE_SECONDS:
        return "banned"
    return "unknown"


async def check_instagram_account(
    username: str,
    output_path: str,
    cached_sig: str | None = None,
    cached_state: dict | None = None,
    trackers_cache: list[dict] | None = None,
) -> dict:
    """
    Look up the username in socialyze and produce a status + profile payload.

    cached_state (optional):
        Previous state dict from status_cache, carrying:
          - socialyze_id: str | None
          - socialyze_added_at: float | None
          - last_known_lastScrapedAt: str | None
          - confirmed: str | None  (previous status)
        Used to detect active -> banned transitions and to honour the grace
        window for never-scraped trackers.

    trackers_cache (optional):
        Pre-fetched list from fetch_all_trackers(). When None, we fetch it
        on demand. Passing it in lets the periodic loop batch one GET per
        tick across many accounts instead of one GET per account.

    Returns a dict shaped exactly like the old GraphQL version so the rest
    of the bot (card renderer, embeds, status cache) doesn't need to change:

      {
        "status":      "active" | "banned" | "unknown",
        "image":       path-or-None,
        "followers":   int-or-None,
        "following":   int-or-None,
        "posts":       int-or-None,
        "full_name":   str-or-None,
        "bio":         None,                       # socialyze doesn't expose bio
        "avatar_bytes":bytes-or-None,
        "avatar_url":  str-or-None,                # = profilePicUrl
        "profile_sig": str-or-None,
        "_socialyze_id":             str-or-None,  # caller persists this
        "_socialyze_added_at":       float-or-None,
        "_last_known_lastScrapedAt": str-or-None,
      }
    """
    result: dict = {
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
        "_socialyze_id": None,
        "_socialyze_added_at": None,
        "_last_known_lastScrapedAt": None,
    }

    cached_state = cached_state or {}
    socialyze_id = cached_state.get("socialyze_id")
    socialyze_added_at = cached_state.get("socialyze_added_at")
    prev_last_scraped_at = cached_state.get("last_known_lastScrapedAt")

    if trackers_cache is None:
        try:
            trackers_cache = await fetch_all_trackers()
        except Exception:
            return result  # transient error → unknown

    tracker = find_tracker_by_username(trackers_cache, username)

    # If the tracker is gone from socialyze (deleted by admin / never created /
    # lost after a quota reset) AND we have a prior id we know about, that's
    # a hard "banned" — but only if we've given socialyze time to come back.
    # Inside grace window, treat as unknown (just slow to appear in the list).
    if tracker is None and socialyze_id is None and socialyze_added_at is None:
        # First time we've ever seen this username, and the list doesn't
        # contain it. Try to add it.
        try:
            tracker = await add_tracker(username)
            socialyze_id = tracker.get("id")
            socialyze_added_at = time.time()
        except Exception:
            return result  # couldn't add right now → unknown

    elif tracker is None and socialyze_id is not None:
        # We had a tracker before but it's gone from the list. Re-add it.
        # socialyze dedupes by username, so this is safe and idempotent.
        try:
            tracker = await add_tracker(username)
            socialyze_id = tracker.get("id")
            # Don't reset socialyze_added_at — keep the original "first seen"
            # timestamp so the grace window reflects how long THIS account
            # has been unbannable, not how long since the re-add.
            if socialyze_added_at is None:
                socialyze_added_at = time.time()
        except Exception:
            # If we can't re-add, treat as banned only if we know the account
            # existed at some point and the prior lastScrapedAt was set.
            if prev_last_scraped_at:
                result["status"] = "banned"
            return result

    status = _derive_status(tracker, socialyze_added_at, prev_last_scraped_at)
    result["status"] = status

    # Persist bookkeeping for the caller.
    result["_socialyze_id"] = (
        tracker.get("id") if tracker else socialyze_id
    )
    result["_socialyze_added_at"] = socialyze_added_at
    result["_last_known_lastScrapedAt"] = (
        tracker.get("lastScrapedAt") if tracker else prev_last_scraped_at
    )

    if status != "active" or tracker is None:
        # For "banned" we still render a banned card (no profile data).
        if status == "banned":
            new_sig = compute_profile_sig(
                "banned", None, None, None, None, None, None
            )
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

    # Active: pull profile fields from the tracker.
    result["full_name"] = tracker.get("fullName") or None
    result["followers"] = (
        int(tracker["currentFollowers"])
        if isinstance(tracker.get("currentFollowers"), (int, float))
        else None
    )
    result["following"] = (
        int(tracker["currentFollowing"])
        if isinstance(tracker.get("currentFollowing"), (int, float))
        else None
    )
    result["posts"] = (
        int(tracker["currentPosts"])
        if isinstance(tracker.get("currentPosts"), (int, float))
        else None
    )
    avatar_url = tracker.get("profilePicUrl") or None
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
    global _socialyze_session, _avatar_session
    if _socialyze_session is not None:
        try:
            await _socialyze_session.close()
        except Exception:
            pass
        _socialyze_session = None
    if _avatar_session is not None and not _avatar_session.closed:
        try:
            await _avatar_session.close()
        except Exception:
            pass
        _avatar_session = None
