"""
Persistent per-username last-status cache.

Stores the last confirmed status for each watched username, so the periodic
loop can:
  1. Skip the card render when nothing has changed (profile_sig match)
  2. Skip the API call entirely when the account isn't due for its tier
     (active accounts checked every 5 min, banned every 1 min, etc.)

Schema (status_cache.json):
{
  "<username_lower>": {
    "confirmed":                 "active" | "banned" | null,
    "last_checked":              <unix_ts>,
    "last_seen":                 <unix_ts>,
    "profile_sig":               "<hash>" | null,
    "avatar_url":                "<url>"  | null,
    "retry_after":               <unix_ts> | null,
    "retry_count":               <int>     | null,

    # socialyze.io bookkeeping (replaces the old per-IP proxy rotation):
    "socialyze_id":              "<id>"    | null,
    "socialyze_added_at":        <unix_ts> | null,
    "last_known_lastScrapedAt":  "<iso8601>" | null
  }
}
"""

import hashlib
import json
import os
import time

import asyncio


def compute_profile_sig(
    status: str,
    followers,
    following,
    posts,
    full_name,
    bio,
    avatar_url,
) -> str:
    """Compute a stable signature for card-rendering inputs. If this matches
    the cached value, the card PNG doesn't need to be re-rendered."""
    parts = [
        str(x)
        for x in [status, followers, following, posts, full_name, bio, avatar_url]
    ]
    return hashlib.md5("|".join(parts).encode(), usedforsecurity=False).hexdigest()


class StatusCache:
    def __init__(self, path: str):
        self.path = path
        self.lock = asyncio.Lock()
        if not os.path.exists(path):
            self._write_atomic({})

    def _read(self) -> dict:
        try:
            with open(self.path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {}

    def _write_atomic(self, data: dict) -> None:
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, self.path)

    async def get(self, username: str) -> dict | None:
        async with self.lock:
            data = self._read()
            return data.get(username.lower())

    async def set(self, username: str, state: dict) -> None:
        async with self.lock:
            data = self._read()
            data[username.lower()] = state
            self._write_atomic(data)

    async def touch_many(self, usernames: list[str]) -> None:
        """Update last_seen for multiple usernames in one I/O cycle."""
        if not usernames:
            return
        async with self.lock:
            data = self._read()
            now = time.time()
            changed = False
            for username in usernames:
                key = username.lower()
                if key in data:
                    data[key]["last_seen"] = now
                    changed = True
            if changed:
                self._write_atomic(data)

    async def get_all(self) -> dict:
        async with self.lock:
            return self._read()

    async def get_due_usernames(
        self, usernames: list[str], intervals: dict[str, float]
    ) -> list[str]:
        """Return usernames that are due for a check based on their last
        known status and the per-status interval.

        intervals = {"active": 300, "banned": 60, "unknown": 120, None: 60}
        A username is "due" if:
          - it has no cached entry (never checked) → always due
          - time since last_checked >= interval for its current status
        """
        if not usernames:
            return []
        async with self.lock:
            data = self._read()
            now = time.time()
            due = []
            for username in usernames:
                key = username.lower()
                entry = data.get(key)
                # Skip if still inside rate-limit backoff window
                retry_after = entry.get("retry_after") if entry else None
                if retry_after is not None and now < retry_after:
                    continue
                if entry is None or entry.get("last_checked") is None:
                    due.append(username)
                    continue
                status = entry.get("confirmed")
                interval = intervals.get(status, intervals.get(None, 60))
                if now - entry["last_checked"] >= interval:
                    due.append(username)
            return due

    async def set_rate_limited(self, username: str) -> None:
        """Mark an account as rate-limited with exponential backoff.
        retry_after = now + 60s, 120s, 240s, ... up to 900s (15 min)."""
        async with self.lock:
            data = self._read()
            key = username.lower()
            entry = data.get(key, {})
            retry_count = entry.get("retry_count", 0) + 1
            delay = min(60 * (2 ** (retry_count - 1)), 900)
            entry["retry_after"] = time.time() + delay
            entry["retry_count"] = retry_count
            data[key] = entry
            self._write_atomic(data)

    async def clear_retry_backoff(self, username: str) -> None:
        """Reset retry backoff when account is successfully checked."""
        async with self.lock:
            data = self._read()
            key = username.lower()
            if key in data:
                data[key].pop("retry_after", None)
                data[key].pop("retry_count", None)
                self._write_atomic(data)

    async def delete(self, username: str) -> None:
        async with self.lock:
            data = self._read()
            key = username.lower()
            if key in data:
                del data[key]
                self._write_atomic(data)
