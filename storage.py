"""
Simple JSON-file backed storage for watch entries.

A "watch" is one row of: (guild, channel, user, instagram username, watch_type, active)
watch_type is either "banned" (notify when account becomes banned/unavailable)
or "unbanned" (notify when account comes back online).

This is intentionally file-based (no external database) so the bot is easy
to host anywhere with zero setup. An asyncio.Lock serializes access since
discord.py callbacks and the background task can run concurrently.
"""

import json
import os
import time
import asyncio


class WatchStore:
    def __init__(self, path: str):
        self.path = path
        self.lock = asyncio.Lock()
        if not os.path.exists(path):
            self._write({"next_id": 1, "watches": []})

    # -- low level helpers -------------------------------------------------
    def _read(self):
        with open(self.path, "r") as f:
            return json.load(f)

    def _write(self, data):
        """Atomic write: tmp file + os.replace so a crash mid-write
        doesn't corrupt the watchlist."""
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, self.path)

    # -- public API -----------------------------------------------------
    async def add_watch(self, guild_id, channel_id, user_id, username, watch_type):
        """Returns False if an identical active watch already exists."""
        async with self.lock:
            data = self._read()
            for w in data["watches"]:
                if (
                    w["guild_id"] == guild_id
                    and w["username"].lower() == username.lower()
                    and w["watch_type"] == watch_type
                    and w["active"]
                ):
                    return False
            watch = {
                "id": data["next_id"],
                "guild_id": guild_id,
                "channel_id": channel_id,
                "user_id": user_id,
                "username": username,
                "watch_type": watch_type,
                "active": True,
                "created_at": time.time(),
            }
            data["watches"].append(watch)
            data["next_id"] += 1
            self._write(data)
            return True

    async def list_watches(self, guild_id):
        async with self.lock:
            data = self._read()
            return [
                w for w in data["watches"] if w["guild_id"] == guild_id and w["active"]
            ]

    async def get_active_watches(self):
        async with self.lock:
            data = self._read()
            return [w for w in data["watches"] if w["active"]]

    async def get_active_watches_grouped_by_username(self) -> dict[str, list[dict]]:
        """Return active watches grouped by lowercase username.
        Allows the periodic loop to probe each unique account once and
        fan-out notifications to all watches for that account."""
        async with self.lock:
            data = self._read()
            grouped: dict[str, list[dict]] = {}
            for w in data["watches"]:
                if not w["active"]:
                    continue
                key = w["username"].lower()
                grouped.setdefault(key, []).append(w)
            return grouped

    async def deactivate(self, watch_id):
        async with self.lock:
            data = self._read()
            for w in data["watches"]:
                if w["id"] == watch_id:
                    w["active"] = False
            self._write(data)

    async def stop_all(self, guild_id):
        async with self.lock:
            data = self._read()
            stopped = 0
            for w in data["watches"]:
                if w["guild_id"] == guild_id and w["active"]:
                    w["active"] = False
                    stopped += 1
            self._write(data)
            return stopped

    async def delete_watch(self, watch_id):
        """Remove a watch entirely (not just deactivate)."""
        async with self.lock:
            data = self._read()
            before = len(data["watches"])
            data["watches"] = [w for w in data["watches"] if w["id"] != watch_id]
            removed = before - len(data["watches"])
            if removed:
                self._write(data)
            return removed

    async def cleanup_dead_watches(self, channel_ids: set[int]):
        """Delete all watches whose channel_id is in the given set (unreachable)."""
        async with self.lock:
            data = self._read()
            before = len(data["watches"])
            data["watches"] = [
                w for w in data["watches"] if w["channel_id"] not in channel_ids
            ]
            removed = before - len(data["watches"])
            if removed:
                self._write(data)
            return removed
