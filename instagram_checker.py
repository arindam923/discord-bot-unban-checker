"""
Checks an Instagram profile, extracts metadata via og:meta tags (no
screenshots), and renders a styled profile card image.

The headless browser is used only for page loading and status detection.
Profile data (name, avatar URL, follower/post counts, bio) comes from
Instagram's server-rendered og:meta tags, which are always present -- even
when a login wall popup covers the visible page.  This means the extracted
data is reliable regardless of overlay popups.

The avatar image is downloaded separately via aiohttp from the og:image URL
and passed to card_renderer.render_profile_card to produce the final PNG.
"""

import json
import os
import random
import re
import ssl

import aiohttp
from dotenv import load_dotenv
from playwright.async_api import async_playwright

load_dotenv()

PROXY_HOST = os.getenv("PROXY_HOST")
PROXY_PORT = os.getenv("PROXY_PORT")
PROXY_USER = os.getenv("PROXY_USER")
PROXY_PASS = os.getenv("PROXY_PASS")

try:
    import certifi

    _AIOHTTP_SSL = ssl.create_default_context(cafile=certifi.where())
except Exception:
    _AIOHTTP_SSL = None  # fall back to default (system) CA bundle

from card_renderer import render_profile_card

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

VIEWPORT = {"width": 1100, "height": 900}

NOT_FOUND_MARKERS = [
    "isn't available",
    "may be broken",
    "may have been removed",
    "page not found",
    "user not found",
]

FOLLOWERS_RE = re.compile(r"([\d,]+(?:[.,]\d+)?[KkMm]?)\s+Followers", re.IGNORECASE)
FOLLOWING_RE = re.compile(r"([\d,]+)\s+Following", re.IGNORECASE)
POSTS_RE = re.compile(r"([\d,]+)\s+Posts", re.IGNORECASE)


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


TITLE_NAME_RE = re.compile(r"^(.+?)\s*\(@")


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
        return None
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url,
                timeout=aiohttp.ClientTimeout(total=10),
                headers={"User-Agent": USER_AGENT},
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

    url = f"https://www.instagram.com/{username}/"
    og_image = None

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-dev-shm-usage",
            ],
        )
        context_kwargs = {
            "user_agent": USER_AGENT,
            "viewport": VIEWPORT,
            "locale": "en-US",
            "timezone_id": "America/New_York",
            "geolocation": {"latitude": 40.7128, "longitude": -74.0060},
            "permissions": ["geolocation"],
        }
        if PROXY_HOST and PROXY_PORT:
            proxy_cfg = {"server": f"http://{PROXY_HOST}:{PROXY_PORT}"}
            if PROXY_USER and PROXY_PASS:
                proxy_cfg["username"] = PROXY_USER
                proxy_cfg["password"] = PROXY_PASS
            context_kwargs["proxy"] = proxy_cfg
            print(f"  [proxy] using {PROXY_HOST}:{PROXY_PORT}")
        context = await browser.new_context(**context_kwargs)
        page = await context.new_page()
        try:
            # Random delay to avoid rate-limit/bot patterns
            await page.wait_for_timeout(random.randint(300, 1500))
            # Navigate with a referer to look like organic traffic
            await page.goto(
                "https://www.instagram.com/",
                wait_until="domcontentloaded",
                timeout=25000,
            )
            await page.wait_for_timeout(random.randint(800, 2000))
            await page.goto(url, wait_until="domcontentloaded", timeout=25000)
            # Wait for Instagram's React hydration to populate og:meta tags
            try:
                await page.wait_for_selector('meta[property="og:title"]', timeout=6000)
            except Exception:
                pass
            await page.wait_for_timeout(2000)

            title = (await page.title()).lower()
            try:
                body_text = (await page.inner_text("body", timeout=5000)).lower()
            except Exception:
                body_text = ""

            # -- og:meta extraction with short timeouts ---------------------------
            async def _safe_meta(selector, attr, timeout_ms=4000):
                try:
                    loc = page.locator(selector)
                    if await loc.count() == 0:
                        return None
                    return await loc.first.get_attribute(attr, timeout=timeout_ms)
                except Exception:
                    return None

            meta_desc = await _safe_meta('meta[property="og:description"]', "content")
            meta_title = await _safe_meta('meta[property="og:title"]', "content")
            og_image = await _safe_meta('meta[property="og:image"]', "content")

            # -- Parse counts from og:description --------------------------------
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

            # -- Parse full_name from og:title -----------------------------------
            # Typical: "Full Name (@username) • Instagram"
            if meta_title:
                m = TITLE_NAME_RE.match(meta_title)
                if m:
                    candidate = m.group(1).strip()
                    if candidate.lower() != username.lower():
                        result["full_name"] = candidate

            # -- Try to extract bio from ld+json or page -------------------------
            try:
                ld_nodes = page.locator('script[type="application/ld+json"]')
                count = await ld_nodes.count()
                for i in range(min(count, 3)):  # bound the loop
                    try:
                        raw = await ld_nodes.nth(i).inner_text(timeout=2000)
                        data = json.loads(raw)
                        desc = data.get("description", "")
                        if desc and len(desc) > 0:
                            result["bio"] = desc.strip()
                            break
                    except (json.JSONDecodeError, AttributeError):
                        continue
                    except Exception:
                        break
            except Exception:
                pass

            # -- Classify status --------------------------------------------------
            # IMPORTANT: detect "banned" FIRST, because Instagram's removed-profile
            # page has title "Profile isn't available" but no og:meta tags.
            haystack = f"{title} {body_text}"
            if any(marker in haystack for marker in NOT_FOUND_MARKERS):
                result["status"] = "banned"
            elif result["followers"] is not None or (
                meta_title and username.lower() in meta_title.lower()
            ):
                result["status"] = "active"
            elif username.lower() in title:
                # Page title mentions the username (e.g. "@username on Instagram")
                # but no meta tags were found — likely active but page loaded oddly.
                result["status"] = "active"
            else:
                # Log what we saw so we can diagnose why it's unknown
                snippet = body_text[:200].replace("\n", " ") if body_text else "<empty>"
                print(
                    f"  [debug] @{username}: title='{title[:100]}' "
                    f"meta_desc={'yes' if meta_desc else 'no'} "
                    f"meta_title={'yes' if meta_title else 'no'} "
                    f"body_snippet='{snippet}'"
                )
                result["status"] = "unknown"

        except Exception as e:
            print(f"Error loading page for @{username}: {e!r}")
            try:
                await browser.close()
            except Exception:
                pass

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

        try:
            await browser.close()
        except Exception:
            pass

    # -- Download avatar (outside browser context, using aiohttp) ---------------
    if og_image:
        result["avatar_bytes"] = await _download_avatar(og_image)

    # -- Render the profile card ------------------------------------------------
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
