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

PROXY USAGE: to save proxy bandwidth, every check tries Instagram DIRECTLY
first. Only if that direct attempt fails with a transport-level / IP-block
signal (timeout, HTTP 403/429, login wall, empty page, network error) does
the bot retry that single check through the proxy. Once a username has been
flagged as "needs proxy", subsequent checks for that user go straight to
proxy for PROXY_CACHE_TTL_SECONDS (default 24h), so we don't burn time on
doomed direct attempts. Set FORCE_PROXY=true in .env to bypass the smart
fallback (always use proxy) -- useful for debugging.
"""

import json
import os
import random
import re
import ssl
import time

import aiohttp
from dotenv import load_dotenv
from playwright.async_api import async_playwright

load_dotenv()

PROXY_HOST = os.getenv("PROXY_HOST")
PROXY_PORT = os.getenv("PROXY_PORT")
PROXY_USER = os.getenv("PROXY_USER")
PROXY_PASS = os.getenv("PROXY_PASS")

FORCE_PROXY = os.getenv("FORCE_PROXY", "").lower() in ("1", "true", "yes")

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

# Substrings (lowercased) that indicate the direct request was BLOCKED, not
# that the account genuinely doesn't exist. If any of these show up in the
# page title or body when og:meta is also missing, that's a soft-block /
# login-wall and we should fall back to the proxy.
DIRECT_BLOCK_MARKERS = [
    "log in",
    "sign up",
    "please wait",
    "rate limit",
    "suspicious login",
    "suspicious activity",
    "temporarily blocked",
    "try again later",
    "your request was blocked",
]

PROXY_CACHE_TTL_SECONDS = 24 * 3600  # 24 hours

# username -> expires_at_epoch (time.time())
_proxy_cache: dict[str, float] = {}


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


def _proxy_config() -> dict | None:
    """Return Playwright proxy kwargs, or None if proxy isn't configured."""
    if not PROXY_HOST or not PROXY_PORT:
        return None
    cfg = {"server": f"http://{PROXY_HOST}:{PROXY_PORT}"}
    if PROXY_USER and PROXY_PASS:
        cfg["username"] = PROXY_USER
        cfg["password"] = PROXY_PASS
    return cfg


def _should_use_proxy(username: str) -> bool:
    """Decide whether to skip the direct attempt and go straight to proxy."""
    if FORCE_PROXY:
        return True
    expires = _proxy_cache.get(username.lower())
    if expires is None:
        return False
    if time.time() >= expires:
        # Expired -- drop it and treat as needing a fresh direct attempt.
        _proxy_cache.pop(username.lower(), None)
        return False
    return True


def _mark_needs_proxy(username: str) -> None:
    """Cache this user as needing proxy for the next 24h."""
    _proxy_cache[username.lower()] = time.time() + PROXY_CACHE_TTL_SECONDS


async def _download_avatar(url: str, use_proxy: bool = False) -> bytes | None:
    if not url:
        return None
    proxy_url = None
    if use_proxy and PROXY_HOST and PROXY_PORT:
        if PROXY_USER and PROXY_PASS:
            proxy_url = f"http://{PROXY_USER}:{PROXY_PASS}@{PROXY_HOST}:{PROXY_PORT}"
        else:
            proxy_url = f"http://{PROXY_HOST}:{PROXY_PORT}"

    for attempt_ssl in (_AIOHTTP_SSL, None):
        try:
            kwargs = dict(
                timeout=aiohttp.ClientTimeout(total=10),
                headers={"User-Agent": USER_AGENT},
            )
            if attempt_ssl is not None:
                kwargs["ssl"] = attempt_ssl
            if proxy_url:
                kwargs["proxy"] = proxy_url
            async with aiohttp.ClientSession() as session:
                async with session.get(url, **kwargs) as resp:
                    if resp.status == 200:
                        data = await resp.read()
                        if data and len(data) > 100:
                            return data
        except Exception:
            continue
    return None


async def _run_check(
    username: str,
    url: str,
    use_proxy: bool,
) -> tuple[dict, bool]:
    """
    Run a single browser session against Instagram for `username`.

    Returns (result_dict, direct_failed_bool).
      - result_dict has status/image/followers/... filled in from this attempt.
      - direct_failed_bool is True iff this attempt hit a transport-level
        / IP-block signal (timeout, network error, login wall, 403/429,
        empty body with block markers). Caller uses it to decide whether to
        fall back to the other path.

    Note: result_dict['image'] is NOT populated here -- caller renders the
    final card after choosing which attempt's data to keep.
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
    direct_failed = False
    og_image = None
    failure_reason = ""

    proxy_cfg = _proxy_config() if use_proxy else None
    if use_proxy and proxy_cfg:
        print(f"  [proxy] using {PROXY_HOST}:{PROXY_PORT}")

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
        if proxy_cfg:
            context_kwargs["proxy"] = proxy_cfg
        context = await browser.new_context(**context_kwargs)
        page = await context.new_page()
        try:
            # Random delay to avoid rate-limit/bot patterns
            await page.wait_for_timeout(random.randint(300, 1500))
            # Navigate with a referer to look like organic traffic
            try:
                await page.goto(
                    "https://www.instagram.com/",
                    wait_until="domcontentloaded",
                    timeout=25000,
                )
            except Exception as e:
                failure_reason = f"homepage navigation: {type(e).__name__}"
                raise
            await page.wait_for_timeout(random.randint(800, 2000))
            response = None
            try:
                response = await page.goto(
                    url, wait_until="domcontentloaded", timeout=25000
                )
            except Exception as e:
                failure_reason = f"profile navigation: {type(e).__name__}"
                raise

            # HTTP-level block (e.g. 403/429 from Instagram's edge)
            if response is not None and response.status in (403, 429):
                failure_reason = f"HTTP {response.status}"
                direct_failed = True
                print(
                    f"  [proxy] @{username} direct attempt returned "
                    f"{response.status}, treating as blocked"
                )

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
                # Distinguish "soft-blocked" from "real unknown" by checking for
                # block markers in the body. This is what triggers proxy fallback.
                is_blocked = any(m in haystack for m in DIRECT_BLOCK_MARKERS) or (
                    response is not None and response.status in (403, 429)
                )
                if is_blocked:
                    direct_failed = True
                    failure_reason = (
                        failure_reason or "block markers in body, no og:meta"
                    )
                    print(
                        f"  [proxy] @{username} direct attempt looks "
                        f"blocked ({failure_reason})"
                    )
                else:
                    # Log what we saw so we can diagnose why it's unknown
                    snippet = (
                        body_text[:200].replace("\n", " ") if body_text else "<empty>"
                    )
                    print(
                        f"  [debug] @{username}: title='{title[:100]}' "
                        f"meta_desc={'yes' if meta_desc else 'no'} "
                        f"meta_title={'yes' if meta_title else 'no'} "
                        f"body_snippet='{snippet}'"
                    )
                    result["status"] = "unknown"

        except Exception as e:
            print(f"Error loading page for @{username}: {e!r}")
            if not failure_reason:
                failure_reason = f"{type(e).__name__}"
            direct_failed = True
            result["status"] = "unknown"
            try:
                await browser.close()
            except Exception:
                pass
            return result, direct_failed

        try:
            await browser.close()
        except Exception:
            pass

    # -- Download avatar (outside browser context, using aiohttp) ---------------
    # Always skip the avatar download if this attempt was a transport failure,
    # so we don't burn bandwidth fetching an image we're about to discard.
    if og_image and not direct_failed:
        result["avatar_bytes"] = await _download_avatar(og_image, use_proxy=use_proxy)

    return result, direct_failed


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

    Strategy: try Instagram DIRECTLY first. Only on transport-level failure
    (timeout, 403/429, login wall, network error) do we retry via the proxy.
    A successful direct attempt never touches the proxy -- this is the main
    bandwidth saving. If a direct failure occurs, the username is cached
    for 24h so the next check goes straight to proxy.
    """
    url = f"https://www.instagram.com/{username}/"

    # Decide which path to take. If the cache says "this user needs proxy",
    # skip the direct attempt entirely and save both time and bandwidth.
    use_proxy_first = _should_use_proxy(username)
    if use_proxy_first:
        print(f"  [proxy] @{username} cached as needing proxy, skipping direct attempt")

    result: dict | None = None
    if not use_proxy_first:
        try:
            result, direct_failed = await _run_check(username, url, use_proxy=False)
        except Exception as e:
            print(f"  [proxy] @{username} direct attempt raised: {e!r}")
            result, direct_failed = None, True

        if result is not None and not direct_failed:
            print(f"  [proxy] @{username} direct attempt succeeded, no proxy used")
        elif not _proxy_config():
            # No proxy configured at all -- nothing to fall back to. Return
            # whatever we got (likely status='unknown') so caller can log it.
            print(
                f"  [proxy] @{username} direct attempt failed but no proxy "
                f"is configured; returning result as-is"
            )
        else:
            # Cache this user as needing proxy, then fall through to retry.
            _mark_needs_proxy(username)
            print(f"  [proxy] @{username} direct attempt failed; retrying via proxy")
            result = None  # force proxy path below

    if result is None:
        try:
            result, _ = await _run_check(username, url, use_proxy=True)
        except Exception as e:
            print(f"  [proxy] @{username} proxy attempt raised: {e!r}")
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
