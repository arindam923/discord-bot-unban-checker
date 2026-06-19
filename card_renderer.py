"""
Composes an Instagram-style profile "share card" image (avatar + username +
Follow badge + stats + name, on a black canvas) instead of relying on a
literal page screenshot. This matches the layout/ratio of the reference
card: a 1200x512 canvas, circular avatar on the left vertically centered,
info stacked to its right, with slightly rounded corners on the whole card.

Rendering our own card (rather than screenshotting the live page) sidesteps
every problem we kept hitting with real screenshots: login-wall popups,
cookie banners, viewport sizing, and DOM elements changing shape. It's also
far cheaper to produce, which matters once checks run every few seconds.
"""

import io
import os
from PIL import Image, ImageDraw, ImageFont, ImageOps

CANVAS_W, CANVAS_H = 1200, 512
CORNER_RADIUS = 30

AVATAR_DIAMETER = 220
AVATAR_LEFT = 60
AVATAR_TOP = (CANVAS_H - AVATAR_DIAMETER) // 2  # vertically centered

TEXT_X = 340
TEXT_MAX_WIDTH = CANVAS_W - TEXT_X - 60

WHITE = (255, 255, 255, 255)
GRAY = (142, 142, 147, 255)
LIGHT_GRAY = (199, 199, 204, 255)
BLUE = (0, 149, 246, 255)
BLACK = (0, 0, 0, 255)
PLACEHOLDER_GRAY = (60, 60, 63)

# Search paths for system fonts (no external asset required).
_FONT_REGULAR_PATHS = [
    "/System/Library/Fonts/Helvetica.ttc",
    "/Library/Fonts/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "C:/Windows/Fonts/arial.ttf",
]
_FONT_BOLD_PATHS = [
    "/System/Library/Fonts/Helvetica.ttc",
    "/Library/Fonts/Arial Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
]


def _find_font(paths, bold=False):
    for path in paths:
        if not os.path.exists(path):
            continue
        try:
            if bold and path.lower().endswith(".ttc"):
                try:
                    return ImageFont.truetype(path, 28, index=1)
                except Exception:
                    pass
            return ImageFont.truetype(path, 28)
        except Exception:
            continue
    try:
        return ImageFont.load_default(size=28)
    except TypeError:
        return ImageFont.load_default()


def _font(bold=False, size=28):
    paths = _FONT_BOLD_PATHS if bold else _FONT_REGULAR_PATHS
    for path in paths:
        if not os.path.exists(path):
            continue
        try:
            if bold and path.lower().endswith(".ttc"):
                try:
                    return ImageFont.truetype(path, size, index=1)
                except Exception:
                    pass
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _circular_avatar(avatar_bytes, size):
    """RGBA circular avatar of the given size; falls back to a plain gray
    placeholder circle (matching Instagram's own default avatar look) if no
    image data is available or it fails to decode."""
    if avatar_bytes:
        try:
            img = Image.open(io.BytesIO(avatar_bytes)).convert("RGB")
            img = ImageOps.fit(img, (size, size), Image.LANCZOS)
        except Exception:
            img = Image.new("RGB", (size, size), PLACEHOLDER_GRAY)
    else:
        img = Image.new("RGB", (size, size), PLACEHOLDER_GRAY)

    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).ellipse((0, 0, size, size), fill=255)
    out = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    out.paste(img, (0, 0), mask)
    return out


def _format_count(n):
    """Mirrors Instagram's own UI abbreviation style: 27800 -> '27.8k'."""
    if n is None:
        return "—"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}m"
    if n >= 10_000:
        return f"{n / 1000:.1f}k"
    return f"{n:,}"


def _wrap_text(draw, text, font, max_width, max_lines=3):
    words = text.split()
    lines, line = [], ""
    for word in words:
        candidate = f"{line} {word}".strip()
        try:
            w = draw.textlength(candidate, font=font)
        except Exception:
            w = draw.textbbox((0, 0), candidate, font=font)[2]
        if w <= max_width:
            line = candidate
        else:
            if line:
                lines.append(line)
            line = word
        if len(lines) >= max_lines:
            break
    if line and len(lines) < max_lines:
        lines.append(line)
    return lines


def render_profile_card(
    username,
    output_path,
    full_name=None,
    bio=None,
    posts=None,
    followers=None,
    following=None,
    avatar_bytes=None,
    **kwargs,
):
    """Builds the share-card-style PNG and saves it to output_path. Returns
    output_path on success."""
    canvas = Image.new("RGBA", (CANVAS_W, CANVAS_H), (0, 0, 0, 0))
    mask = Image.new("L", (CANVAS_W, CANVAS_H), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        (0, 0, CANVAS_W, CANVAS_H), radius=CORNER_RADIUS, fill=255
    )
    bg = Image.new("RGBA", (CANVAS_W, CANVAS_H), BLACK)
    canvas.paste(bg, (0, 0), mask)

    draw = ImageDraw.Draw(canvas)

    avatar = _circular_avatar(avatar_bytes, AVATAR_DIAMETER)
    canvas.paste(avatar, (AVATAR_LEFT, AVATAR_TOP), avatar)

    username_font = _font(bold=True, size=42)
    badge_font = _font(bold=True, size=21)
    stats_num_font = _font(bold=True, size=28)
    stats_label_font = _font(bold=False, size=28)
    name_font = _font(bold=True, size=26)
    bio_font = _font(bold=False, size=24)

    row1_y = AVATAR_TOP + 18
    draw.text((TEXT_X, row1_y), username, font=username_font, fill=WHITE)
    try:
        uname_w = draw.textlength(username, font=username_font)
    except Exception:
        uname_w = draw.textbbox((0, 0), username, font=username_font)[2]

    badge_x = TEXT_X + uname_w + 24
    badge_y = row1_y + 10
    badge_w, badge_h = 100, 38
    draw.rounded_rectangle(
        (badge_x, badge_y, badge_x + badge_w, badge_y + badge_h),
        radius=8,
        fill=BLUE,
    )
    draw.text(
        (badge_x + badge_w / 2, badge_y + badge_h / 2),
        "Follow",
        font=badge_font,
        fill=WHITE,
        anchor="mm",
    )

    dots_x = badge_x + badge_w + 22
    draw.text(
        (dots_x, badge_y + badge_h / 2),
        "•••",
        font=badge_font,
        fill=LIGHT_GRAY,
        anchor="lm",
    )

    stats_y = row1_y + 70
    cursor_x = TEXT_X
    for value, label in (
        (_format_count(posts), "posts"),
        (_format_count(followers), "followers"),
        (_format_count(following), "following"),
    ):
        draw.text((cursor_x, stats_y), value, font=stats_num_font, fill=WHITE)
        try:
            cursor_x += draw.textlength(value, font=stats_num_font) + 8
        except Exception:
            cursor_x += draw.textbbox((0, 0), value, font=stats_num_font)[2] + 8
        draw.text((cursor_x, stats_y), label, font=stats_label_font, fill=GRAY)
        try:
            cursor_x += draw.textlength(label, font=stats_label_font) + 28
        except Exception:
            cursor_x += draw.textbbox((0, 0), label, font=stats_label_font)[2] + 28

    next_y = stats_y + 50
    if full_name:
        draw.text((TEXT_X, next_y), full_name, font=name_font, fill=WHITE)
        next_y += 36

    if bio:
        for line in _wrap_text(draw, bio, bio_font, TEXT_MAX_WIDTH):
            draw.text((TEXT_X, next_y), line, font=bio_font, fill=LIGHT_GRAY)
            next_y += 30

    canvas.save(output_path, "PNG")
    return output_path
