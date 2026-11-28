import bisect
import contextlib
import io
import math
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

BPM = 120
BEAT = 60.0 / BPM
FPS = 30
WIDTH = 1920
HEIGHT = 1080
RATIO = 2.35
CONTENT_H = int(round(WIDTH / RATIO / 2.0)) * 2
BAR_H = (HEIGHT - CONTENT_H) // 2
TOTAL_BEATS = 180
BITRATE = "12000k"
AUDIO_RATE = 44100

SOURCE_DIR = r"D:\Code\Video"
OUTPUT_DIR = r"D:\Code\Video"
OUTPUT_NAME = "Timeline_Fixed_Cinema.mp4"
WORK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_timeline_fixed")
MANIFEST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_manifest.json")

SCHOOL_EN = "HAIKOU JINPAN"
SCHOOL_CN = "海口市金盘实验学校"
CLASS_EN = "CLASS 20"
CLASS_CN = "七(20)班"
MOTTO = "NEVER GIVE UP"

FONT_CN = r"C:\Windows\Fonts\msyhbd.ttc"
FONT_HEAVY = r"C:\Windows\Fonts\impact.ttf"
FONT_SANS = r"C:\Windows\Fonts\arialbd.ttf"

TARGET_IMAGES = 40
TARGET_VIDEO_CLIPS = 10
HERO_COUNT = 14
PHASH_DUPLICATE = 12
PHASH_DIVERSITY = 18
HERO_W = 2208
HERO_H = 938
GRAIN_PLATES = 8
GRAIN_STRENGTH = 0.30
DEDUPE_WINDOW = 150.0
LIMIT_SECONDS = float(os.environ.get("EPIC_LIMIT", "0"))

CLOCK = {"stage": time.time()}


def log(message):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), message), flush=True)


def stage(title):
    print("", flush=True)
    print("-" * 72, flush=True)
    print(title, flush=True)
    print("-" * 72, flush=True)
    CLOCK["stage"] = time.time()


def stage_done(note):
    log("stage finished in %.1f s | %s" % (time.time() - CLOCK["stage"], note))


def resolve_ffmpeg():
    candidates = []
    configured = os.environ.get("FFMPEG_BINARY", "")
    if configured and configured not in ("ffmpeg-imageio", "auto-detect", "unset"):
        candidates.append(configured)
    try:
        import imageio_ffmpeg

        candidates.append(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:
        pass
    for name in ("ffmpeg.exe", "ffmpeg"):
        located = shutil.which(name)
        if located:
            candidates.append(located)
    for path in candidates:
        if path and os.path.isfile(path):
            return os.path.abspath(path)
    raise RuntimeError("ffmpeg not found")


FFMPEG = resolve_ffmpeg()
os.environ["FFMPEG_BINARY"] = FFMPEG
os.environ["IMAGEIO_FFMPEG_EXE"] = FFMPEG

from moviepy import AudioClip, VideoClip, VideoFileClip


def natural_key(name):
    chunks = re.split(r"(\d+)", os.path.basename(name))
    return [(0, int(chunk), "") if chunk.isdigit() else (1, 0, chunk.lower()) for chunk in chunks]


def media_timestamp(path):
    name = os.path.basename(path)
    match = re.search(r"(20\d{2})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})", name)
    if match:
        try:
            return datetime(*(int(value) for value in match.groups())), "filename"
        except ValueError:
            pass
    if name.lower().endswith(".mp4"):
        result = subprocess.run([FFMPEG, "-hide_banner", "-i", path], capture_output=True, text=True, errors="replace")
        match = re.search(r"creation_time\s*:\s*(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})", result.stderr)
        if match:
            return datetime.strptime(match.group(1), "%Y-%m-%dT%H:%M:%S") + timedelta(hours=8), "metadata"
    return datetime.fromtimestamp(os.path.getmtime(path)), "mtime"


def collect_media():
    found = []
    for entry in os.scandir(SOURCE_DIR):
        if not entry.is_file():
            continue
        name = entry.name
        lowered = name.lower()
        if not (lowered.endswith(".jpg") or lowered.endswith(".mp4")):
            continue
        if name.startswith(("_", "Final_", "Class20_", "Epic_", "Timeline_")):
            continue
        moment, origin = media_timestamp(entry.path)
        found.append(
            {
                "path": entry.path,
                "kind": "video" if lowered.endswith(".mp4") else "image",
                "stamp": moment,
                "origin": origin,
            }
        )
    found.sort(key=lambda item: (item["stamp"], item["path"]))
    return found


def decode_full(path):
    with Image.open(path) as handle:
        width, height = handle.size
    flag = cv2.IMREAD_COLOR if width * height <= 20000000 else cv2.IMREAD_REDUCED_COLOR_2
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), flag)
    if image is None:
        return None
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def decode_thumb(path, limit=900):
    image = decode_full(path)
    if image is None:
        return None
    height, width = image.shape[:2]
    scale = limit / float(max(height, width))
    if scale < 1.0:
        image = cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
    return image


def phash(image):
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    low = cv2.dct(small)[:8, :8]
    return (low > np.median(low[1:])).ravel().astype(np.uint8)


def hamming(first, second):
    return int(np.count_nonzero(first ^ np.asarray(second, dtype=np.uint8)))


def analyse_image(path):
    thumb = decode_thumb(path)
    if thumb is None:
        return None
    gray = cv2.cvtColor(thumb, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    return {
        "path": path,
        "hash": phash(thumb),
        "sharpness": float(cv2.Laplacian(blurred, cv2.CV_64F).var()),
        "contrast": float(gray.std()),
        "brightness": float(gray.mean()),
        "warmth": float(thumb[:, :, 0].mean() - thumb[:, :, 2].mean()),
    }


def motion_profile(path, fps=6, width=64, height=36):
    command = [
        FFMPEG,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        path,
        "-vf",
        "fps=%d,scale=%d:%d" % (fps, width, height),
        "-pix_fmt",
        "gray",
        "-f",
        "rawvideo",
        "-",
    ]
    result = subprocess.run(command, capture_output=True)
    data = result.stdout
    frame_size = width * height
    count = len(data) // frame_size
    if count < 8:
        return None
    frames = np.frombuffer(data[: count * frame_size], dtype=np.uint8).reshape(count, height, width).astype(np.float32)
    return np.abs(np.diff(frames, axis=0)).mean(axis=(1, 2)), fps, count


def grab_frame(path, moment, width=320, height=180):
    result = subprocess.run(
        [
            FFMPEG,
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            "%.3f" % max(0.0, moment),
            "-i",
            path,
            "-frames:v",
            "1",
            "-vf",
            "scale=%d:%d" % (width, height),
            "-pix_fmt",
            "bgr24",
            "-f",
            "rawvideo",
            "-",
        ],
        capture_output=True,
    )
    data = result.stdout
    if len(data) < width * height * 3:
        return None
    return np.frombuffer(data[: width * height * 3], dtype=np.uint8).reshape(height, width, 3)


def probe_duration(path):
    result = subprocess.run([FFMPEG, "-hide_banner", "-i", path], capture_output=True, text=True, errors="replace")
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", result.stderr)
    if not match:
        return 0.0
    return int(match.group(1)) * 3600 + int(match.group(2)) * 60 + float(match.group(3))


def annotate_media(items):
    for index, item in enumerate(items):
        if item["kind"] == "image":
            thumb = decode_thumb(item["path"])
            item["duration"] = 0.0
        else:
            thumb = grab_frame(item["path"], 1.0)
            item["duration"] = probe_duration(item["path"])
        if thumb is None:
            item["hash"] = None
            item["sharpness"] = 0.0
            item["contrast"] = 0.0
        else:
            item["hash"] = phash(thumb)
            gray = cv2.cvtColor(thumb, cv2.COLOR_BGR2GRAY)
            item["sharpness"] = float(cv2.Laplacian(cv2.GaussianBlur(gray, (3, 3), 0), cv2.CV_64F).var())
            item["contrast"] = float(gray.std())
        if (index + 1) % 60 == 0:
            log("  fingerprinted %d/%d" % (index + 1, len(items)))
    return items


def dedupe_consecutive(items, window_seconds=DEDUPE_WINDOW):
    kept = []
    dropped = 0
    for item in items:
        if not kept:
            kept.append(item)
            continue
        previous = kept[-1]
        gap = abs((item["stamp"] - previous["stamp"]).total_seconds())
        if item["hash"] is not None and previous["hash"] is not None and gap <= window_seconds:
            if hamming(item["hash"], previous["hash"]) <= PHASH_DUPLICATE:
                dropped += 1
                continue
        kept.append(item)
    return kept, dropped


def split_chapters(items, count):
    size = len(items) / float(count)
    groups = []
    for index in range(count):
        first = int(round(index * size))
        last = int(round((index + 1) * size))
        groups.append(items[first:last])
    return groups


def pick_sequence(group, count):
    if not group:
        return []
    if count >= len(group):
        return list(group)
    step = len(group) / float(count)
    return [group[min(len(group) - 1, int(round(index * step)))] for index in range(count)]
def build_lut():
    x = np.arange(256, dtype=np.float64) / 255.0
    y = 0.5 + (x - 0.5) * 1.62
    y = np.clip(y, 0.0, 1.0)
    y = np.power(y, 1.18)
    y = y + 0.012 * (1.0 - y) * (1.0 - y)
    knee = 0.72
    over = np.maximum(y - knee, 0.0)
    y = np.where(y > knee, knee + over / (1.0 + 1.35 * over), y)
    base = np.clip(y, 0.0, 1.0) * 255.0
    shade = (1.0 - x) ** 2
    light = x ** 2
    table = np.zeros((256, 1, 3), dtype=np.uint8)
    table[:, 0, 0] = np.clip(base - 22.0 * shade + 30.0 * light - 6.0, 0, 255)
    table[:, 0, 1] = np.clip(base - 4.0 * shade + 8.0 * light - 4.0, 0, 255)
    table[:, 0, 2] = np.clip(base + 26.0 * shade - 12.0 * light + 6.0, 0, 255)
    return table


def build_vignette():
    yy, xx = np.mgrid[0:CONTENT_H, 0:WIDTH]
    nx = (xx - (WIDTH - 1) / 2.0) / ((WIDTH - 1) / 2.0)
    ny = (yy - (CONTENT_H - 1) / 2.0) / ((CONTENT_H - 1) / 2.0)
    radius = np.sqrt(nx * nx * 0.88 + ny * ny * 0.88)
    falloff = 1.0 - 0.42 * np.clip(radius - 0.42, 0.0, None) ** 1.45
    falloff = np.clip(falloff, 0.55, 1.0)
    return np.ascontiguousarray((np.repeat(falloff[:, :, None], 3, axis=2) * 255.0).astype(np.uint8))


def build_grain():
    rng = np.random.RandomState(20261128)
    plates = []
    for index in range(GRAIN_PLATES):
        noise = rng.normal(128.0, 26.0, (CONTENT_H, WIDTH, 3))
        plates.append(np.clip(noise, 0, 255).astype(np.uint8))
    return plates


def build_warp_maps():
    yy, xx = np.mgrid[0:CONTENT_H, 0:WIDTH].astype(np.float32)
    center_x = (WIDTH - 1) / 2.0
    center_y = (CONTENT_H - 1) / 2.0
    nx = (xx - center_x) / center_x
    ny = (yy - center_y) / center_y
    radius = nx * nx + ny * ny
    maps = []
    for strength in (0.12, 0.24, 0.38, 0.55):
        scale = 1.0 + strength * (radius - 0.45)
        maps.append(
            (
                np.ascontiguousarray(center_x + nx * center_x * scale),
                np.ascontiguousarray(center_y + ny * center_y * scale),
            )
        )
    return maps


def build_leaks():
    leaks = []
    yy, xx = np.mgrid[0:CONTENT_H, 0:WIDTH].astype(np.float32)
    for index in range(3):
        cx = 0.12 + 0.76 * (index / 2.0)
        cy = 0.18 + 0.35 * ((index + 1) % 2)
        dx = (xx / WIDTH - cx) / 0.42
        dy = (yy / CONTENT_H - cy) / 0.55
        glow = np.exp(-(dx * dx + dy * dy) * 2.1)
        streak = np.exp(-((yy / CONTENT_H - cy) ** 2) * 26.0) * np.clip(1.0 - np.abs(xx / WIDTH - cx) * 1.4, 0, 1)
        field = np.clip(glow * 0.85 + streak * 0.5, 0.0, 1.0)
        colour = [(255, 214, 170), (255, 176, 120), (198, 226, 255)][index]
        layer = np.dstack([field * colour[0], field * colour[1], field * colour[2]])
        leaks.append(layer.astype(np.float32))
    return leaks


GRADE = build_lut()
VIGNETTE = build_vignette()
GRAIN = build_grain()
WARPS = build_warp_maps()
LEAKS = build_leaks()
LEAK_LEVELS = (0.10, 0.16, 0.22, 0.28)
LEAK_PLATES = [[np.clip(leak * level, 0, 255).astype(np.uint8) for leak in LEAKS] for level in LEAK_LEVELS]
WHITE = np.full((CONTENT_H, WIDTH, 3), 255, dtype=np.uint8)
BLACK = np.zeros((CONTENT_H, WIDTH, 3), dtype=np.uint8)


def make_card(text, font_path, size, stroke_ratio, fill=(255, 255, 255)):
    font = ImageFont.truetype(font_path, size)
    stroke = max(3, int(round(size * stroke_ratio)))
    probe = ImageDraw.Draw(Image.new("RGBA", (16, 16)))
    box = probe.textbbox((0, 0), text, font=font, stroke_width=stroke)
    pad = stroke * 3 + 10
    canvas = Image.new("RGBA", (box[2] - box[0] + pad * 2, box[3] - box[1] + pad * 2), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (pad - box[0], pad - box[1]),
        text,
        font=font,
        fill=(fill[0], fill[1], fill[2], 255),
        stroke_width=stroke,
        stroke_fill=(0, 0, 0, 255),
    )
    array = np.array(canvas)
    rgb = np.ascontiguousarray(array[:, :, :3])
    alpha = np.ascontiguousarray(array[:, :, 3].astype(np.float32) / 255.0)
    return rgb, alpha


def beat_pattern(total, shots, first, last):
    weights = np.linspace(first, last, shots)
    raw = weights / weights.sum() * total
    values = np.maximum(2, np.round(raw).astype(int))
    difference = total - int(values.sum())
    index = 0
    while difference != 0:
        position = index % shots
        step = 1 if difference > 0 else -1
        if values[position] + step >= 2:
            values[position] += step
            difference -= step
        index += 1
    return [int(value) for value in values]


CHAPTER_SPECS = [
    {
        "name": "PROLOGUE",
        "cn": "序章",
        "en": "PROLOGUE",
        "total": 14,
        "shots": 3,
        "first": 6,
        "last": 4,
        "mode": "pan",
        "glitch": 0.20,
        "ramp": False,
        "energy": 0.40,
    },
    {
        "name": "ARENA",
        "cn": "竞技",
        "en": "ARENA",
        "total": 34,
        "shots": 12,
        "first": 4,
        "last": 2,
        "mode": "pan",
        "glitch": 0.55,
        "ramp": True,
        "energy": 0.85,
    },
    {
        "name": "CLIMAX",
        "cn": "高潮",
        "en": "CLIMAX",
        "total": 72,
        "shots": 26,
        "first": 4,
        "last": 2,
        "mode": "rush",
        "glitch": 1.00,
        "ramp": True,
        "energy": 1.00,
    },
    {
        "name": "CODA",
        "cn": "尾声",
        "en": "CODA",
        "total": 60,
        "shots": 14,
        "first": 6,
        "last": 3,
        "mode": "pan",
        "glitch": 0.35,
        "ramp": False,
        "energy": 0.65,
    },
]

CHAPTERS = []
for _spec in CHAPTER_SPECS:
    _entry = dict(_spec)
    _entry["beats"] = beat_pattern(_spec["total"], _spec["shots"], _spec["first"], _spec["last"])
    CHAPTERS.append(_entry)

CHAPTER_START = []
_cursor = 0.0
for _entry in CHAPTERS:
    CHAPTER_START.append(_cursor)
    _cursor += sum(_entry["beats"]) * BEAT

CHAPTER_SHOT_OFFSET = []
_offset = 0
for _entry in CHAPTERS:
    CHAPTER_SHOT_OFFSET.append(_offset)
    _offset += len(_entry["beats"])
def build_cards(starts):
    cards = []
    prologue = starts[0]
    arena = starts[1]
    climax = starts[2]
    coda = starts[3]

    def add(key, text, font, size, cx, cy, start, end, mode="fly", glow=0.55, colour=(255, 255, 255)):
        rgb, alpha = make_card(text, font, size, 0.05, colour)
        cards.append(
            {
                "key": key,
                "text": text,
                "rgb": rgb,
                "alpha": alpha,
                "cx": cx,
                "cy": cy,
                "start": start,
                "end": end,
                "mode": mode,
                "glow": glow,
                "mask": False,
            }
        )

    add("school_cn", SCHOOL_CN, FONT_CN, 104, 0.5, 0.54, prologue + 0.30, prologue + 3.60, "slam", 0.55)
    add("school_en", SCHOOL_EN, FONT_HEAVY, 210, 0.5, 0.36, prologue + 0.60, prologue + 3.60, "slam", 0.95)
    add("tag1", "序章 PROLOGUE", FONT_CN, 62, 0.26, 0.74, prologue + 4.00, prologue + 5.60, "slam", 0.55)
    add("class_cn", CLASS_CN, FONT_CN, 150, 0.5, 0.62, arena + 0.30, arena + 3.20, "slam", 1.05)
    add("class_en", CLASS_EN, FONT_HEAVY, 258, 0.5, 0.38, arena + 0.60, arena + 3.20, "slam", 1.00)
    add("tag2", "竞技 ARENA", FONT_CN, 62, 0.26, 0.74, arena + 3.80, arena + 5.40, "slam", 0.55)
    add("grit", "GRIT", FONT_HEAVY, 250, 0.5, 0.42, 12.00, 14.20, "slam", 1.00)
    add("sweat", "SWEAT", FONT_HEAVY, 250, 0.5, 0.42, 17.00, 19.20, "slam", 1.00)
    add("tag3", "高潮 CLIMAX", FONT_CN, 62, 0.26, 0.74, climax + 0.30, climax + 1.90, "slam", 0.65)
    add("motto", MOTTO, FONT_HEAVY, 190, 0.5, 0.44, 37.00, 39.60, "slam", 1.05)
    add("glory", "GLORY", FONT_HEAVY, 240, 0.5, 0.42, 50.00, 52.40, "slam", 1.00)
    add("tag4", "尾声 CODA", FONT_CN, 62, 0.26, 0.74, coda + 0.30, coda + 1.90, "slam", 0.50)
    add("end_cn", SCHOOL_CN, FONT_CN, 90, 0.5, 0.44, 86.00, 89.40, "rise", 0.55)
    add("end_class", CLASS_CN, FONT_CN, 64, 0.5, 0.58, 86.60, 89.40, "rise", 0.55)
    return cards
CARDS = build_cards(CHAPTER_START)
MASK_BY_KEY = {card["key"]: card for card in CARDS}


def ease(value):
    value = min(1.0, max(0.0, value))
    return 1.0 - (1.0 - value) ** 3


def warp_card(card, progress, mode, pulse=0.0):
    rgb = card["rgb"]
    alpha = card["alpha"]
    source_h, source_w = alpha.shape
    if mode == "slam":
        scale = 1.0 + 0.55 * (1.0 - ease(progress)) + 0.06 * pulse
        keystone = 1.0
        offset_y = -0.16 * (1.0 - ease(progress))
    elif mode == "rise":
        scale = 0.62 + 0.38 * ease(progress) + 0.04 * pulse
        keystone = 0.55 + 0.45 * ease(progress)
        offset_y = 0.10 * (1.0 - ease(progress))
    else:
        scale = 0.32 + 0.68 * ease(progress) + 0.05 * pulse
        keystone = 0.58 + 0.42 * ease(progress)
        offset_y = -0.05 * (1.0 - ease(progress))
    target_w = source_w * scale
    target_h = source_h * scale
    center_x = card["cx"] * WIDTH
    center_y = (card["cy"] + offset_y) * CONTENT_H
    half_top = target_w * 0.5 * keystone
    half_bottom = target_w * 0.5
    source_points = np.float32([[0, 0], [source_w, 0], [source_w, source_h], [0, source_h]])
    target_points = np.float32(
        [
            [center_x - half_top, center_y - target_h * 0.5],
            [center_x + half_top, center_y - target_h * 0.5],
            [center_x + half_bottom, center_y + target_h * 0.5],
            [center_x - half_bottom, center_y + target_h * 0.5],
        ]
    )
    matrix = cv2.getPerspectiveTransform(source_points, target_points)
    left = int(max(0, math.floor(target_points[:, 0].min()) - 4))
    right = int(min(WIDTH, math.ceil(target_points[:, 0].max()) + 4))
    top = int(max(0, math.floor(target_points[:, 1].min()) - 4))
    bottom = int(min(CONTENT_H, math.ceil(target_points[:, 1].max()) + 4))
    if right <= left or bottom <= top:
        return None
    matrix[0, 2] -= left
    matrix[1, 2] -= top
    size = (right - left, bottom - top)
    warped_rgb = cv2.warpPerspective(rgb, matrix, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    warped_alpha = cv2.warpPerspective(alpha, matrix, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    return left, top, warped_rgb, warped_alpha


def composite_text(frame, card, progress, pulse, alpha_scale, echo=0.0):
    warped = warp_card(card, progress, card["mode"], pulse)
    if warped is None or alpha_scale <= 0.01:
        return frame
    left, top, warped_rgb, warped_alpha = warped
    mask = warped_alpha * alpha_scale
    height, width = mask.shape
    region = frame[top : top + height, left : left + width]
    if region.shape[:2] != mask.shape:
        return frame
    if echo > 0.02:
        trail = max(2, int(round(echo * 26.0)))
        red = cv2.multiply(warped_rgb, np.array([0.35, 0.35, 1.25], dtype=np.float32))
        cyan = cv2.multiply(warped_rgb, np.array([1.25, 0.40, 0.35], dtype=np.float32))
        red = np.roll(red, trail, axis=1)
        cyan = np.roll(cyan, -trail, axis=1)
        warped_rgb = np.clip(warped_rgb.astype(np.float32) * 0.55 + red * echo * 0.55 + cyan * echo * 0.55, 0, 255).astype(np.uint8)
    mask_u8 = (mask * 255.0).astype(np.uint8)
    glow = cv2.GaussianBlur(warped_rgb, (0, 0), 6.0)
    softened = cv2.GaussianBlur(mask_u8, (0, 0), 9.0).astype(np.float32) * (1.5 / 255.0)
    soft_alpha = np.clip(softened, 0.0, 1.0)
    glow_weight = (soft_alpha * card["glow"]).astype(np.float32)
    glow_layer = (glow.astype(np.float32) * glow_weight[:, :, None]).astype(np.uint8)
    mask_rgb = cv2.cvtColor(mask_u8, cv2.COLOR_GRAY2RGB)
    inverse_rgb = cv2.bitwise_not(mask_rgb)
    kept = cv2.multiply(region, inverse_rgb, scale=1.0 / 255.0)
    laid = cv2.multiply(warped_rgb, mask_rgb, scale=1.0 / 255.0)
    blended = cv2.add(cv2.add(kept, laid), glow_layer)
    frame[top : top + height, left : left + width] = blended
    return frame


def overlay_cards(frame, moment):
    for card in CARDS:
        if moment < card["start"] or moment > card["end"]:
            continue
        span = card["end"] - card["start"]
        local = moment - card["start"]
        progress = min(1.0, local / max(0.22, min(0.62, span * 0.32)))
        alpha = 1.0
        fade = 0.16
        if local < fade:
            alpha = local / fade
        if span - local < fade:
            alpha = min(alpha, (span - local) / fade)
        pulse = 0.5 + 0.5 * math.sin(2.0 * math.pi * (moment / BEAT))
        echo = max(0.0, 1.0 - local / 0.28) if card["mode"] == "slam" else 0.0
        frame = composite_text(frame, card, progress, pulse, alpha, echo)
    return frame


def grade_frame(frame):
    graded = cv2.LUT(frame, GRADE)
    gray = cv2.cvtColor(graded, cv2.COLOR_RGB2GRAY)
    gray3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    return cv2.addWeighted(graded, 1.12, gray3, -0.12, 0)


def bloom_pass(frame, strength):
    if strength <= 0.01:
        return frame
    small = cv2.resize(frame, (WIDTH // 4, CONTENT_H // 4), interpolation=cv2.INTER_AREA)
    bright = cv2.subtract(small, np.array([168], dtype=np.uint8))
    glow = cv2.GaussianBlur(bright, (0, 0), 7.0)
    glow[:, :, 2] = (glow[:, :, 2] * 0.72).astype(np.uint8)
    glow[:, :, 1] = (glow[:, :, 1] * 0.90).astype(np.uint8)
    glow = cv2.resize(glow, (WIDTH, CONTENT_H), interpolation=cv2.INTER_LINEAR)
    return cv2.addWeighted(frame, 1.0, glow, strength, 0)


def motion_blur_pass(frame, amount):
    if amount <= 0.05:
        return frame
    step = max(2, int(round(10 + 26 * amount)))
    shifted = np.roll(frame, -step, axis=1)
    shifted2 = np.roll(frame, -step * 2, axis=1)
    blended = frame.astype(np.float32) * 0.55 + shifted.astype(np.float32) * 0.28 + shifted2.astype(np.float32) * 0.17
    return blended.astype(np.uint8)


def lens_warp(frame, strength):
    if strength <= 0.02:
        return frame
    index = 0
    if strength > 0.44:
        index = 3
    elif strength > 0.30:
        index = 2
    elif strength > 0.18:
        index = 1
    map_x, map_y = WARPS[index]
    return cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def glitch_pass(frame, strength, local, seed):
    if strength <= 0.02:
        return frame
    rng = np.random.RandomState(int(seed * 977 + local * 613) % (2 ** 31))
    shift = int(3 + 14 * strength)
    output = frame.copy()
    output[:, :, 0] = np.roll(frame[:, :, 0], -shift, axis=1)
    output[:, :, 2] = np.roll(frame[:, :, 2], shift, axis=1)
    bands = int(2 + 6 * strength)
    for band in range(bands):
        height = int(rng.randint(8, 70))
        top = int(rng.randint(0, max(1, CONTENT_H - height)))
        offset = int(rng.randint(-int(40 + 120 * strength), int(40 + 120 * strength)))
        output[top : top + height] = np.roll(output[top : top + height], offset, axis=1)
    if strength > 0.45:
        scan = (np.arange(CONTENT_H) % 3 == 0).astype(np.float32)[:, None, None]
        noise = rng.normal(0, 26 * strength, (CONTENT_H, WIDTH, 1)).astype(np.float32)
        output = np.clip(output.astype(np.float32) + scan * noise, 0, 255).astype(np.uint8)
    return output


def leak_pass(frame, moment, strength):
    if strength <= 0.01:
        return frame
    index = int(moment * 0.7) % len(LEAKS)
    level = min(range(len(LEAK_LEVELS)), key=lambda step: abs(LEAK_LEVELS[step] - strength))
    return cv2.add(frame, LEAK_PLATES[level][index])


def finish_pass(frame, index, grain_scale):
    frame = cv2.addWeighted(frame, 1.0, GRAIN[index % GRAIN_PLATES], GRAIN_STRENGTH * grain_scale, -128.0 * GRAIN_STRENGTH * grain_scale)
    return cv2.multiply(frame, VIGNETTE, scale=1.0 / 255.0)


def pan_window(source_w, source_h, zoom, pan_x, pan_y):
    base = max(WIDTH / float(source_w), CONTENT_H / float(source_h)) * zoom
    window_w = WIDTH / base
    window_h = CONTENT_H / base
    if window_w > source_w:
        window_w = float(source_w)
        window_h = window_w * CONTENT_H / float(WIDTH)
    if window_h > source_h:
        window_h = float(source_h)
        window_w = window_h * WIDTH / float(CONTENT_H)
    max_x = max(0.0, source_w - window_w)
    max_y = max(0.0, source_h - window_h)
    left = min(max(pan_x * max_x, 0.0), max_x)
    top = min(max(pan_y * max_y, 0.0), max_y)
    return left, top, window_w, window_h


def render_pan_scan(source, zoom, pan_x, pan_y, sharpen=True):
    height, width = source.shape[:2]
    left, top, window_w, window_h = pan_window(width, height, zoom, pan_x, pan_y)
    x0 = int(round(left))
    y0 = int(round(top))
    x1 = min(width, max(x0 + 2, int(round(left + window_w))))
    y1 = min(height, max(y0 + 2, int(round(top + window_h))))
    patch = source[y0:y1, x0:x1]
    interpolation = cv2.INTER_AREA if patch.shape[1] >= WIDTH else cv2.INTER_CUBIC
    frame = cv2.resize(patch, (WIDTH, CONTENT_H), interpolation=interpolation)
    if sharpen:
        frame = usm_sharpen(frame)
    return frame


def usm_sharpen(frame, radius=1.6, amount=0.85, threshold=2.0):
    blurred = cv2.GaussianBlur(frame, (0, 0), radius)
    return cv2.addWeighted(frame, 1.0 + amount, blurred, -amount, 0)


def camera_shake(frame, amplitude, phase, index):
    if amplitude <= 0.05:
        return frame
    rng = np.random.RandomState((int(phase * 1000) * 37 + index * 131) % (2 ** 31))
    shift_x = int(round(rng.uniform(-1.0, 1.0) * amplitude * 18.0))
    shift_y = int(round(rng.uniform(-1.0, 1.0) * amplitude * 13.0))
    angle = float(rng.uniform(-1.0, 1.0) * amplitude * 1.05)
    matrix = cv2.getRotationMatrix2D((WIDTH / 2.0, CONTENT_H / 2.0), angle, 1.0 + amplitude * 0.012)
    matrix[0, 2] += shift_x
    matrix[1, 2] += shift_y
    return cv2.warpAffine(frame, matrix, (WIDTH, CONTENT_H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


IMAGE_CACHE = {}
IMAGE_ORDER = []
VIDEO_CACHE = {}
VIDEO_ORDER = []


def load_image(path):
    cached = IMAGE_CACHE.get(path)
    if cached is not None:
        return cached
    image = decode_full(path)
    if image is None:
        raise RuntimeError("unreadable image")
    IMAGE_CACHE[path] = image
    IMAGE_ORDER.append(path)
    while len(IMAGE_ORDER) > 3:
        stale = IMAGE_ORDER.pop(0)
        IMAGE_CACHE.pop(stale, None)
    return image


def video_frame(path, moment):
    clip = VIDEO_CACHE.get(path)
    if clip is None:
        with contextlib.redirect_stdout(io.StringIO()):
            clip = VideoFileClip(path, audio=False)
        VIDEO_CACHE[path] = clip
        VIDEO_ORDER.append(path)
        while len(VIDEO_ORDER) > 2:
            stale = VIDEO_ORDER.pop(0)
            dropped = VIDEO_CACHE.pop(stale, None)
            if dropped is not None:
                dropped.close()
    limit = max(0.0, clip.duration - 1.5 / FPS)
    return clip.get_frame(min(max(moment, 0.0), limit))


MOTION_POOL = [
    (1.00, 1.34, 0.02, 0.98, 0.50, 0.50),
    (1.00, 1.34, 0.98, 0.02, 0.50, 0.50),
    (1.00, 1.30, 0.50, 0.50, 0.02, 0.98),
    (1.00, 1.30, 0.50, 0.50, 0.98, 0.02),
    (1.02, 1.42, 0.06, 0.94, 0.10, 0.90),
    (1.02, 1.42, 0.94, 0.06, 0.90, 0.10),
    (1.06, 1.52, 0.50, 0.50, 0.50, 0.50),
    (1.00, 1.26, 0.84, 0.16, 0.50, 0.50),
]


SPEEDS = [
    (0.50, 0.50, "slowmo"),
    (0.50, 5.00, "slam_up"),
    (5.00, 0.50, "slam_down"),
    (1.00, 1.00, "real"),
    (0.40, 6.00, "hyper"),
    (6.00, 0.40, "hyper_down"),
    (0.35, 0.35, "slow2"),
    (0.50, 3.20, "kick_out"),
]


def build_timeline(groups):
    shots = []
    beat_cursor = 0
    order = 0
    for chapter_index, chapter in enumerate(CHAPTERS):
        pattern = chapter["beats"]
        items = pick_sequence(groups[chapter_index], len(pattern))
        for position, beats in enumerate(pattern):
            if not items:
                continue
            item = items[min(position, len(items) - 1)]
            duration = beats * BEAT
            shot = {
                "index": order,
                "chapter": chapter_index,
                "chapter_name": chapter["name"],
                "kind": item["kind"],
                "beats": beats,
                "start": beat_cursor * BEAT,
                "duration": duration,
                "path": item["path"],
                "stamp": item["stamp"],
                "origin": item["origin"],
                "energy": chapter["energy"],
                "hero": False,
                "glitch": 0.0,
                "warp": 0.0,
                "blur": 0.0,
                "leak": 0.10 + 0.16 * chapter["energy"],
                "bloom": 0.16 + 0.22 * chapter["energy"],
                "mask": None,
                "shake": 0.0,
            }
            if item["kind"] == "video":
                source_span = max(1.0, float(item.get("duration") or 4.0) - 2.0)
                if chapter["ramp"]:
                    s0, s1, label = SPEEDS[order % len(SPEEDS)]
                    needed = duration * (s0 + s1) / 2.0
                    if needed > source_span:
                        factor = source_span / needed
                        s0 *= factor
                        s1 *= factor
                else:
                    s0, s1, label = 1.0, 1.0, "real"
                    if duration > source_span:
                        s0 = s1 = source_span / duration
                shot["src_start"] = 1.0
                shot["speed"] = (s0, s1, label)
                shot["motion"] = (1.03, 1.12, 0.50, 0.50, 0.47, 0.47)
                shot["blur"] = min(1.0, (s0 + s1) / 4.0) if chapter["ramp"] else 0.0
            else:
                base = MOTION_POOL[order % len(MOTION_POOL)]
                if chapter["mode"] == "parallax":
                    shot["motion"] = (base[0], base[1], base[2], base[3], base[4], base[5])
                    shot["hero"] = position % 2 == 0 or chapter_index == 0
                else:
                    shot["motion"] = (base[0], base[0] + (base[1] - base[0]) * 1.35, base[2], base[3], base[4], base[5])
                    if chapter["ramp"]:
                        shot["blur"] = 0.55 + 0.35 * min(1.0, order / 20.0)
            if beat_cursor % 8 == 0 and chapter_index > 0:
                shot["glitch"] = min(1.0, chapter["glitch"] + 0.12)
            elif chapter["glitch"] >= 0.9 and beats <= 3:
                shot["glitch"] = 0.72
            elif chapter["glitch"] >= 0.4 and order % 5 == 0:
                shot["glitch"] = 0.45
            if order in (1, CHAPTER_SHOT_OFFSET[2], CHAPTER_SHOT_OFFSET[3]):
                shot["warp"] = 0.55
                shot["shake"] = 0.85
            if beat_cursor % 16 == 0 and chapter_index > 0:
                shot["shake"] = max(shot["shake"], 0.70)
            if chapter["energy"] >= 1.0 and beats <= 3:
                shot["shake"] = max(shot["shake"], 0.55)
            if order == 0:
                shot["transition"] = "open"
            elif beat_cursor % 8 == 0:
                shot["transition"] = "overexpose"
            elif beats <= 2:
                shot["transition"] = "blink"
            else:
                shot["transition"] = "cut"
            shots.append(shot)
            beat_cursor += beats
            order += 1
    return shots, beat_cursor
MASK_PLAN = [(CHAPTER_SHOT_OFFSET[1], "class_cn"), (CHAPTER_SHOT_OFFSET[2], "grit")]


def apply_mask_plan(shots):
    for shot_index, key in MASK_PLAN:
        if 0 <= shot_index < len(shots):
            shots[shot_index]["mask"] = key
            shots[shot_index]["transition"] = "textmask"
    return shots
def speed_offset(shot, local):
    duration = shot["duration"]
    s0, s1, label = shot["speed"]
    ratio = 0.0 if duration <= 0 else min(1.0, max(0.0, local / duration))
    return s0 * local + (s1 - s0) * duration * ratio * ratio / 2.0


def render_shot(shot, local):
    progress = min(1.0, max(0.0, local / shot["duration"]))
    motion = shot["motion"]
    s0, s1 = motion[0], motion[1]
    zoom = s0 + (s1 - s0) * progress
    pan_x = motion[2] + (motion[3] - motion[2]) * progress
    pan_y = motion[4] + (motion[5] - motion[4]) * progress
    if shot["kind"] == "image":
        source = load_image(shot["path"])
        if source is None:
            raise RuntimeError("unreadable image")
    else:
        source = video_frame(shot["path"], shot["src_start"] + speed_offset(shot, local))
    return render_pan_scan(source, zoom, pan_x, pan_y)


def build_clip(shots, total_seconds):
    starts = [shot["start"] for shot in shots]
    stats = {"glitch": 0, "warp": 0, "hero": 0, "mask": 0, "blur": 0, "flash": 0}

    def locate(moment):
        index = bisect.bisect_right(starts, moment) - 1
        index = max(0, min(index, len(shots) - 1))
        return index, moment - starts[index]

    def build_frame(t):
        moment = float(t)
        index, local = locate(moment)
        shot = shots[index]
        frame = render_shot(shot, local)
        if shot["warp"] > 0.02:
            envelope = max(0.0, 1.0 - local / 0.42)
            frame = lens_warp(frame, shot["warp"] * envelope)
        frame = grade_frame(frame)
        frame = bloom_pass(frame, shot["bloom"])
        if shot["blur"] > 0.05:
            frame = motion_blur_pass(frame, shot["blur"])
        if shot["glitch"] > 0.02:
            envelope = 1.0 if local < 0.30 else max(0.0, 1.0 - (local - 0.30) / 0.35)
            level = shot["glitch"] * (0.55 + 0.45 * math.sin(2.0 * math.pi * moment / BEAT) ** 2)
            frame = glitch_pass(frame, level * envelope, local, shot["index"])
        frame = leak_pass(frame, moment, shot["leak"])
        frame = finish_pass(frame, index, 1.0 if shot["kind"] == "video" else 0.8)
        shake = shot["shake"] * math.exp(-local * 6.5)
        if shake > 0.02:
            frame = camera_shake(frame, shake, moment, index)
        if moment < 2.86:
            dim = 0.10 + 0.22 * (moment / 2.86)
            frame = cv2.convertScaleAbs(frame, alpha=dim, beta=0)
        elif moment < 3.30:
            rise = (moment - 2.86) / 0.44
            frame = cv2.addWeighted(frame, min(1.0, 0.32 + rise), WHITE, max(0.0, 0.68 * (1.0 - rise)), 0)
        if shot["transition"] == "overexpose" and local < 0.20:
            weight = 0.92 * (1.0 - local / 0.20)
            frame = cv2.addWeighted(frame, 1.0 - weight, WHITE, weight, 0)
        if shot["transition"] == "blink" and local < 0.13:
            gain = local / 0.13
            frame = cv2.convertScaleAbs(frame, alpha=gain, beta=0)
        if shot["mask"]:
            card = MASK_BY_KEY.get(shot["mask"])
            if card is not None and local < 0.34:
                reveal = ease(min(1.0, local / 0.30))
                mask_rgb, mask_alpha = card["rgb"], card["alpha"]
                target_w = int(WIDTH * (0.55 + 0.55 * reveal))
                target_h = int(target_w * mask_alpha.shape[0] / float(mask_alpha.shape[1]))
                resized_alpha = cv2.resize(mask_alpha, (target_w, target_h), interpolation=cv2.INTER_AREA)
                resized_rgb = cv2.resize(mask_rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)
                canvas_mask = np.zeros((CONTENT_H, WIDTH), dtype=np.float32)
                left = (WIDTH - target_w) // 2
                top = (CONTENT_H - target_h) // 2
                x0 = max(0, left)
                y0 = max(0, top)
                x1 = min(WIDTH, left + target_w)
                y1 = min(CONTENT_H, top + target_h)
                if x1 > x0 and y1 > y0:
                    canvas_mask[y0:y1, x0:x1] = resized_alpha[y0 - top : y0 - top + (y1 - y0), x0 - left : x0 - left + (x1 - x0)]
                    canvas_rgb = np.zeros((CONTENT_H, WIDTH, 3), dtype=np.uint8)
                    canvas_rgb[y0:y1, x0:x1] = resized_rgb[y0 - top : y0 - top + (y1 - y0), x0 - left : x0 - left + (x1 - x0)]
                else:
                    canvas_rgb = np.zeros((CONTENT_H, WIDTH, 3), dtype=np.uint8)
                mask_strength = canvas_mask[:, :, None] * (1.0 - reveal)
                frame = np.clip(frame.astype(np.float32) * (1.0 - mask_strength * 0.85) + canvas_rgb.astype(np.float32) * mask_strength * 0.85, 0, 255).astype(np.uint8)
        frame = overlay_cards(frame, moment)
        tail = total_seconds - moment
        if tail < 1.8:
            frame = cv2.convertScaleAbs(frame, alpha=max(0.0, tail / 1.8), beta=0)
        canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        canvas[BAR_H : BAR_H + CONTENT_H] = frame
        return canvas

    return VideoClip(frame_function=build_frame, duration=total_seconds), stats


def build_score(total_seconds):
    sample_count = int(round(total_seconds * AUDIO_RATE))
    mix = np.zeros((sample_count, 2), dtype=np.float32)
    rng = np.random.RandomState(20261128)

    def timeline(duration):
        return np.arange(int(round(duration * AUDIO_RATE)), dtype=np.float64) / AUDIO_RATE

    def add(start_seconds, wave, gain, pan=0.0):
        start = int(round(start_seconds * AUDIO_RATE))
        if start < 0 or start >= sample_count or wave.size == 0:
            return
        length = min(wave.size, sample_count - start)
        mix[start : start + length, 0] += wave[:length] * gain * (1.0 - 0.28 * pan)
        mix[start : start + length, 1] += wave[:length] * gain * (1.0 + 0.28 * pan)

    kick_time = timeline(0.55)
    kick_freq = 44.0 + 175.0 * np.exp(-kick_time * 40.0)
    kick = np.sin(2.0 * np.pi * np.cumsum(kick_freq) / AUDIO_RATE) * np.exp(-kick_time * 6.0)
    kick += rng.randn(kick_time.size) * np.exp(-kick_time * 260.0) * 0.22
    kick = (kick * 0.95).astype(np.float32)

    heart_time = timeline(0.65)
    heart = np.sin(2.0 * np.pi * 42.0 * heart_time) * np.exp(-heart_time * 7.2)
    heart += np.sin(2.0 * np.pi * 84.0 * heart_time) * np.exp(-heart_time * 11.0) * 0.22
    heart = (heart * 0.9).astype(np.float32)

    snare_time = timeline(0.26)
    snare = (np.diff(rng.randn(snare_time.size), prepend=0.0) * 0.48 + np.sin(2.0 * np.pi * 192.0 * snare_time) * 0.2) * np.exp(-snare_time * 22.0)
    snare = snare.astype(np.float32)

    hat_time = timeline(0.085)
    hat = (np.diff(rng.randn(hat_time.size), prepend=0.0) * np.exp(-hat_time * 120.0) * 0.34).astype(np.float32)

    impact_time = timeline(2.2)
    impact = np.sin(2.0 * np.pi * (62.0 * np.exp(-impact_time * 1.3) + 24.0) * impact_time) * np.exp(-impact_time * 2.0)
    impact += rng.randn(impact_time.size) * np.exp(-impact_time * 9.0) * 0.20
    impact = (impact * 0.95).astype(np.float32)
    attack = max(1, int(0.006 * AUDIO_RATE))
    impact[:attack] *= np.linspace(0.0, 1.0, attack)

    rise_time = timeline(1.4)
    rise_noise = np.diff(rng.randn(rise_time.size), prepend=0.0)
    rise = (rise_noise * ((rise_time / 1.4) ** 2.6) * 0.55).astype(np.float32)

    wind_time = timeline(total_seconds)
    wind_raw = np.cumsum(rng.randn(wind_time.size))
    wind = wind_raw / (float(np.abs(wind_raw).max()) + 1e-9)
    wind = np.convolve(wind, np.ones(220) / 220.0, mode="same")
    wind = wind * (0.55 + 0.45 * np.sin(2.0 * np.pi * 0.06 * wind_time))
    wind = (wind * 1.6).astype(np.float32)

    def note_frequency(midi):
        return 440.0 * (2.0 ** ((midi - 69) / 12.0))

    def formant_gain(frequency, formants):
        gain = 0.0
        for centre, width, weight in formants:
            gain += weight * np.exp(-((np.log(frequency + 1e-6) - np.log(centre)) ** 2) / (2.0 * width * width))
        return gain

    def choir(chord, duration, gain=1.0):
        formants = [(720.0, 0.16, 1.0), (1180.0, 0.22, 0.62), (2560.0, 0.26, 0.34)]
        local = timeline(duration)
        wave = np.zeros(local.size, dtype=np.float64)
        for midi in chord:
            base = note_frequency(midi)
            vibrato = 1.0 + 0.0045 * np.sin(2.0 * np.pi * 5.1 * local + rng.rand() * 6.28)
            for harmonic in range(1, 15):
                frequency = base * harmonic * vibrato
                amplitude = formant_gain(frequency, formants) / (harmonic ** 0.55)
                wave += np.sin(2.0 * np.pi * frequency * local + rng.rand() * 6.28) * amplitude
        envelope = np.minimum(1.0, local / 0.55) * np.minimum(1.0, np.maximum(0.0, (duration - local) / 0.7))
        return (wave * envelope * (0.05 * gain / max(1, len(chord)))).astype(np.float32)

    def bass_note(midi, duration):
        frequency = note_frequency(midi)
        local = timeline(duration)
        wave = np.zeros(local.size, dtype=np.float64)
        for harmonic in range(1, 7):
            wave += np.sin(2.0 * np.pi * frequency * harmonic * local) / (harmonic ** 1.2)
        return (wave * np.minimum(1.0, local / 0.004) * np.exp(-local * 4.8) * 0.32).astype(np.float32)

    def pad_chord(midis, duration, gain=1.0):
        local = timeline(duration)
        wave = np.zeros(local.size, dtype=np.float64)
        for midi in midis:
            frequency = note_frequency(midi)
            for detune in (-0.4, 0.0, 0.4):
                wave += np.sin(2.0 * np.pi * (frequency + detune) * local + rng.rand() * 6.283)
        envelope = np.minimum(1.0, local / 0.35) * np.minimum(1.0, np.maximum(0.0, (duration - local) / 0.45))
        return (wave * envelope * (0.05 * gain / max(1, len(midis)))).astype(np.float32)

    def pluck(midi, duration, gain=1.0):
        frequency = note_frequency(midi)
        local = timeline(duration)
        wave = np.sin(2.0 * np.pi * frequency * local) + np.sin(2.0 * np.pi * frequency * 3.0 * local) * 0.22 + np.sin(2.0 * np.pi * frequency * 5.0 * local) * 0.08
        return (wave * np.exp(-local * 7.0) * np.minimum(1.0, local / 0.003) * 0.14 * gain).astype(np.float32)

    progression = [
        (33, [57, 60, 64, 69]),
        (29, [53, 57, 60, 65]),
        (36, [55, 60, 64, 67]),
        (31, [55, 59, 62, 67]),
        (33, [57, 60, 64, 69]),
        (31, [55, 59, 62, 67]),
    ]

    add(0.0, wind[: int(3.0 * AUDIO_RATE)], 0.22, 0.0)
    bars = int(math.ceil(total_seconds / (BEAT * 4)))
    for bar in range(bars):
        bar_start = bar * BEAT * 4
        if bar_start >= total_seconds:
            break
        root, chord = progression[bar % len(progression)]
        prologue = bar < 6
        arena = 6 <= bar < 19
        climax = 19 <= bar < 34
        coda = bar >= 34
        dawn = prologue
        sun = arena
        limit = climax
        glory = coda
        if bar >= 1:
            add(bar_start, pad_chord(chord, min(BEAT * 4 + 0.3, total_seconds - bar_start + 0.3), 1.0 if not glory else 1.4), 1.0, 0.0)
        if dawn:
            for beat_index in range(4):
                add(bar_start + beat_index * BEAT, heart, 0.42, 0.0)
            if bar >= 4:
                for step in range(8):
                    add(bar_start + step * BEAT * 0.5, hat, 0.07, 0.2)
            if bar == 5:
                add(bar_start + BEAT * 2.0, rise, 0.55, 0.0)
                add(bar_start + BEAT * 3.0, choir([57, 64, 69], 1.2, 1.0), 0.9, 0.0)
        elif sun:
            for beat_index in range(4):
                add(bar_start + beat_index * BEAT, kick, 0.82, 0.0)
            for beat_index in (1, 3):
                add(bar_start + beat_index * BEAT, snare, 0.38, -0.1)
            for step in range(16):
                add(bar_start + step * BEAT * 0.25, hat, 0.075 if step % 2 else 0.13, 0.25)
            for step in range(16):
                note = root + 12 if step % 4 == 3 else root
                add(bar_start + step * BEAT * 0.25, bass_note(note, BEAT * 0.22), 0.25, 0.0)
            for step in range(8):
                add(bar_start + step * BEAT * 0.5, pluck(chord[step % len(chord)] + 12, 0.28), 0.85, -0.2)
            if bar == 18:
                add(bar_start + BEAT * 2.0, rise, 0.75, 0.0)
        elif limit:
            if bar == 19:
                add(bar_start, impact, 1.0, 0.0)
                for beat_index in range(1, 4):
                    add(bar_start + beat_index * BEAT, kick, 0.78, 0.0)
                add(bar_start, choir([57, 60, 64], 2.4, 1.1), 0.8, 0.0)
            else:
                for beat_index in (0, 2, 3):
                    add(bar_start + beat_index * BEAT, kick, 0.9, 0.0)
                add(bar_start + BEAT * 2.5, kick, 0.52, 0.0)
            for beat_index in (1, 3):
                add(bar_start + beat_index * BEAT, snare, 0.44, 0.12)
            for step in range(16):
                add(bar_start + step * BEAT * 0.25, hat, 0.095 if step % 2 else 0.16, -0.2)
            for step in range(16):
                note = root + 12 if step % 8 in (3, 7) else (root + 7 if step % 4 == 2 else root)
                add(bar_start + step * BEAT * 0.25, bass_note(note, BEAT * 0.2), 0.24, 0.0)
            if bar == 33:
                add(bar_start + BEAT * 2.0, rise, 0.8, 0.0)
        else:
            if bar == 34:
                add(bar_start, impact, 1.05, 0.0)
                add(bar_start, choir([57, 60, 64, 69], 3.4, 1.3), 0.95, 0.0)
            for beat_index in range(4):
                add(bar_start + beat_index * BEAT, kick, 0.78 if bar < 42 else 0.6, 0.0)
            for beat_index in (1, 3):
                add(bar_start + beat_index * BEAT, snare, 0.36, -0.12)
            for step in range(8):
                add(bar_start + step * BEAT * 0.5, pluck(chord[step % len(chord)] + 12, 0.4), 0.95, 0.25)
            if bar % 2 == 0:
                add(bar_start + BEAT * 2.0, choir([chord[0], chord[2], chord[3]], 2.2, 0.9), 0.75, 0.0)
            if bar == 43:
                add(bar_start, impact, 0.95, 0.0)

    drop_start = 3.0
    add(drop_start - 0.02, impact, 1.25, 0.0)
    add(drop_start - 0.30, rise, 0.5, 0.0)

    gate_end = int(2.86 * AUDIO_RATE)
    if gate_end > 0:
        mix[:gate_end] = 0.0
        for beat_index in range(3):
            add(beat_index * 1.0, heart, 0.12, 0.0)
        add(0.0, wind[: int(2.86 * AUDIO_RATE)], 0.045, 0.0)

    mix = np.tanh(mix * 1.2)
    peak = float(np.max(np.abs(mix))) if mix.size else 1.0
    if peak > 0:
        mix *= 0.50 / peak
    fade_out = int(min(3.0, total_seconds * 0.2) * AUDIO_RATE)
    if fade_out > 0:
        mix[-fade_out:] *= np.linspace(1.0, 0.0, fade_out)[:, None]
    return mix.astype(np.float32)


def score_clip(samples):
    length = samples.shape[0]

    def build_frame(times):
        array = np.asarray(times, dtype=np.float64)
        if array.ndim == 0:
            index = int(round(float(array) * AUDIO_RATE))
            return samples[min(max(index, 0), length - 1)]
        indices = np.clip(np.round(array.ravel() * AUDIO_RATE).astype(np.int64), 0, length - 1)
        return samples[indices]

    return AudioClip(frame_function=build_frame, duration=length / float(AUDIO_RATE), fps=AUDIO_RATE)


def grab_output(path, moment):
    result = subprocess.run(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-ss", "%.3f" % max(0.0, moment), "-i", path, "-frames:v", "1", "-pix_fmt", "bgr24", "-f", "rawvideo", "-"],
        capture_output=True,
    )
    data = result.stdout
    if len(data) < WIDTH * HEIGHT * 3:
        return None
    return np.frombuffer(data[: WIDTH * HEIGHT * 3], dtype=np.uint8).reshape(HEIGHT, WIDTH, 3)


def channel_offset(band, maximum=36):
    red = band[:, :, 2].astype(np.float32)
    blue = band[:, :, 0].astype(np.float32)
    red = red[::2, ::2]
    blue = blue[::2, ::2]
    red = red - red.mean()
    blue = blue - blue.mean()
    best_offset = 0
    best_score = -1e18
    for shift in range(-maximum, maximum + 1):
        shifted = np.roll(blue, shift, axis=1)
        score = float(np.mean(red * shifted))
        if score > best_score:
            best_score = score
            best_offset = shift
    return best_offset


def fx_unit_checks():
    base = np.full((CONTENT_H, WIDTH, 3), 18, np.uint8)
    cv2.circle(base, (int(WIDTH * 0.30), int(CONTENT_H * 0.35)), 40, (255, 255, 255), -1)
    cv2.rectangle(base, (int(WIDTH * 0.62), int(CONTENT_H * 0.52)), (int(WIDTH * 0.73), int(CONTENT_H * 0.76)), (235, 235, 235), -1)
    gray0 = cv2.cvtColor(base, cv2.COLOR_RGB2GRAY).astype(np.float32)
    halo_before = float(((gray0 > 26) & (gray0 < 200)).mean())
    bloomed = bloom_pass(base, 0.6)
    gray1 = cv2.cvtColor(bloomed, cv2.COLOR_RGB2GRAY).astype(np.float32)
    halo_after = float(((gray1 > 26) & (gray1 < 200)).mean())
    glitched = glitch_pass(base, 0.9, 0.1, 3)
    split = channel_offset(glitched)
    warped = lens_warp(base, 0.5)
    warp_delta = float(np.abs(warped.astype(np.int16) - base.astype(np.int16)).mean())
    blurred = motion_blur_pass(base, 1.0)
    blur_delta = float(np.abs(blurred.astype(np.int16) - base.astype(np.int16)).mean())
    return halo_before, halo_after, split, warp_delta, blur_delta


def verify(path, shots, total_seconds, score_samples):
    info = subprocess.run([FFMPEG, "-hide_banner", "-i", path], capture_output=True, text=True, errors="replace").stderr
    duration = 0.0
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", info)
    if match:
        duration = int(match.group(1)) * 3600 + int(match.group(2)) * 60 + float(match.group(3))
    size = os.path.getsize(path)
    log("file           : " + path)
    log("size           : %.2f MB" % (size / 1048576.0))
    log("duration       : %.2f s" % duration)
    log("streams        : video %d | audio %d" % (len(re.findall(r"Stream\s+\x23\d+:\d+.*: Video:", info)), len(re.findall(r"Stream\s+\x23\d+:\d+.*: Audio:", info))))
    decode = subprocess.run([FFMPEG, "-hide_banner", "-y", "-i", path, "-f", "null", "-"], capture_output=True, text=True, errors="replace")
    problems = [line for line in decode.stderr.splitlines() if re.search(r"\b(error|corrupt|invalid)\b", line, re.IGNORECASE)]
    frames = [int(value) for value in re.findall(r"frame=\s*(\d+)", decode.stderr)]
    decoded = max(frames) if frames else 0
    expected = int(round(duration * FPS))
    log("frames decoded : %d (expected %d) | decode issues %d" % (decoded, expected, len(problems)))

    letterbox_ok = False
    grade_ok = False
    bloom_ok = False
    sample = None
    best_spread = -1.0
    for moment in (4.5, 9.0, 21.0, 33.0, 46.0, 58.0, 71.0, 82.0):
        if moment >= duration - 0.2:
            continue
        candidate = grab_output(path, moment)
        if candidate is None:
            continue
        spread = float(candidate[BAR_H : HEIGHT - BAR_H].astype(np.float32).std())
        if spread > best_spread:
            best_spread = spread
            sample = candidate
    if sample is not None:
        content = sample[BAR_H : HEIGHT - BAR_H]
        top = sample[:BAR_H].astype(np.float32)
        bottom = sample[HEIGHT - BAR_H :].astype(np.float32)
        log("letterbox      : bars %.2f / %.2f | ratio %.3f:1" % (float(top.mean()), float(bottom.mean()), WIDTH / float(CONTENT_H)))
        letterbox_ok = float(top.mean()) <= 2.0 and float(bottom.mean()) <= 2.0 and abs(WIDTH / float(CONTENT_H) - RATIO) < 0.02
        gray = content.astype(np.float32).mean(axis=2)
        low_mask = gray < np.percentile(gray, 25)
        high_mask = gray > np.percentile(gray, 75)
        low = content[low_mask].astype(np.float32).mean(axis=0) if low_mask.any() else np.zeros(3)
        high = content[high_mask].astype(np.float32).mean(axis=0) if high_mask.any() else np.zeros(3)
        log("grade          : shadows B%.0f G%.0f R%.0f | highlights B%.0f G%.0f R%.0f" % (low[0], low[1], low[2], high[0], high[1], high[2]))
        lut_ok = int(GRADE[24, 0, 2]) > int(GRADE[24, 0, 0]) and int(GRADE[232, 0, 0]) > int(GRADE[232, 0, 2])
        cast = float(np.ptp(content.reshape(-1, 3).astype(np.float32).mean(axis=0)))
        log("grade curve    : shadow B%d/R%d | highlight R%d/B%d | frame cast %.1f" % (int(GRADE[24, 0, 2]), int(GRADE[24, 0, 0]), int(GRADE[232, 0, 0]), int(GRADE[232, 0, 2]), cast))
        grade_ok = lut_ok and cast > 4.0
        bright = float((gray > 232.0).mean())
        halo = float(((gray > 190.0) & (gray <= 232.0)).mean())
        log("bloom          : saturated %.4f | halo %.4f | frame spread %.1f" % (bright, halo, best_spread))

    base_probe = np.full((CONTENT_H, WIDTH, 3), 60, np.uint8)
    cv2.rectangle(base_probe, (200, 120), (900, 640), (220, 220, 220), -1)
    soft_probe = cv2.GaussianBlur(base_probe, (0, 0), 2.4)
    sharp_probe = usm_sharpen(soft_probe)
    soft_grad = float(np.abs(np.diff(soft_probe.astype(np.float32), axis=1)).mean())
    sharp_grad = float(np.abs(np.diff(sharp_probe.astype(np.float32), axis=1)).mean())
    sharpen_ok = sharp_grad > soft_grad * 1.02
    log("sharpen unit   : gradient %.3f -> %.3f (%.2fx)" % (soft_grad, sharp_grad, sharp_grad / max(soft_grad, 1e-6)))
    halo_before, halo_after, unit_split, warp_delta, blur_delta = fx_unit_checks()
    log("fx unit        : bloom halo %.4f -> %.4f | rgb split %+d px | warp delta %.2f | blur delta %.2f" % (halo_before, halo_after, unit_split, warp_delta, blur_delta))
    bloom_ok = halo_after > max(halo_before * 2.0, 0.004) and unit_split >= 8 and warp_delta > 1.0 and blur_delta > 1.0

    glitch_shots = [shot for shot in shots if shot["glitch"] >= 0.5]
    measured = []
    for shot in glitch_shots[:8]:
        moment = shot["start"] + 0.10
        if moment >= duration - 0.2:
            continue
        frame = grab_output(path, moment)
        if frame is None:
            continue
        measured.append((round(moment, 2), channel_offset(frame[BAR_H : HEIGHT - BAR_H])))
    splits = [value for moment, value in measured if abs(value) >= 2]
    log("rgb split      : " + ", ".join("%.2fs=%+d" % (moment, value) for moment, value in measured))
    log("glitch hits    : %d/%d sampled shots show channel separation" % (len(splits), len(measured)))

    warp_shots = [shot for shot in shots if shot["warp"] > 0.2]
    shake_shots = [shot for shot in shots if shot["shake"] > 0.2]
    mask_shots = [shot for shot in shots if shot["mask"]]
    blur_shots = [shot for shot in shots if shot["blur"] > 0.3]
    pan_shots = [shot for shot in shots if abs(shot["motion"][3] - shot["motion"][2]) > 0.2 or abs(shot["motion"][5] - shot["motion"][4]) > 0.2]
    log("manifest       : glitch %d | warp %d | shake %d | textmask %d | motionblur %d | pan %d" % (len(glitch_shots), len(warp_shots), len(shake_shots), len(mask_shots), len(blur_shots), len(pan_shots)))
    ramp_shots = [shot for shot in shots if shot["kind"] == "video" and abs(shot["speed"][1] - shot["speed"][0]) >= 1.5]
    log("speed ramps    : %d shots with ratio >= 1.5x | max %.2fx" % (len(ramp_shots), max([max(shot["speed"][0], shot["speed"][1]) / max(1e-6, min(shot["speed"][0], shot["speed"][1])) for shot in shots if shot["kind"] == "video"] + [1.0])))

    offsets = [abs(shot["start"] / BEAT - round(shot["start"] / BEAT)) for shot in shots[1:]]
    grid_ok = max(offsets) < 1e-6 if offsets else False
    log("beat grid      : %d cuts | max off-grid %.6f beats" % (len(shots) - 1, max(offsets) if offsets else 0.0))

    def window_rms(samples, start, span):
        first = int(start * AUDIO_RATE)
        last = int((start + span) * AUDIO_RATE)
        segment = samples[max(0, first) : min(len(samples), last)]
        return float(np.sqrt((segment.astype(np.float64) ** 2).mean())) if segment.size else 0.0

    intro_rms = window_rms(score_samples, 0.2, 2.4)
    drop_rms = window_rms(score_samples, 3.05, 1.0)
    body_rms = window_rms(score_samples, 20.0, 4.0)
    log("score dynamics : intro %.4f | drop %.4f | body %.4f | ratio %.1fx" % (intro_rms, drop_rms, body_rms, drop_rms / max(intro_rms, 1e-6)))
    drop_ok = drop_rms > max(0.04, intro_rms * 8.0) and intro_rms < 0.03

    pcm = subprocess.run(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-i", path, "-vn", "-ac", "1", "-ar", str(AUDIO_RATE), "-f", "f32le", "-acodec", "pcm_f32le", "-"],
        capture_output=True,
    ).stdout
    decoded_audio = np.frombuffer(pcm, dtype=np.float32) if pcm else np.zeros(1, dtype=np.float32)
    absolute = np.abs(decoded_audio.astype(np.float64))
    span = 441
    cumulative = np.cumsum(np.insert(absolute, 0, 0.0))
    lowpassed = (cumulative[span:] - cumulative[:-span]) / span
    block = 441
    block_count = max(1, len(lowpassed) // block)
    envelope = lowpassed[: block_count * block].reshape(block_count, block).max(axis=1)
    derivative = np.diff(envelope)
    threshold = float(derivative.mean() + 2.0 * derivative.std())
    kicks = []
    for position in np.where(derivative > threshold)[0]:
        moment = position * 0.01 + 0.02
        if not kicks or moment - kicks[-1] > 0.12:
            kicks.append(float(moment))
    hit = 0
    cut_count = 0
    for shot in shots[1:]:
        if shot["start"] >= duration - 0.2 or not kicks:
            continue
        cut_count += 1
        if min(abs(k - shot["start"]) for k in kicks) <= 0.05:
            hit += 1
    sync = 0.0 if cut_count == 0 else hit / float(cut_count)
    level = subprocess.run([FFMPEG, "-hide_banner", "-i", path, "-af", "volumedetect", "-f", "null", "-"], capture_output=True, text=True, errors="replace").stderr
    values = {}
    for line in level.splitlines():
        match = re.search(r"(mean|max)_volume:\s*(-?[\d.]+)", line)
        if match:
            values[match.group(1)] = float(match.group(2))
    log("audio level    : mean %.1f dB | peak %.1f dB" % (values.get("mean", 0.0), values.get("max", 0.0)))
    log("cut sync       : %d/%d cuts on a kick (%.0f%%) | kicks %d" % (hit, cut_count, 100.0 * sync, len(kicks)))

    checks = {
        "duration": abs(duration - total_seconds) <= 1.0,
        "resolution": bool(re.search(r"1920x1080", info)),
        "streams": "Video:" in info and "Audio:" in info,
        "decode": len(problems) == 0 and abs(decoded - expected) <= max(3, expected * 0.01),
        "letterbox": letterbox_ok,
        "grade": grade_ok,
        "bloom": bloom_ok,
        "rgb_split": len(splits) >= 3,
        "text_mask": len(mask_shots) >= 1,
        "pan_scan": len(pan_shots) >= 10,
        "shake": len(shake_shots) >= 8,
        "sharpen": sharpen_ok,
        "motion_blur": len(blur_shots) >= 4,
        "drop": drop_ok,
        "beat_grid": grid_ok,
        "cut_sync": sync >= 0.8,
        "audio": values.get("max", -99.0) < -0.5,
    }
    log("checks         : " + " ".join("%s=%s" % (key, value) for key, value in checks.items()))
    return checks, size, duration


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    stage("STAGE 1/7  ENVIRONMENT")
    log("ffmpeg         : " + FFMPEG)
    log("opencv         : " + cv2.__version__)
    log("tempo          : %d BPM | %d beats = %.1f s" % (BPM, TOTAL_BEATS, TOTAL_BEATS * BEAT))
    log("frame          : %dx%d | visible %dx%d (%.3f:1) | bars %d px" % (WIDTH, HEIGHT, WIDTH, CONTENT_H, WIDTH / float(CONTENT_H), BAR_H))
    log("fx banks       : warp maps %d | leaks %d | grain %d" % (len(WARPS), len(LEAKS), len(GRAIN)))
    if os.path.isdir(WORK_DIR):
        shutil.rmtree(WORK_DIR)
    os.makedirs(WORK_DIR)
    stage_done("toolchain ready")

    stage("STAGE 2/7  CHRONOLOGICAL INGEST")
    media = collect_media()
    log("source media   : %d files" % len(media))
    log("time span      : %s -> %s" % (media[0]["stamp"].strftime("%Y-%m-%d %H:%M:%S"), media[-1]["stamp"].strftime("%Y-%m-%d %H:%M:%S")))
    annotate_media(media)
    kept, dropped = dedupe_consecutive(media)
    log("consecutive dup: %d dropped -> %d kept (order preserved)" % (dropped, len(kept)))
    groups = split_chapters(kept, len(CHAPTERS))
    for index, group in enumerate(groups):
        log(
            "  %-9s %3d items | %s -> %s"
            % (
                CHAPTERS[index]["name"],
                len(group),
                group[0]["stamp"].strftime("%m-%d %H:%M") if group else "-",
                group[-1]["stamp"].strftime("%m-%d %H:%M") if group else "-",
            )
        )
    stage_done("%d files resolved to a single ascending timeline" % len(kept))

    stage("STAGE 3/7  TIMELINE AND FX MAP")
    shots, beat_total = build_timeline(groups)
    shots = apply_mask_plan(shots)
    if beat_total != TOTAL_BEATS:
        raise RuntimeError("beat mismatch %d" % beat_total)
    total_seconds = beat_total * BEAT
    hero_count = sum(1 for shot in shots if shot["hero"])
    glitch_count = sum(1 for shot in shots if shot["glitch"] > 0.02)
    log("shots          : %d | hero3d %d | glitch %d | textmask %d" % (len(shots), hero_count, glitch_count, len(MASK_PLAN)))
    log("text cards     : %d" % len(CARDS))
    for key in ("school_cn", "school_en", "class_en", "class_cn", "motto"):
        card = next(item for item in CARDS if item["key"] == key)
        log("  card %-9s %s" % (key, card["text"]))
    for index, chapter in enumerate(CHAPTERS):
        log(
            "  chapter %-9s %5.1f-%5.1f s | %2d shots | %s"
            % (
                chapter["name"],
                CHAPTER_START[index],
                CHAPTER_START[index] + sum(chapter["beats"]) * BEAT,
                len(chapter["beats"]),
                chapter["mode"],
            )
        )
    ordered = all(shots[index]["stamp"] <= shots[index + 1]["stamp"] for index in range(len(shots) - 1))
    log("order check    : %s (shots strictly ascending in capture time)" % ("PASS" if ordered else "FAIL"))
    if not ordered:
        raise RuntimeError("timeline order violated")
    stage_done("shot list locked to beat grid")

    stage("STAGE 4/7  SCORE SYNTHESIS")
    score = build_score(total_seconds)
    log("score          : %.2f s stereo | peak %.3f" % (score.shape[0] / float(AUDIO_RATE), float(np.abs(score).max())))
    stage_done("epic bed with choir, wind and drop")

    stage("STAGE 5/7  RENDER")
    render_seconds = total_seconds
    if LIMIT_SECONDS > 0:
        render_seconds = min(LIMIT_SECONDS, total_seconds)
        log("debug limit    : %.2f s" % render_seconds)
    clip, _ = build_clip(shots, total_seconds)
    if render_seconds < total_seconds:
        clip = clip.with_duration(render_seconds)
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_NAME)
    silent_path = os.path.join(WORK_DIR, "video_only.mp4")
    audio_path = os.path.join(WORK_DIR, "score.m4a")
    soundtrack = score_clip(score)
    if render_seconds < total_seconds:
        soundtrack = soundtrack.with_duration(render_seconds)
    soundtrack.write_audiofile(audio_path, fps=AUDIO_RATE, codec="aac", bitrate="256k", logger=None)
    log("audio track    : %.2f MB" % (os.path.getsize(audio_path) / 1048576.0))
    del soundtrack
    clip.write_videofile(
        silent_path,
        fps=FPS,
        codec="libx264",
        bitrate=BITRATE,
        audio=False,
        preset="medium",
        threads=3,
        pixel_format="yuv420p",
        ffmpeg_params=["-profile:v", "high", "-level", "4.2", "-g", "30", "-bf", "2"],
    )
    clip.close()
    del clip
    log("video track    : %.2f MB" % (os.path.getsize(silent_path) / 1048576.0))
    mux = subprocess.run(
        [
            FFMPEG,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            silent_path,
            "-i",
            audio_path,
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "copy",
            "-movflags",
            "+faststart",
            "-shortest",
            output_path,
        ],
        capture_output=True,
        text=True,
        errors="replace",
    )
    if mux.returncode != 0:
        raise RuntimeError("mux failed: " + mux.stderr[-400:])
    stage_done("master encoded")

    stage("STAGE 6/7  VERIFICATION")
    checks, size, duration = verify(output_path, shots, render_seconds, score)
    stage_done("verified")

    stage("STAGE 7/7  CLEANUP")
    shutil.rmtree(WORK_DIR, ignore_errors=True)
    log("cleanup        : workspace removed")
    passed = all(checks.values())
    print("", flush=True)
    log("RESULT         : " + ("PASS" if passed else "FAIL"))
    log("OUTPUT         : %s | %.2f MB | %.2f s | 1920x1080 | 30 fps | h264 %s | aac" % (OUTPUT_NAME, size / 1048576.0, duration, BITRATE))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
