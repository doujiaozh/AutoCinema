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
OUTPUT_NAME = "Class20_Sports_Cinema.mp4"
WORK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_render")

SCHOOL_EN = "HAIKOU JINPAN"
SCHOOL_CN = "海口市金盘实验学校"
CLASS_EN = "CLASS 20"
CLASS_CN = "七(20)班"
MOTTO = "NEVER GIVE UP"

FONT_CN = r"C:\Windows\Fonts\msyhbd.ttc"
FONT_HEAVY = r"C:\Windows\Fonts\impact.ttf"
FONT_SANS = r"C:\Windows\Fonts\arialbd.ttf"

TARGET_IMAGES = 44
TARGET_VIDEO_CLIPS = 12
PHASH_DUPLICATE = 12
PHASH_DIVERSITY = 18
GRAIN_PLATES = 8
GRAIN_STRENGTH = 0.34
LIMIT_SECONDS = float(os.environ.get("AUTOCINEMA_LIMIT", "0"))

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


def list_sources(extension):
    found = []
    for entry in os.scandir(SOURCE_DIR):
        if not entry.is_file():
            continue
        name = entry.name
        if not name.lower().endswith(extension) or name.startswith("_"):
            continue
        if name.startswith("Final_") or name.startswith("Class20_"):
            continue
        found.append(os.path.join(SOURCE_DIR, name))
    found.sort(key=natural_key)
    return found


def probe_duration(path):
    result = subprocess.run([FFMPEG, "-hide_banner", "-i", path], capture_output=True, text=True, errors="replace")
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", result.stderr)
    if not match:
        return 0.0
    return int(match.group(1)) * 3600 + int(match.group(2)) * 60 + float(match.group(3))


def decode_full(path):
    with Image.open(path) as handle:
        width, height = handle.size
    flag = cv2.IMREAD_COLOR if width * height <= 20000000 else cv2.IMREAD_REDUCED_COLOR_2
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), flag)
    if image is None:
        return None
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def decode_thumb(path, limit=960):
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
    sharp = float(cv2.Laplacian(blurred, cv2.CV_64F).var())
    contrast = float(gray.std())
    brightness = float(gray.mean())
    warmth = float(thumb[:, :, 0].mean() - thumb[:, :, 2].mean())
    return {
        "path": path,
        "hash": phash(thumb),
        "sharpness": sharp,
        "contrast": contrast,
        "brightness": brightness,
        "warmth": warmth,
        "width": int(thumb.shape[1]),
        "height": int(thumb.shape[0]),
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
    difference = np.abs(np.diff(frames, axis=0)).mean(axis=(1, 2))
    return difference, fps, count


def best_window(difference, fps, seconds):
    span = max(2, int(round(seconds * fps)))
    if len(difference) < span:
        return 0.0, float(difference.mean())
    smoothed = np.convolve(difference, np.ones(span) / span, mode="valid")
    index = int(np.argmax(smoothed))
    return index / float(fps), float(smoothed[index])


def grab_frame(path, moment, width=320, height=180):
    command = [
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
    ]
    result = subprocess.run(command, capture_output=True)
    data = result.stdout
    if len(data) < width * height * 3:
        return None
    return np.frombuffer(data[: width * height * 3], dtype=np.uint8).reshape(height, width, 3)


def select_images(paths):
    records = []
    for index, path in enumerate(paths):
        record = analyse_image(path)
        if record:
            records.append(record)
        if (index + 1) % 40 == 0:
            log("  analysed %d/%d images" % (index + 1, len(paths)))
    count = len(records)
    distance = np.zeros((count, count))
    for first in range(count):
        for second in range(first + 1, count):
            value = hamming(records[first]["hash"], records[second]["hash"])
            distance[first, second] = value
            distance[second, first] = value
    tree = linkage(squareform(distance, checks=False), method="complete")
    labels = fcluster(tree, t=PHASH_DUPLICATE, criterion="distance")
    clusters = {}
    for record, label in zip(records, labels):
        clusters.setdefault(int(label), []).append(record)
    representatives = []
    for group in clusters.values():
        usable = [r for r in group if 18.0 <= r["brightness"] <= 242.0]
        pool = usable if usable else group
        best = max(pool, key=lambda r: r["sharpness"] * 0.6 + r["contrast"] * 0.4)
        best["cluster_size"] = len(group)
        representatives.append(best)
    representatives.sort(key=lambda r: -(r["sharpness"] * 0.55 + r["contrast"] * 0.45))
    chosen = []
    for candidate in representatives:
        if all(hamming(candidate["hash"], other["hash"]) >= PHASH_DIVERSITY for other in chosen):
            chosen.append(candidate)
        if len(chosen) >= TARGET_IMAGES:
            break
    chosen.sort(key=lambda r: natural_key(r["path"]))
    return chosen, {"analysed": count, "clusters": len(representatives), "selected": len(chosen)}


def select_videos(paths):
    candidates = []
    for index, path in enumerate(paths):
        profile = motion_profile(path)
        if profile is None:
            continue
        difference, fps, frames = profile
        start, score = best_window(difference, fps, 2.0)
        duration = frames / float(fps)
        start = min(max(0.0, start), max(0.0, duration - 4.0))
        thumb = grab_frame(path, start + 1.0)
        candidates.append(
            {
                "path": path,
                "start": round(start, 3),
                "peak": round(score, 3),
                "mean": round(float(difference.mean()), 3),
                "duration": round(duration, 2),
                "hash": phash(thumb) if thumb is not None else None,
            }
        )
        if (index + 1) % 6 == 0:
            log("  scanned %d/%d videos" % (index + 1, len(paths)))
    usable = [c for c in candidates if c["peak"] >= 8.0 and c["mean"] >= 4.0]
    usable.sort(key=lambda c: -c["peak"])
    chosen = []
    for candidate in usable:
        if candidate["hash"] is not None:
            if any(hamming(candidate["hash"], other["hash"]) < PHASH_DUPLICATE for other in chosen if other["hash"] is not None):
                continue
        chosen.append(candidate)
        if len(chosen) >= TARGET_VIDEO_CLIPS:
            break
    return chosen


def build_lut():
    x = np.arange(256, dtype=np.float64) / 255.0
    y = 0.5 + (x - 0.5) * 1.22
    y = np.clip(y, 0.0, 1.0)
    y = y + 0.018 * (1.0 - y) * (1.0 - y)
    y = np.clip(y, 0.0, 1.0)
    knee = 0.82
    over = np.maximum(y - knee, 0.0)
    y = np.where(y > knee, knee + over / (1.0 + 3.2 * over), y)
    base = np.clip(y, 0.0, 1.0) * 255.0
    shade = (1.0 - x) ** 2
    light = x ** 2
    table = np.zeros((256, 1, 3), dtype=np.uint8)
    table[:, 0, 0] = np.clip(base - 11.0 * shade + 15.0 * light, 0, 255)
    table[:, 0, 1] = np.clip(base + 4.0 * shade + 5.0 * light, 0, 255)
    table[:, 0, 2] = np.clip(base + 15.0 * shade - 13.0 * light, 0, 255)
    return table


def build_vignette():
    yy, xx = np.mgrid[0:CONTENT_H, 0:WIDTH]
    nx = (xx - (WIDTH - 1) / 2.0) / ((WIDTH - 1) / 2.0)
    ny = (yy - (CONTENT_H - 1) / 2.0) / ((CONTENT_H - 1) / 2.0)
    radius = np.sqrt(nx * nx * 0.9 + ny * ny * 0.9)
    falloff = 1.0 - 0.34 * np.clip(radius - 0.48, 0.0, None) ** 1.5
    falloff = np.clip(falloff, 0.62, 1.0)
    mask = np.repeat(falloff[:, :, None], 3, axis=2)
    return np.ascontiguousarray((mask * 255.0).astype(np.uint8))


def build_grain():
    rng = np.random.RandomState(20261127)
    plates = []
    for index in range(GRAIN_PLATES):
        noise = rng.normal(128.0, 16.0, (CONTENT_H, WIDTH, 3))
        plates.append(np.clip(noise, 0, 255).astype(np.uint8))
    return plates


GRADE = build_lut()
VIGNETTE = build_vignette()
GRAIN = build_grain()
WHITE = np.full((CONTENT_H, WIDTH, 3), 255, dtype=np.uint8)

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
        raise RuntimeError("unreadable image " + os.path.basename(path))
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


def crop_fill(source, scale, center_x, center_y):
    height, width = source.shape[:2]
    window_w = min(width / scale, height * RATIO)
    window_h = window_w / RATIO
    if window_h > height:
        window_h = height
        window_w = window_h * RATIO
    left = center_x * width - window_w / 2.0
    top = center_y * height - window_h / 2.0
    left = min(max(left, 0.0), max(0.0, width - window_w))
    top = min(max(top, 0.0), max(0.0, height - window_h))
    x0 = int(round(left))
    y0 = int(round(top))
    x1 = min(width, max(x0 + 2, int(round(left + window_w))))
    y1 = min(height, max(y0 + 2, int(round(top + window_h))))
    patch = source[y0:y1, x0:x1]
    interpolation = cv2.INTER_AREA if patch.shape[1] >= WIDTH else cv2.INTER_LINEAR
    return cv2.resize(patch, (WIDTH, CONTENT_H), interpolation=interpolation)


def grade_frame(frame):
    graded = cv2.LUT(frame, GRADE)
    gray = cv2.cvtColor(graded, cv2.COLOR_RGB2GRAY)
    gray3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    graded = cv2.addWeighted(graded, 1.11, gray3, -0.11, 0)
    return graded


def finish_frame(frame, index, strength):
    frame = cv2.addWeighted(frame, 1.0, GRAIN[index % GRAIN_PLATES], strength, -128.0 * strength)
    frame = cv2.multiply(frame, VIGNETTE, scale=1.0 / 255.0)
    return frame


def make_card(text, font_path, size, stroke_ratio, fill=(255, 255, 255)):
    font = ImageFont.truetype(font_path, size)
    stroke = max(2, int(round(size * stroke_ratio)))
    probe = ImageDraw.Draw(Image.new("RGBA", (16, 16)))
    box = probe.textbbox((0, 0), text, font=font, stroke_width=stroke)
    pad = stroke * 3 + 8
    width = box[2] - box[0] + pad * 2
    height = box[3] - box[1] + pad * 2
    canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
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
    return {"rgb": np.ascontiguousarray(array[:, :, :3]), "alpha": array[:, :, 3].astype(np.float32) / 255.0}


def build_cards():
    cards = []

    def add(key, text, font, size, cx, cy, start, end, fade=0.5, amp=0.045, colour=(255, 255, 255)):
        cards.append(
            {
                "key": key,
                "text": text,
                "card": make_card(text, font, size, 0.055, colour),
                "cx": cx,
                "cy": cy,
                "start": start,
                "end": end,
                "fade": fade,
                "amp": amp,
            }
        )

    add("school_en", SCHOOL_EN, FONT_HEAVY, 168, 0.5, 0.40, 0.7, 4.9, 0.55, 0.035)
    add("school_cn", SCHOOL_CN, FONT_CN, 68, 0.5, 0.62, 1.1, 4.9, 0.55, 0.035)
    add("m1_en", "I. DAWN", FONT_SANS, 96, 0.24, 0.72, 6.2, 9.4, 0.45, 0.03)
    add("m1_cn", "黎明", FONT_CN, 60, 0.24, 0.82, 6.2, 9.4, 0.45, 0.03)
    add("m2_en", "II. SUN", FONT_SANS, 96, 0.24, 0.72, 18.4, 21.4, 0.4, 0.03)
    add("m2_cn", "烈日", FONT_CN, 60, 0.24, 0.82, 18.4, 21.4, 0.4, 0.03)
    add("grit", "GRIT", FONT_HEAVY, 210, 0.5, 0.42, 24.0, 27.4, 0.45, 0.05)
    add("sweat_word", "SWEAT", FONT_HEAVY, 210, 0.5, 0.42, 33.0, 36.4, 0.45, 0.05)
    add("m3_en", "III. LIMIT", FONT_SANS, 96, 0.24, 0.72, 42.4, 45.6, 0.4, 0.03)
    add("m3_cn", "汗水", FONT_CN, 60, 0.24, 0.82, 42.4, 45.6, 0.4, 0.03)
    add("motto", MOTTO, FONT_HEAVY, 150, 0.5, 0.45, 52.0, 56.4, 0.5, 0.05)
    add("m4_en", "IV. GLORY", FONT_SANS, 96, 0.24, 0.72, 66.4, 69.6, 0.4, 0.03)
    add("m4_cn", "荣光", FONT_CN, 60, 0.24, 0.82, 66.4, 69.6, 0.4, 0.03)
    add("class_en", CLASS_EN, FONT_HEAVY, 300, 0.5, 0.40, 70.2, 75.6, 0.55, 0.055)
    add("class_cn", CLASS_CN, FONT_CN, 118, 0.5, 0.66, 70.7, 75.6, 0.55, 0.055)
    add("glory", "GLORY", FONT_HEAVY, 200, 0.5, 0.42, 79.5, 84.2, 0.5, 0.05)
    add("end_cn", SCHOOL_CN, FONT_CN, 76, 0.5, 0.44, 85.6, 89.4, 0.6, 0.03)
    add("end_class", CLASS_CN, FONT_CN, 54, 0.5, 0.58, 86.0, 89.4, 0.6, 0.03)
    return cards


CARDS = build_cards()


def overlay_cards(frame, moment):
    for card in CARDS:
        if moment < card["start"] or moment > card["end"]:
            continue
        local = moment - card["start"]
        span = card["end"] - card["start"]
        alpha = 1.0
        if local < card["fade"]:
            alpha = local / card["fade"]
        if span - local < card["fade"]:
            alpha = min(alpha, (span - local) / card["fade"])
        if alpha <= 0.01:
            continue
        scale = 1.0 + card["amp"] * math.sin(2.0 * math.pi * 0.26 * local)
        rgb = card["card"]["rgb"]
        mask = card["card"]["alpha"]
        target_w = max(2, int(round(rgb.shape[1] * scale)))
        target_h = max(2, int(round(rgb.shape[0] * scale)))
        if target_w > WIDTH:
            target_w = WIDTH
            target_h = max(2, int(round(rgb.shape[0] * target_w / float(rgb.shape[1]))))
        scaled_rgb = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)
        scaled_alpha = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_AREA) * alpha
        center_x = int(round(card["cx"] * WIDTH))
        center_y = int(round((card["cy"] + 0.006 * math.sin(2.0 * math.pi * 0.13 * local)) * CONTENT_H))
        left = center_x - target_w // 2
        top = center_y - target_h // 2
        x0 = max(0, left)
        y0 = max(0, top)
        x1 = min(WIDTH, left + target_w)
        y1 = min(CONTENT_H, top + target_h)
        if x1 <= x0 or y1 <= y0:
            continue
        source_x0 = x0 - left
        source_y0 = y0 - top
        patch_rgb = scaled_rgb[source_y0 : source_y0 + (y1 - y0), source_x0 : source_x0 + (x1 - x0)]
        patch_alpha = scaled_alpha[source_y0 : source_y0 + (y1 - y0), source_x0 : source_x0 + (x1 - x0)]
        region = frame[y0:y1, x0:x1]
        blended = region.astype(np.float32) * (1.0 - patch_alpha[:, :, None]) + patch_rgb.astype(np.float32) * patch_alpha[:, :, None]
        frame[y0:y1, x0:x1] = np.clip(blended, 0, 255).astype(np.uint8)
    return frame


MOVEMENTS = [
    {
        "name": "DAWN",
        "cn": "黎明",
        "roman": "I",
        "beats": [8, 6, 6, 5, 5, 6],
        "start_beat": 0,
        "video_slots": 0,
        "energy": 0.35,
    },
    {
        "name": "SUN",
        "cn": "烈日",
        "roman": "II",
        "beats": [4, 3, 3, 3, 3, 2, 3, 3, 2, 3, 3, 2, 3, 3, 3, 3, 2],
        "start_beat": 36,
        "video_slots": 6,
        "energy": 0.8,
    },
    {
        "name": "SWEAT",
        "cn": "汗水",
        "roman": "III",
        "beats": [3, 2, 2, 3, 2, 2, 3, 2, 2, 3, 2, 2, 3, 2, 2, 3, 2, 2, 3, 3],
        "start_beat": 84,
        "video_slots": 4,
        "energy": 1.0,
    },
    {
        "name": "GLORY",
        "cn": "荣光",
        "roman": "IV",
        "beats": [6, 5, 4, 3, 3, 4, 3, 3, 4, 3, 3, 3, 4],
        "start_beat": 132,
        "video_slots": 2,
        "energy": 0.6,
    },
]

IMAGE_MOTIONS = [
    {"s0": 1.06, "s1": 1.20, "cx0": 0.5, "cx1": 0.5, "cy0": 0.46, "cy1": 0.46},
    {"s0": 1.20, "s1": 1.06, "cx0": 0.5, "cx1": 0.5, "cy0": 0.46, "cy1": 0.46},
    {"s0": 1.16, "s1": 1.16, "cx0": 0.57, "cx1": 0.43, "cy0": 0.47, "cy1": 0.47},
    {"s0": 1.16, "s1": 1.16, "cx0": 0.43, "cx1": 0.57, "cy0": 0.47, "cy1": 0.47},
    {"s0": 1.08, "s1": 1.22, "cx0": 0.44, "cx1": 0.56, "cy0": 0.58, "cy1": 0.42},
    {"s0": 1.22, "s1": 1.08, "cx0": 0.56, "cx1": 0.44, "cy0": 0.42, "cy1": 0.58},
    {"s0": 1.05, "s1": 1.34, "cx0": 0.5, "cx1": 0.5, "cy0": 0.5, "cy1": 0.44},
    {"s0": 1.30, "s1": 1.05, "cx0": 0.5, "cx1": 0.5, "cy0": 0.44, "cy1": 0.5},
]

SPEED_PROFILES = [
    {"s0": 0.32, "s1": 0.32, "label": "slowmo"},
    {"s0": 0.30, "s1": 2.60, "label": "ramp_up"},
    {"s0": 2.40, "s1": 1.00, "label": "fast_in"},
    {"s0": 1.00, "s1": 1.00, "label": "real"},
    {"s0": 1.00, "s1": 3.20, "label": "ramp_out"},
    {"s0": 0.45, "s1": 0.45, "label": "slow2"},
]


def build_timeline(images, videos):
    shots = []
    beat_cursor = 0
    image_index = 0
    video_index = 0
    order = 0
    for movement_index, movement in enumerate(MOVEMENTS):
        pattern = list(movement["beats"])
        slots = list(range(len(pattern)))
        video_slots = set()
        if movement["video_slots"] > 0 and len(pattern) > 0:
            step = max(1, len(pattern) // movement["video_slots"])
            for position in range(1, len(pattern), step):
                if len(video_slots) >= movement["video_slots"]:
                    break
                video_slots.add(position)
        for position, beats in enumerate(pattern):
            duration = beats * BEAT
            use_video = position in video_slots and videos
            shot = {
                "index": order,
                "movement": movement_index,
                "movement_name": movement["name"],
                "kind": "video" if use_video else "image",
                "beats": beats,
                "start": beat_cursor * BEAT,
                "duration": duration,
                "energy": movement["energy"],
            }
            if use_video:
                clip = videos[video_index % len(videos)]
                profile = SPEED_PROFILES[order % len(SPEED_PROFILES)]
                max_span = max(1.0, clip["duration"] - clip["start"] - 1.2)
                average_speed = (profile["s0"] + profile["s1"]) / 2.0
                needed = duration * average_speed
                if needed > max_span:
                    scale_back = max_span / needed
                    profile = {"s0": profile["s0"] * scale_back, "s1": profile["s1"] * scale_back, "label": profile["label"]}
                shot["path"] = clip["path"]
                shot["src_start"] = clip["start"]
                shot["speed"] = profile
                shot["motion"] = {
                    "s0": 1.04,
                    "s1": 1.12,
                    "cx0": 0.5,
                    "cx1": 0.5,
                    "cy0": 0.47,
                    "cy1": 0.47,
                }
                video_index += 1
            else:
                record = images[image_index % len(images)]
                shot["path"] = record["path"]
                shot["motion"] = IMAGE_MOTIONS[order % len(IMAGE_MOTIONS)]
                if movement["energy"] >= 0.95:
                    motion = dict(shot["motion"])
                    motion["s1"] = motion["s1"] * 1.12
                    shot["motion"] = motion
                image_index += 1
            if beat_cursor % 8 == 0:
                shot["transition"] = "flash_strong"
            elif beats <= 2 and movement["energy"] >= 0.95:
                shot["transition"] = "flash"
            elif order % 7 == 3:
                shot["transition"] = "punch"
            else:
                shot["transition"] = "cut"
            if order == 0:
                shot["transition"] = "open"
            shots.append(shot)
            beat_cursor += beats
            order += 1
    return shots, beat_cursor


def speed_offset(shot, local):
    duration = shot["duration"]
    s0 = shot["speed"]["s0"]
    s1 = shot["speed"]["s1"]
    ratio = 0.0 if duration <= 0 else min(1.0, max(0.0, local / duration))
    return s0 * local + (s1 - s0) * duration * ratio * ratio / 2.0


def build_clip(shots, total_seconds):
    starts = [shot["start"] for shot in shots]

    def locate(moment):
        index = bisect.bisect_right(starts, moment) - 1
        index = max(0, min(index, len(shots) - 1))
        return index, moment - starts[index]

    def build_frame(t):
        moment = float(t)
        index, local = locate(moment)
        shot = shots[index]
        progress = min(1.0, max(0.0, local / shot["duration"]))
        motion = shot["motion"]
        kick = 1.0
        if shot["transition"] == "punch" and local < 0.18:
            kick = 1.0 + 0.22 * (1.0 - local / 0.18)
        scale = (motion["s0"] + (motion["s1"] - motion["s0"]) * progress) * kick
        center_x = motion["cx0"] + (motion["cx1"] - motion["cx0"]) * progress
        center_y = motion["cy0"] + (motion["cy1"] - motion["cy0"]) * progress
        if shot["kind"] == "image":
            source = load_image(shot["path"])
        else:
            source = video_frame(shot["path"], shot["src_start"] + speed_offset(shot, local))
        frame = crop_fill(source, scale, center_x, center_y)
        frame = grade_frame(frame)
        grain = 1.0 if shot["kind"] == "video" else 0.78
        frame = finish_frame(frame, index, GRAIN_STRENGTH * grain)
        if shot["transition"] in ("flash", "flash_strong"):
            length = 0.16 if shot["transition"] == "flash_strong" else 0.12
            if local < length:
                weight = (0.96 if shot["transition"] == "flash_strong" else 0.78) * (1.0 - local / length)
                frame = cv2.addWeighted(frame, 1.0 - weight, WHITE, weight, 0)
        frame = overlay_cards(frame, moment)
        if moment < 0.7:
            frame = cv2.convertScaleAbs(frame, alpha=moment / 0.7, beta=0)
        tail = total_seconds - moment
        if tail < 1.6:
            frame = cv2.convertScaleAbs(frame, alpha=max(0.0, tail / 1.6), beta=0)
        canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        canvas[BAR_H : BAR_H + CONTENT_H] = frame
        return canvas

    return VideoClip(frame_function=build_frame, duration=total_seconds)


def build_score(total_seconds):
    sample_count = int(round(total_seconds * AUDIO_RATE))
    mix = np.zeros((sample_count, 2), dtype=np.float32)
    rng = np.random.RandomState(20261127)

    def timeline(duration):
        return np.arange(int(round(duration * AUDIO_RATE)), dtype=np.float64) / AUDIO_RATE

    def add(start_seconds, wave, gain, pan=0.0):
        start = int(round(start_seconds * AUDIO_RATE))
        if start < 0 or start >= sample_count or wave.size == 0:
            return
        length = min(wave.size, sample_count - start)
        left = gain * (1.0 - 0.28 * pan)
        right = gain * (1.0 + 0.28 * pan)
        mix[start : start + length, 0] += wave[:length] * left
        mix[start : start + length, 1] += wave[:length] * right

    kick_time = timeline(0.55)
    kick_freq = 44.0 + 165.0 * np.exp(-kick_time * 38.0)
    kick = np.sin(2.0 * np.pi * np.cumsum(kick_freq) / AUDIO_RATE) * np.exp(-kick_time * 6.4)
    kick += rng.randn(kick_time.size) * np.exp(-kick_time * 240.0) * 0.20
    kick = (kick * 0.95).astype(np.float32)

    heart_time = timeline(0.42)
    heart = np.sin(2.0 * np.pi * 54.0 * heart_time) * np.exp(-heart_time * 9.5)
    heart += np.sin(2.0 * np.pi * 108.0 * heart_time) * np.exp(-heart_time * 14.0) * 0.28
    heart = (heart * 0.85).astype(np.float32)

    snare_time = timeline(0.26)
    snare_noise = np.diff(rng.randn(snare_time.size), prepend=0.0)
    snare = (snare_noise * 0.46 + np.sin(2.0 * np.pi * 188.0 * snare_time) * 0.20) * np.exp(-snare_time * 22.0)
    snare = snare.astype(np.float32)

    hat_time = timeline(0.085)
    hat = (np.diff(rng.randn(hat_time.size), prepend=0.0) * np.exp(-hat_time * 120.0) * 0.34).astype(np.float32)

    open_time = timeline(0.30)
    open_hat = (np.diff(rng.randn(open_time.size), prepend=0.0) * np.exp(-open_time * 24.0) * 0.26).astype(np.float32)

    gun_time = timeline(0.55)
    gun_noise = rng.randn(gun_time.size)
    gun_noise = np.diff(gun_noise, prepend=0.0)
    gun = gun_noise * np.exp(-gun_time * 26.0)
    gun += np.sin(2.0 * np.pi * (900.0 - 700.0 * gun_time / 0.55) * gun_time) * np.exp(-gun_time * 30.0) * 0.35
    gun += np.sin(2.0 * np.pi * 62.0 * gun_time) * np.exp(-gun_time * 11.0) * 0.55
    gun = (gun * 0.9).astype(np.float32)
    gun_attack = max(1, int(0.005 * AUDIO_RATE))
    gun[:gun_attack] *= np.linspace(0.0, 1.0, gun_attack)

    impact_time = timeline(1.8)
    impact = np.sin(2.0 * np.pi * (58.0 * np.exp(-impact_time * 1.4) + 26.0) * impact_time) * np.exp(-impact_time * 2.2)
    impact += rng.randn(impact_time.size) * np.exp(-impact_time * 12.0) * 0.16
    impact = (impact * 0.95).astype(np.float32)
    impact_attack = max(1, int(0.006 * AUDIO_RATE))
    impact[:impact_attack] *= np.linspace(0.0, 1.0, impact_attack)

    rise_time = timeline(1.2)
    rise_noise = rng.randn(rise_time.size)
    rise_noise = np.diff(rise_noise, prepend=0.0)
    rise_env = (rise_time / 1.2) ** 2.4
    rise = (rise_noise * rise_env * 0.5).astype(np.float32)

    def note_frequency(midi):
        return 440.0 * (2.0 ** ((midi - 69) / 12.0))

    def bass_note(midi, duration):
        frequency = note_frequency(midi)
        local = timeline(duration)
        wave = np.zeros(local.size, dtype=np.float64)
        for harmonic in range(1, 7):
            wave += np.sin(2.0 * np.pi * frequency * harmonic * local) / (harmonic ** 1.2)
        envelope = np.minimum(1.0, local / 0.004) * np.exp(-local * 4.8)
        return (wave * envelope * 0.32).astype(np.float32)

    def pad_chord(midis, duration, gain=1.0):
        local = timeline(duration)
        wave = np.zeros(local.size, dtype=np.float64)
        for midi in midis:
            frequency = note_frequency(midi)
            for detune in (-0.35, 0.0, 0.35):
                wave += np.sin(2.0 * np.pi * (frequency + detune) * local + rng.rand() * 6.283)
        attack = np.minimum(1.0, local / 0.35)
        release = np.minimum(1.0, np.maximum(0.0, (duration - local) / 0.45))
        return (wave * attack * release * (0.052 * gain / max(1, len(midis)))).astype(np.float32)

    def pluck(midi, duration):
        frequency = note_frequency(midi)
        local = timeline(duration)
        wave = np.sin(2.0 * np.pi * frequency * local)
        wave += np.sin(2.0 * np.pi * frequency * 3.0 * local) * 0.22
        wave += np.sin(2.0 * np.pi * frequency * 5.0 * local) * 0.09
        envelope = np.exp(-local * 7.5) * np.minimum(1.0, local / 0.003)
        return (wave * envelope * 0.14).astype(np.float32)

    progression = [
        (33, [57, 60, 64, 69]),
        (29, [53, 57, 60, 65]),
        (36, [55, 60, 64, 67]),
        (31, [55, 59, 62, 67]),
        (33, [57, 60, 64, 69]),
        (31, [55, 59, 62, 67]),
    ]

    bars = int(math.ceil(total_seconds / (BEAT * 4)))
    for bar in range(bars):
        bar_start = bar * BEAT * 4
        if bar_start >= total_seconds:
            break
        root, chord = progression[bar % len(progression)]
        dawn = bar < 9
        sun = 9 <= bar < 21
        sweat = 21 <= bar < 33
        glory = bar >= 33
        pad_gain = 1.15 if dawn else (0.9 if sun else (1.0 if sweat else 1.3))
        add(bar_start, pad_chord(chord, min(BEAT * 4 + 0.3, total_seconds - bar_start + 0.3), pad_gain), 1.0, 0.0)

        if dawn:
            for beat_index in range(4):
                add(bar_start + beat_index * BEAT, heart, 0.62 if beat_index == 0 else 0.34, 0.0)
            if bar >= 6:
                for step in range(8):
                    add(bar_start + step * BEAT * 0.5, hat, 0.07, 0.2)
            if bar == 8:
                add(bar_start + BEAT * 2.0, rise, 0.55, 0.0)
        elif sun:
            for beat_index in range(4):
                add(bar_start + beat_index * BEAT, kick, 0.86, 0.0)
            for beat_index in (1, 3):
                add(bar_start + beat_index * BEAT, snare, 0.40, -0.1)
            for step in range(16):
                add(bar_start + step * BEAT * 0.25, hat, 0.08 if step % 2 else 0.14, 0.25)
            for step in range(16):
                note = root + 12 if step % 4 == 3 else root
                add(bar_start + step * BEAT * 0.25, bass_note(note, BEAT * 0.22), 0.26, 0.0)
            for step in range(8):
                add(bar_start + step * BEAT * 0.5, pluck(chord[step % len(chord)] + 12, 0.28), 0.9, -0.2)
            if bar == 20:
                add(bar_start + BEAT * 2.0, rise, 0.7, 0.0)
        elif sweat:
            if bar == 21:
                add(bar_start, impact, 1.0, 0.0)
                for beat_index in range(1, 4):
                    add(bar_start + beat_index * BEAT, kick, 0.8, 0.0)
            else:
                for beat_index in (0, 2, 3):
                    add(bar_start + beat_index * BEAT, kick, 0.92, 0.0)
                add(bar_start + BEAT * 2.5, kick, 0.55, 0.0)
            for beat_index in (1, 3):
                add(bar_start + beat_index * BEAT, snare, 0.46, 0.12)
            for step in range(16):
                add(bar_start + step * BEAT * 0.25, hat, 0.10 if step % 2 else 0.17, -0.2)
            for step in range(16):
                note = root + 12 if step % 8 in (3, 7) else (root + 7 if step % 4 == 2 else root)
                add(bar_start + step * BEAT * 0.25, bass_note(note, BEAT * 0.2), 0.24, 0.0)
            if bar == 32:
                add(bar_start + BEAT * 2.0, rise, 0.75, 0.0)
        else:
            if bar == 33:
                add(bar_start, impact, 1.05, 0.0)
            for beat_index in range(4):
                add(bar_start + beat_index * BEAT, kick, 0.82 if bar < 42 else 0.65, 0.0)
            for beat_index in (1, 3):
                add(bar_start + beat_index * BEAT, snare, 0.38, -0.12)
            for step in range(8):
                add(bar_start + step * BEAT * 0.5, open_hat, 0.16, 0.22)
            for step in range(8):
                add(bar_start + step * BEAT * 0.5, pluck(chord[step % len(chord)] + 12, 0.4), 1.0, 0.25)
            if bar == 43:
                add(bar_start, impact, 0.95, 0.0)

    gun_position = 36 * BEAT
    add(gun_position - 0.02, gun, 1.25, 0.0)

    drop_fade_start = int(41.64 * AUDIO_RATE)
    drop_silence_start = int(41.72 * AUDIO_RATE)
    drop_end = int(42.0 * AUDIO_RATE)
    if drop_end < sample_count:
        mix[drop_fade_start:drop_silence_start] *= np.linspace(1.0, 0.0, drop_silence_start - drop_fade_start)[:, None]
        mix[drop_silence_start:drop_end] = 0.0

    fade_in = int(0.5 * AUDIO_RATE)
    fade_out = int(min(3.2, total_seconds * 0.25) * AUDIO_RATE)
    if fade_in > 0:
        mix[:fade_in] *= np.linspace(0.0, 1.0, fade_in)[:, None]
    if fade_out > 0:
        mix[-fade_out:] *= np.linspace(1.0, 0.0, fade_out)[:, None]
    mix = np.tanh(mix * 1.15)
    peak = float(np.max(np.abs(mix))) if mix.size else 1.0
    if peak > 0:
        mix *= 0.50 / peak
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
            "-pix_fmt",
            "bgr24",
            "-f",
            "rawvideo",
            "-",
        ],
        capture_output=True,
    )
    data = result.stdout
    if len(data) < WIDTH * HEIGHT * 3:
        return None
    return np.frombuffer(data[: WIDTH * HEIGHT * 3], dtype=np.uint8).reshape(HEIGHT, WIDTH, 3)


def verify(output_path, shots, total_seconds):
    info = subprocess.run([FFMPEG, "-hide_banner", "-i", output_path], capture_output=True, text=True, errors="replace").stderr
    duration = 0.0
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", info)
    if match:
        duration = int(match.group(1)) * 3600 + int(match.group(2)) * 60 + float(match.group(3))
    size = os.path.getsize(output_path)
    log("file           : " + output_path)
    log("size           : %.2f MB" % (size / 1048576.0))
    log("duration       : %.2f s" % duration)
    log("streams        : video %d | audio %d" % (len(re.findall(r"Stream\s+\x23\d+:\d+.*: Video:", info)), len(re.findall(r"Stream\s+\x23\d+:\d+.*: Audio:", info))))
    decode = subprocess.run([FFMPEG, "-hide_banner", "-y", "-i", output_path, "-f", "null", "-"], capture_output=True, text=True, errors="replace")
    problems = [line for line in decode.stderr.splitlines() if re.search(r"\b(error|corrupt|invalid)\b", line, re.IGNORECASE)]
    frames = [int(value) for value in re.findall(r"frame=\s*(\d+)", decode.stderr)]
    decoded = max(frames) if frames else 0
    expected = int(round(duration * FPS))
    log("frames decoded : %d (expected %d) | decode issues %d" % (decoded, expected, len(problems)))

    letterbox_ok = False
    grade_ok = False
    grain_ok = False
    sample = grab_output(output_path, duration * 0.45)
    if sample is not None:
        top = sample[:BAR_H].astype(np.float32)
        bottom = sample[HEIGHT - BAR_H :].astype(np.float32)
        content = sample[BAR_H : HEIGHT - BAR_H].astype(np.float32)
        log("letterbox      : bar means %.2f / %.2f | ratio %.3f:1" % (float(top.mean()), float(bottom.mean()), WIDTH / float(CONTENT_H)))
        letterbox_ok = float(top.mean()) <= 2.0 and float(bottom.mean()) <= 2.0 and abs(WIDTH / float(CONTENT_H) - RATIO) < 0.02
        gray = content.mean(axis=2)
        low = content[gray < np.percentile(gray, 25)]
        high = content[gray > np.percentile(gray, 75)]
        shadow = low.mean(axis=0)
        light = high.mean(axis=0)
        log("grade          : shadows B%.0f G%.0f R%.0f | highlights B%.0f G%.0f R%.0f" % (shadow[0], shadow[1], shadow[2], light[0], light[1], light[2]))
        grade_ok = shadow[0] > shadow[2] and light[2] > light[0]
        flat = cv2.cvtColor(sample[BAR_H : HEIGHT - BAR_H], cv2.COLOR_BGR2GRAY)
        high_freq = float(np.abs(np.diff(flat.astype(np.float32), axis=1)).mean())
        log("grain/detail   : mean horizontal gradient %.3f" % high_freq)
        grain_ok = high_freq > 3.0

    offsets = []
    for shot in shots[1:]:
        offsets.append(abs(shot["start"] / BEAT - round(shot["start"] / BEAT)))
    grid_ok = max(offsets) < 1e-6 if offsets else False
    log("beat grid      : %d cuts | max off-grid %.6f beats" % (len(shots) - 1, max(offsets) if offsets else 0.0))

    audio = subprocess.run([FFMPEG, "-hide_banner", "-i", output_path, "-af", "volumedetect", "-f", "null", "-"], capture_output=True, text=True, errors="replace").stderr
    level = {}
    for line in audio.splitlines():
        match = re.search(r"(mean|max)_volume:\s*(-?[\d.]+)", line)
        if match:
            level[match.group(1)] = float(match.group(2))
    log("audio level    : mean %.1f dB | peak %.1f dB" % (level.get("mean", 0.0), level.get("max", 0.0)))

    pcm = subprocess.run(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-i", output_path, "-vn", "-ac", "1", "-ar", str(AUDIO_RATE), "-f", "f32le", "-acodec", "pcm_f32le", "-"],
        capture_output=True,
    ).stdout
    decoded_audio = np.frombuffer(pcm, dtype=np.float32) if pcm else np.zeros(1, dtype=np.float32)

    def window_peak(start_seconds, span_seconds):
        first = int(start_seconds * AUDIO_RATE)
        last = int((start_seconds + span_seconds) * AUDIO_RATE)
        segment = decoded_audio[max(0, first) : min(len(decoded_audio), last)]
        return float(np.abs(segment).max()) if segment.size else 0.0

    log(
        "sound design   : gun %.3f | pre-drop %.4f | impact %.3f"
        % (window_peak(18.0, 0.15), window_peak(41.78, 0.2), window_peak(42.02, 0.3))
    )

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
    grid_offsets = [abs(k - round(k / BEAT) * BEAT) for k in kicks]
    hit = 0
    cut_count = 0
    for shot in shots[1:]:
        if shot["start"] >= duration - 0.2 or not kicks:
            continue
        cut_count += 1
        if min(abs(k - shot["start"]) for k in kicks) <= 0.05:
            hit += 1
    sync = 0.0 if cut_count == 0 else hit / float(cut_count)
    log("kick onsets    : %d | grid median %.3f s" % (len(kicks), float(np.median(grid_offsets)) if grid_offsets else -1.0))
    log("cut sync       : %d/%d cuts on a kick (%.0f%%)" % (hit, cut_count, 100.0 * sync))

    checks = {
        "duration": abs(duration - total_seconds) <= 1.0,
        "resolution": bool(re.search(r"1920x1080", info)),
        "streams": "Video:" in info and "Audio:" in info,
        "decode": len(problems) == 0 and abs(decoded - expected) <= max(3, expected * 0.01),
        "letterbox": letterbox_ok,
        "grade": grade_ok,
        "grain": grain_ok,
        "beat_grid": grid_ok,
        "cut_sync": sync >= 0.8,
        "audio": level.get("max", -99.0) < -0.5 and level.get("mean", -99.0) > -40.0,
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
    log("tempo          : %d BPM | beat %.2f s | %d beats = %.1f s" % (BPM, BEAT, TOTAL_BEATS, TOTAL_BEATS * BEAT))
    log("frame          : %dx%d | visible %dx%d (%.3f:1) | bars %d px" % (WIDTH, HEIGHT, WIDTH, CONTENT_H, WIDTH / float(CONTENT_H), BAR_H))
    if os.path.isdir(WORK_DIR):
        shutil.rmtree(WORK_DIR)
    os.makedirs(WORK_DIR)
    stage_done("toolchain ready")

    stage("STAGE 2/7  MATERIAL ANALYSIS")
    image_paths = list_sources(".jpg")
    video_paths = list_sources(".mp4")
    log("source images  : %d" % len(image_paths))
    log("source videos  : %d" % len(video_paths))
    images, stats = select_images(image_paths)
    log("dedupe         : %d analysed -> %d clusters -> %d selected" % (stats["analysed"], stats["clusters"], stats["selected"]))
    if not images:
        raise RuntimeError("no usable images")
    videos = select_videos(video_paths)
    log("video highlights: %d" % len(videos))
    stage_done("%d stills + %d motion clips" % (len(images), len(videos)))

    stage("STAGE 3/7  NARRATIVE TIMELINE")
    shots, beat_total = build_timeline(images, videos)
    if beat_total != TOTAL_BEATS:
        raise RuntimeError("timeline beats mismatch: %d" % beat_total)
    total_seconds = beat_total * BEAT
    from collections import Counter

    kinds = Counter(shot["kind"] for shot in shots)
    transitions = Counter(shot["transition"] for shot in shots)
    speeds = Counter(shot["speed"]["label"] for shot in shots if shot["kind"] == "video")
    log("shots          : %d (image %d / video %d)" % (len(shots), kinds["image"], kinds["video"]))
    log("movements      : " + " | ".join("%s %s %ds" % (m["roman"], m["name"], len(m["beats"]) and sum(m["beats"]) // 2) for m in MOVEMENTS))
    log("transitions    : " + " ".join("%s=%d" % (k, v) for k, v in sorted(transitions.items())))
    log("speed profiles : " + " ".join("%s=%d" % (k, v) for k, v in sorted(speeds.items())))
    log("text cards     : %d" % len(CARDS))
    for key in ("school_en", "school_cn", "class_en", "class_cn", "motto"):
        card = next(item for item in CARDS if item["key"] == key)
        log("  card %-9s %s" % (key, card["text"]))
    stage_done("shot list locked to beat grid")

    stage("STAGE 4/7  SCORE SYNTHESIS")
    score = build_score(total_seconds)
    log("score          : %.2f s stereo @ %d Hz" % (score.shape[0] / float(AUDIO_RATE), AUDIO_RATE))
    log("score peak/rms : %.3f / %.3f" % (float(np.abs(score).max()), float(np.sqrt((score ** 2).mean()))))
    stage_done("120 BPM epic electronic orchestral bed")

    stage("STAGE 5/7  RENDER")
    render_seconds = total_seconds
    if LIMIT_SECONDS > 0:
        render_seconds = min(LIMIT_SECONDS, total_seconds)
        log("debug limit    : rendering first %.2f s of %.2f s" % (render_seconds, total_seconds))
    clip = build_clip(shots, total_seconds)
    if render_seconds < total_seconds:
        clip = clip.with_duration(render_seconds)
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_NAME)
    soundtrack = score_clip(score)
    if render_seconds < total_seconds:
        soundtrack = soundtrack.with_duration(render_seconds)
    sequence = clip.with_audio(soundtrack)
    sequence.write_videofile(
        output_path,
        fps=FPS,
        codec="libx264",
        bitrate=BITRATE,
        audio_codec="aac",
        audio_bitrate="256k",
        preset="medium",
        threads=4,
        pixel_format="yuv420p",
        temp_audiofile=os.path.join(WORK_DIR, "score.m4a"),
        temp_audiofile_path=WORK_DIR,
        ffmpeg_params=["-profile:v", "high", "-level", "4.2", "-movflags", "+faststart", "-g", "30", "-bf", "2"],
    )
    sequence.close()
    stage_done("master encoded")

    stage("STAGE 6/7  VERIFICATION")
    checks, size, duration = verify(output_path, shots, render_seconds)
    stage_done("verified")

    stage("STAGE 7/7  CLEANUP")
    shutil.rmtree(WORK_DIR, ignore_errors=True)
    log("cleanup        : render workspace removed")
    passed = all(checks.values())
    print("", flush=True)
    log("RESULT         : " + ("PASS" if passed else "FAIL"))
    log("OUTPUT         : %s | %.2f MB | %.2f s | 1920x1080 | 30 fps | h264 %s | aac" % (OUTPUT_NAME, size / 1048576.0, duration, BITRATE))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
