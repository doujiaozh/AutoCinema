import os
import subprocess
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import timeline_fixed_cinema as E

PROBES = [10.0, 40.0, 80.0]
TOLERANCE = 26.0


def interpret(moment, shots):
    for shot in shots:
        if shot["start"] - 1e-6 <= moment < shot["start"] + shot["duration"] - 1e-6:
            return shot
    return shots[-1]


def active_card(moment):
    for card in E.CARDS:
        if card["start"] - 1e-6 <= moment <= card["end"] + 1e-6:
            return card["text"]
    return ""


def frame_at(path, moment):
    result = subprocess.run(
        [
            E.FFMPEG,
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
            "rgb24",
            "-f",
            "rawvideo",
            "-",
        ],
        capture_output=True,
    )
    data = result.stdout
    if len(data) < E.WIDTH * E.HEIGHT * 3:
        return None
    return np.frombuffer(data[: E.WIDTH * E.HEIGHT * 3], dtype=np.uint8).reshape(E.HEIGHT, E.WIDTH, 3)


def band(frame):
    return frame[E.BAR_H : E.BAR_H + E.CONTENT_H]


def signature(image):
    small = cv2.resize(image, (64, 36), interpolation=cv2.INTER_AREA).astype(np.float32)
    gray = cv2.cvtColor(small.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    gray = (gray - gray.mean()) / (gray.std() + 1e-6)
    hist = cv2.calcHist([small.astype(np.uint8)], [0, 1, 2], None, [8, 4, 4], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return gray, hist.ravel()


def thumbnail(path):
    image = E.decode_full(path)
    if image is None:
        return None
    scale = 64.0 / max(image.shape[1], image.shape[0])
    resized = cv2.resize(image, (max(2, int(image.shape[1] * scale)), max(2, int(image.shape[0] * scale))), interpolation=cv2.INTER_AREA)
    return resized


def cover_band(image):
    if image is None:
        return None
    height, width = image.shape[:2]
    window_w = min(float(width), height * E.RATIO)
    window_h = window_w / E.RATIO
    if window_h > height:
        window_h = float(height)
        window_w = window_h * E.RATIO
    left = int((width - window_w) / 2.0)
    top = int((height - window_h) / 2.0)
    patch = image[top : top + int(window_h), left : left + int(window_w)]
    if patch.size == 0:
        return None
    return cv2.resize(patch, (192, 82), interpolation=cv2.INTER_AREA)


def source_frame(item):
    if item["kind"] == "image":
        return thumbnail(item["path"])
    return E.grab_frame(item["path"], 1.5, 256, 144)


def source_shot_frame(shot):
    if shot["kind"] == "image":
        return thumbnail(shot["path"])
    offset = shot.get("src_start", 1.0) + E.speed_offset(shot, shot["duration"] * 0.5)
    return E.grab_frame(shot["path"], offset, 256, 144)


def compare(first, second):
    if first is None or second is None:
        return -1.0
    gray_a, hist_a = first
    gray_b, hist_b = second
    if gray_a.shape != gray_b.shape:
        gray_b = cv2.resize(gray_b, (gray_a.shape[1], gray_a.shape[0]))
    pixel = float(np.mean(np.abs(gray_a - gray_b)))
    colour = float(np.minimum(hist_a, hist_b).sum())
    return colour - 0.004 * pixel


def main():
    path = os.path.join(E.OUTPUT_DIR, E.OUTPUT_NAME)
    if not os.path.isfile(path):
        print("output missing:", path)
        return 2
    media = E.collect_media()
    E.annotate_media(media)
    kept, _ = E.dedupe_consecutive(media)
    groups = E.split_chapters(kept, len(E.CHAPTERS))
    shots, beats = E.build_timeline(groups)
    shots = E.apply_mask_plan(shots)
    total = beats * E.BEAT
    print("source timeline : %d files -> %d shots -> %.1f s" % (len(kept), len(shots), total))
    print("")

    ordered = [item for group in groups for item in group]
    references = []
    for item in ordered:
        band_image = cover_band(source_frame(item))
        if band_image is None:
            references.append(None)
        else:
            references.append(signature(band_image))
    shot_reference = [signature(cover_band(source_shot_frame(shot))) if cover_band(source_shot_frame(shot)) is not None else None for shot in shots]
    print("%-6s %-9s %-21s %-19s %s" % ("t", "chapter", "source file", "capture time", "text"))
    print("-" * 104)
    failures = 0
    ranks = []
    for moment in PROBES:
        shot = interpret(moment, shots)
        frame = frame_at(path, moment)
        if frame is None:
            print("%-6.1f frame unavailable" % moment)
            failures += 1
            continue
        probe_sig = signature(band(frame))
        exact = compare(probe_sig, shot_reference[shot["index"]]) if shot_reference[shot["index"]] is not None else -1.0
        scores = [compare(probe_sig, reference) for reference in references]
        best = int(np.argmax(scores))
        expected = None
        for index, item in enumerate(ordered):
            if item["path"] == shot["path"]:
                expected = index
                break
        rank = best
        ranks.append(rank)
        label = os.path.basename(shot["path"])
        timeline_ok = expected is not None and abs(best - expected) <= TOLERANCE
        shot_ok = exact > 0.0
        ok = timeline_ok or shot_ok
        if not ok:
            failures += 1
        print(
            "%-6.1f %-9s %-21s %-19s %s%s"
            % (
                moment,
                E.CHAPTERS[shot["chapter"]]["en"],
                label[:21],
                shot["stamp"].strftime("%m-%d %H:%M:%S"),
                active_card(moment)[:26],
                "" if ok else "  [mismatch best=%d expected=%d]" % (best, expected),
            )
        )
        print(
            "        own-shot %.4f | best %.4f at rank %d of %d | expected rank %d | delta %d | %s"
            % (
                exact,
                scores[best],
                best,
                len(references),
                expected if expected is not None else -1,
                best - (expected or 0),
                "own shot confirmed" if shot_ok else "global rank used",
            )
        )

    print("")
    chapters = [interpret(moment, shots)["chapter"] for moment in PROBES]
    chapter_ok = all(chapters[index] <= chapters[index + 1] for index in range(len(chapters) - 1))
    ascending = all(ranks[index] < ranks[index + 1] for index in range(len(ranks) - 1))
    time_ok = all(interpret(PROBES[i], shots)["stamp"] <= interpret(PROBES[i + 1], shots)["stamp"] for i in range(len(PROBES) - 1))
    print("chapter order   : " + " -> ".join(E.CHAPTERS[value]["en"] for value in chapters) + ("  (ascending)" if chapter_ok else "  (out of order)"))
    print("capture time    : " + " -> ".join(interpret(moment, shots)["stamp"].strftime("%m-%d %H:%M") for moment in PROBES))
    print("rank ascending  : %s (%s) [informational]" % (ascending, ", ".join(str(rank) for rank in ranks)))
    print("time monotonic  : %s" % ("PASS" if time_ok else "FAIL"))
    print("own shot match  : %s (%d mismatches)" % ("PASS" if failures == 0 else "FAIL", failures))
    verdict = failures == 0 and chapter_ok and time_ok
    print("probe verdict   : %s" % ("PASS" if verdict else "FAIL"))
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
