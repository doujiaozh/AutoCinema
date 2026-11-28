import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import timeline_fixed_cinema as E

OUT_DIR = r"D:\Code\Video\debug_frames"
PROBES = (10.0, 40.0)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for name in os.listdir(OUT_DIR):
        os.remove(os.path.join(OUT_DIR, name))

    media = E.collect_media()
    E.annotate_media(media)
    kept, dropped = E.dedupe_consecutive(media)
    groups = E.split_chapters(kept, len(E.CHAPTERS))
    shots, beats = E.build_timeline(groups)
    shots = E.apply_mask_plan(shots)
    total = beats * E.BEAT
    print("media %d -> kept %d -> shots %d -> %.1f s" % (len(media), len(kept), len(shots), total))

    clip, _ = E.build_clip(shots, total)
    saved = []
    for moment in PROBES:
        frame = clip.get_frame(moment)
        path = os.path.join(OUT_DIR, "frame_%05.1fs.png" % moment)
        Image.fromarray(frame).save(path)
        band = frame[E.BAR_H : E.BAR_H + E.CONTENT_H]
        gray = band.mean(axis=2)
        dark = float((gray < 6.0).mean())
        print("  t=%5.1f s -> %s" % (moment, path))
        print("     band mean %.1f std %.1f dark %.4f" % (float(gray.mean()), float(gray.std()), dark))
        for shot in shots:
            if shot["start"] <= moment < shot["start"] + shot["duration"]:
                source = E.load_image(shot["path"]) if shot["kind"] == "image" else None
                size = "%dx%d" % (source.shape[1], source.shape[0]) if source is not None else "video"
                progress = (moment - shot["start"]) / shot["duration"]
                zoom = shot["motion"][0] + (shot["motion"][1] - shot["motion"][0]) * progress
                print(
                    "     shot %d %s %s | chapter %s | source %s | zoom %.3f | beats %d"
                    % (shot["index"], shot["kind"], os.path.basename(shot["path"])[:22], E.CHAPTERS[shot["chapter"]]["name"], size, min(max(zoom, 1.0), 1.10), shot["beats"])
                )
                break
        saved.append(path)

    print("")
    print("saved %d debug frames to %s" % (len(saved), OUT_DIR))
    return 0


if __name__ == "__main__":
    sys.exit(main())
