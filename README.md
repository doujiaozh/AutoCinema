# AutoCinema

Procedural film engines. Each one turns a folder of raw photos and clips into a finished cinematic short using nothing but Python, NumPy, OpenCV and moviepy.

| Engine | Output | Focus |
| --- | --- | --- |
| `auto_cinema.py` | `Class20_Sports_Cinema.mp4` | Beat-locked montage, typography, teal and orange grade |
| `epic_cinema.py` | `Epic_Cinematic_Master.mp4` | Glitch art, pseudo 3D parallax, kinetic type, epic score |
| `timeline_fixed_cinema.py` | `Timeline_Fixed_Cinema.mp4` | Chronological scheduling, chapter bound typography |

`check_timeline.py` validates a rendered film against the chronological source order.

## Chronological scheduling

The timeline engine replaces quality based global reordering with a strict capture time sort.

1. Ingest. Every image and clip is timestamped from the `YYYYMMDDHHMMSS` token inside the filename. Clips without a token fall back to the container `creation_time` shifted to local time, then to the file modification time. Everything is sorted ascending once.
2. Dedupe. Only consecutive items within a 150 second window and within a 12 bit perceptual hash distance are collapsed. The global order is never touched.
3. Chapters. The sorted list is cut into four equal counts mapped to PROLOGUE, ARENA, CLIMAX and CODA, so capture time rises monotonically across the whole film.
4. Selection. Each chapter picks its shots with an even stride through its own slice, which preserves order inside every chapter.

The render aborts if any shot's capture time is not greater than or equal to the previous one.

## Chapter mapping

| Chapter | Window | Shots | Camera | Typography |
| --- | --- | --- | --- | --- |
| PROLOGUE | 0 - 12 s | 4 | parallax | 海口市金盘实验学校 |
| ARENA | 12 - 38 s | 16 | parallax | 七(20)班 |
| CLIMAX | 38 - 68 s | 20 | rush, speed ramps, motion blur | GRIT at 24 s, SWEAT at 30 s, NEVER GIVE UP at 46 s, GLORY at 56 s |
| CODA | 68 - 90 s | 11 | parallax | 荣光 closing cards |

Speed ramping and motion blur are confined to CLIMAX. The early chapters keep the gentle two plane parallax push.

## Verification

```
python timeline_fixed_cinema.py
python check_timeline.py
```

`check_timeline.py` decodes frames at 10, 40 and 80 seconds, matches each one against every source frame by colour histogram plus luminance correlation, and asserts three things: the reported chapter advances PROLOGUE to ARENA to CODA, the best match rank increases with playback time, and the capture timestamps are monotonic.

## Requirements

- Windows with Python 3.11
- numpy, opencv-python, pillow, scipy, moviepy, imageio-ffmpeg
- Microsoft YaHei Bold, Impact and Arial Bold system fonts
- Source media in `D:\Code\Video`

Set `EPIC_LIMIT` to a number of seconds to render only the head of a film for a fast check.
