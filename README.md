# AutoCinema

Procedural sports-film engine. A single Python program turns a folder of raw photos and clips into a 90 second, 2.35:1, beat-locked cinematic short with typography, colour grading, film grain, speed ramping and a fully synthesised score.

## Target film

| Field | Value |
| --- | --- |
| Title | Class20_Sports_Cinema |
| Subject | Haikou Jinpan Experimental School sports day, class 7(20) |
| Duration | 90.00 s at 30 fps, 120 BPM |
| Format | 1920x1080, H.264 12000k, AAC 256k |
| Aspect | 2.35:1 with permanent 132 px black bars |

## Narrative

| Movement | Window | Beats | Function |
| --- | --- | --- | --- |
| I DAWN | 0.0 - 18.0 s | 36 | Hold, gather, low pulse, rising tension |
| II SUN | 18.0 - 42.0 s | 48 | Starting gun, first sprint montage |
| III LIMIT | 42.0 - 66.0 s | 48 | Drop, extreme speed ramping, compression |
| IV GLORY | 66.0 - 90.0 s | 48 | Impacts, anthem, class reveal, resolve |

## Pipeline

1. Material scan. Perceptual hash, sharpness, contrast, brightness and warmth for every still. Motion profile for every clip.
2. Curation. Complete linkage clustering on Hamming distance, quality ranking, perceptual diversity floor, then movement assignment.
3. Beat map. Every shot length is an integer number of beats, so all 55 cuts land exactly on the 120 BPM grid.
4. Frame engine. Cover crop to 2.35:1, eight motion primitives, Linear speed ramp for motion clips, teal shadow and warm highlight LUT, saturation lift, eight film grain plates, radial vignette, white flash and punch transitions.
5. Typography. Bold sans cards rendered with PIL, black stroke, breathing scale, per-card alpha envelope, composited before the letterbox.
6. Score. NumPy additive synthesis: heartbeat pulse, four on the floor kick, snare, sixteenth hats, harmonic bass, plucked arpeggio, detuned pad, starting gun, riser, drop, impact. All events quantised to 0.125 s.
7. Verification. Decode sweep, letterbox measurement, shadow and highlight sampling, grain gradient, beat grid alignment, kick onset detection and cut to kick correlation.

## Requirements

- Windows with Python 3.11
- numpy, opencv-python, pillow, scipy, moviepy, imageio-ffmpeg
- Microsoft YaHei Bold, Impact and Arial Bold system fonts
- Source media in `D:\Code\Video`

## Run

```
python auto_cinema.py
```

Set `AUTOCINEMA_LIMIT` to a number of seconds to render only the head of the film for a fast check.

## Output contract

The run prints a stage log, a per-check verdict line and a final `RESULT: PASS` or `RESULT: FAIL`. The master is written to `D:\Code\Video\Class20_Sports_Cinema.mp4`.
