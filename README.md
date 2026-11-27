# AutoCinema

Procedural film engines. Each one turns a folder of raw photos and clips into a finished 90 second cinematic short, using nothing but Python, NumPy, OpenCV and moviepy.

| Engine | Output | Focus |
| --- | --- | --- |
| `auto_cinema.py` | `Class20_Sports_Cinema.mp4` | Beat-locked montage, typography, teal and orange grade |
| `epic_cinema.py` | `Epic_Cinematic_Master.mp4` | Glitch art, pseudo 3D parallax, kinetic type, epic score |

## Target film (epic engine)

| Field | Value |
| --- | --- |
| Subject | Haikou Jinpan Experimental School sports day, class 7(20) |
| Duration | 90.00 s at 30 fps, 120 BPM, 180 beats |
| Format | 1920x1080, H.264 12000k, AAC 256k, 2.35:1 letterbox |

## Structure

| Act | Window | Beats | Function |
| --- | --- | --- | --- |
| Prologue | 0.00 - 3.00 s | 6 | Absolute silence, sub heartbeat, title rises out of black |
| I Dawn | 3.00 - 18.00 s | 30 | Drop, first parallax reveal, slow build |
| II Sun | 18.00 - 42.00 s | 48 | Sprint montage, glitch accents, starting gun |
| III Limit | 42.00 - 66.00 s | 48 | Impact, extreme speed ramps, heavy glitch, text masks |
| IV Glory | 66.00 - 90.00 s | 48 | Choir, class reveal, resolve |

## Visual systems

- Glitch: RGB plane separation, band displacement, scanline noise, gated to the beat envelope.
- Lens warp: four precomputed radial remap banks, fired as impacts.
- Pseudo 3D: each hero still is split into subject and backdrop by a saliency and center prior, then the two planes travel at different speeds.
- Kinetic type: perspective keystone cards that slam out of depth, with black stroke and additive bloom.
- Text masking: giant glyphs cut the picture during four transitions.
- Grade, bloom, film grain, vignette, light leaks and directional motion blur run on every frame.

## Audio systems

NumPy additive synthesis at 120 BPM: heartbeat, kick, snare, hats, harmonic bass, plucked arpeggio, detuned pad, formant choir, brown noise wind, risers, impacts, a three seconde silence gate and a drop at 3.00 s. Every event is quantised to 0.125 s so all 54 cuts land on a kick.

## Requirements

- Windows with Python 3.11
- numpy, opencv-python, pillow, scipy, moviepy, imageio-ffmpeg
- Microsoft YaHei Bold, Impact and Arial Bold system fonts
- Source media in `D:\Code\Video`

## Run

```
python auto_cinema.py
python epic_cinema.py
```

Set `AUTOCINEMA_LIMIT` or `EPIC_LIMIT` to a number of seconds to render only the head of a film for a fast check.

## Verification

Both engines finish with a check matrix covering duration, resolution, streams, decode sweep, letterbox geometry, grade sign, effect unit tests, beat grid alignment, cut to kick correlation and audio headroom, then print `RESULT: PASS` or `RESULT: FAIL`.
