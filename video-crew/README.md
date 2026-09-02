# video-crew

A crew of agents that summarises a video, coordinated through a task board
on disk. Covers video understanding (scene cuts, keyframes, tracking, frame
captioning), a multi-agent system with an independent verifier, image
generation for a poster, and an eval against synthetic ground truth.

## Setup

```sh
cd video-crew
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt pytest
pip install torch diffusers transformers accelerate   # optional, for the diffusion poster
```

## Use

```sh
python make_video.py --out data/clip1 --scenes 4 --seed 1   # synthetic clip + truth.json
python summarise.py data/clip1/video.mp4 --out runs/clip1   # classical crew, mosaic poster
python summarise.py data/clip1/video.mp4 --mode claude      # Claude vision captions (needs ANTHROPIC_API_KEY)
python summarise.py data/clip1/video.mp4 --diffusion        # Stable Diffusion poster (sd-turbo, ~2.5 GB download)
python evaluate.py --clips 10                                # score against ground truth
pytest
```

## How it works

**Video understanding** (`vision/`)

- `scenes.py`: a cut is a frame whose HSV histogram is far from the previous
  one (Bhattacharyya distance). Moving objects shift it a little; a
  background change shifts it a lot. Keyframe is the middle of each scene.
- `motion.py`: every pixel is assigned its nearest named colour, then blobs
  are found per colour. Shape comes from area ratios: how much of the
  bounding box and of the enclosing circle the blob fills. Motion is the
  centroid displacement between first and last frame of the scene.
- `describe.py`: two describers with one caption schema. `classical` uses
  the above. `ClaudeDescriber` sends first, middle and last frame to Claude
  vision and asks for the same JSON. Either can be swapped in without the
  crew noticing.

**The crew** (`crew/`)

- `board.py`: one JSON file per task. Claiming is an exclusive lock file
  (`O_EXCL`), safe across threads and processes. Tasks declare dependencies.
  There is no manager and no messaging. Every agent reads the whole board.
- `roles.py`:
  - **splitter** detects scenes and posts one describe task and one verify
    task per scene, plus a compose task that depends on all verifies.
  - **describers** run in parallel, claim tasks, write captions.
  - **verifier** never trusts the caption. It flattens it into atomic claims
    ("there is a red circle", "the red circle moves left") and re-derives
    each one from the frames. Rejected claims are recorded on the board.
  - **composer** assembles the summary from captions minus rejected claims.
- `run.py` runs the four stages and writes `report.json` with timings and
  the board state.

**Poster** (`vision/poster.py`): `diffusion_poster` runs sd-turbo through
diffusers, prompted from the verified summary. It uses Apple MPS when
available. `mosaic_poster` is the fallback: keyframe grid plus title.

## Eval

`evaluate.py` generates clips with known scenes, objects, colours and
motions, runs the crew, and scores it. It then plants one wrong fact per
scene caption and checks whether the verifier rejects it.

Results on 10 synthetic clips:

| metric | result |
|---|---|
| scene count correct | 10 / 10 |
| object recall | 100% |
| motion accuracy | 100% |
| background accuracy | 100% |
| planted wrong captions caught | 37 / 37 |

Perfect scores mean the synthetic set is now too easy for the classical
path. That is the point where real footage should replace it.

## Things learned building it

- The first detector merged overlapping objects into one blob. Segmenting by
  colour before finding blobs fixed it.
- Colour distance in int16 overflowed silently and labelled every pixel
  wrong. Always check the dtype before squaring.
- Vertex counting is unreliable for small anti-aliased shapes. Area ratios
  against the bounding box and enclosing circle are much more stable.
- The generator had a placement bug that started leftward-moving objects off
  screen. The eval caught it, which is what the eval is for.
- Codec chroma blur makes black on navy and yellow on cream unreadable. The
  generator avoids those pairs, and a real system needs to know the same.
