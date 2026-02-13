# Option A: Automated flat-portrait conversion

This repository now includes a practical **Option A** batch pipeline that converts portrait photos into a flat, simplified illustration style similar to your sketch.

## What it does

- Foreground extraction (person separation from background)
- Background replacement with a flat color
- Color reduction to a small palette
- Skin/hair tone normalization for cleaner cartoon output

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install opencv-python numpy
```

## Usage

Single file:

```bash
python option_a_pipeline.py input.jpg output.png --colors 6 --background "#e6e6e6"
```

Folder batch:

```bash
python option_a_pipeline.py ./uploads ./outputs --colors 6 --background "230,230,230"
```

## Notes

- Best results are portrait photos where subjects are centered.
- For tighter style matching, adjust `--colors` (try 5-8).
- If you want even cleaner silhouettes, run this output through vectorization (`potrace`/Illustrator image trace).
