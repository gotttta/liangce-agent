# DRAM Defect Metrology Agent

MVP runner for the DRAM defect metrology Agent described in `docs/design.md`.

The current version is a lightweight baseline:

- reads an unannotated target image
- segments bright/dark anomaly regions
- measures connected components for metrology
- writes the five planned output files

It does not connect to an LLM, SAM, or Gradio yet.

## Run

```bash
python3 main.py \
  --target path/to/target.png \
  --description "bright particle defect, measure area and count"
```

Optional reference image:

```bash
python3 main.py \
  --reference path/to/reference.png \
  --target path/to/target.png \
  --description "bridge defect, measure area and gap"
```

## Web UI

```bash
python3 -m ui.app
```

Open `http://127.0.0.1:7860`.

## Outputs

Each run creates:

```text
outputs/<timestamp>_<defect_type>/
├── algorithm.py
├── strategy.json
├── result_annotated.png
├── mask.png
└── measurements.json
```
