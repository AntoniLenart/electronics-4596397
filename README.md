# SDN Telemetry + Snort IDS: Machine-Learning DDoS Detection

Processed datasets and the ML evaluation suite accompanying the study on
telemetry-only DDoS detection for 5G/6G mobile-edge networks (MDPI *Electronics*,
submission 4596397).

## Layout

```
snort/   Snort rules used on the testbed
ml/      Dataset generation, campaign combiner, evaluation suite
data/    Per-campaign and pooled datasets + results
```

## Usage

```bash
python -m venv .venv-ml && .venv-ml/Scripts/activate
pip install -r requirements.txt
python ml/run_all.py
```

## License

Code: MIT (see `LICENSE`). Datasets: CC BY 4.0.
