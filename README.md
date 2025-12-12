# Stereo Chorus of Whispers

Winner solution of the Cadenza Lyric Intelligibility Prediction (CLIP) Challenge at ICASSP 2026 — see [Official Results (T045)](https://cadenzachallenge.org/docs/clip1/results)

---

## Setup

```bash
conda create -n cadenza python=3.10
conda activate cadenza

git clone https://github.com/jinlongbin/Stereo-Chorus-of-Whispers.git
cd Stereo-Chorus-of-Whispers

pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
# PESQ (Windows): install via conda-forge
conda install -c conda-forge pesq
```
---

## Quick Start (inference only)
Run inference for validation and evaluation:
```bash
python infer.py
```
Outputs: `checkpoints/final/inference.csv`, `checkpoints/final/eval_inference.csv`.

---

## 1. Data prep (JSON -> CSV)
Download the data from Zenodo: https://zenodo.org/records/17789818 (train/valid/eval packages). After extracting into `data/`, replace the metadata JSONs with those from `cadenza_clip1_data.metadata.v1.2.tar.gz`. Final structure:
```
data/
  cadenza_data_train/
    metadata/train_metadata.json  # from cadenza_clip1_data.metadata.v1.2.tar.gz
    train/
      signals/*.flac
      unprocessed/*.flac
  cadenza_data_valid/
    metadata/valid_metadata.json  # from cadenza_clip1_data.metadata.v1.2.tar.gz
    valid/
      signals/*.flac
      unprocessed/*.flac
  cadenza_data_eval/
    metadata/eval_metadata.json  # from cadenza_clip1_data.metadata.v1.2.tar.gz
    eval/
      signals/*.flac
      unprocessed/*.flac
```

The CSVs are already generated under `data/cadenza_data_*/metadata/*.csv`; regenerate from JSON if needed:
```bash
python metadata_to_csv.py --json data/cadenza_data_train/metadata/train_metadata.json
python metadata_to_csv.py --json data/cadenza_data_valid/metadata/valid_metadata.json
python metadata_to_csv.py --json data/cadenza_data_eval/metadata/eval_metadata.json
```

---

## 2. Feature extraction (Whisper scoring)
Outputs already present in the metadata CSVs; rerun if you want to refresh.

Config: `feature_config.yaml` (section `whisper` plus shared `datasets`).
```bash
python whisper_inference.py --config feature_config.yaml
# run selected splits
python whisper_inference.py --config feature_config.yaml --splits=train,valid,eval
# run only right-channel datasets
python whisper_inference.py --config feature_config.yaml --channel=right
# run specific datasets by name
python whisper_inference.py --config feature_config.yaml --datasets eval_left eval_right
```

---

## 3. Audio quality (STOI/eSTOI/PESQ) and DNSMOS
Outputs already present in the metadata CSVs; rerun if you want to refresh.

Config: `feature_config.yaml` (section `metrics` plus shared `datasets`).
```bash
python compute_metrics.py --config feature_config.yaml
# run selected splits
python compute_metrics.py --config feature_config.yaml --splits=train,valid,eval
# quality only
python compute_metrics.py --config feature_config.yaml --quality --datasets signal_eval_left signal_eval_right
# DNS only
python compute_metrics.py --config feature_config.yaml --dns
```

---

## 4. Model training & inference

Config: `checkpoints/final/config.yaml`.

Train:
```bash
python train.py
```
Artifacts saved to `checkpoints/final/` (`model.pt`, `train_log.csv`).

Inference (runs valid + eval by default):
```bash
python infer.py
```
Override splits:
```bash
python infer.py infer.splits=[valid]
python infer.py infer.splits=[eval]
```
Outputs land in `checkpoints/final/` (`inference.csv`, `eval_inference.csv`).

---

## Citation
If you use this code, please cite:
```
@inproceedings{scow2026,
  title        = {Stereo Chorus of Whispers: Perceptually-Augmented Ear-Specific Intelligibility Prediction},
  author       = {Longbin Jin and Heungsoo Kim and Youngmin Kim},
  booktitle    = {Proc. IEEE ICASSP},
  year         = {2026},
  note         = {To appear}
}
```

