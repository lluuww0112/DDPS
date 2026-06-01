# VQAv2 Download

`vqa_v2.py` downloads a full split from `lmms-lab/VQAv2` and saves it with Hugging Face `datasets.save_to_disk()`.

## Requirements

```bash
pip install datasets
```

## Usage

```bash
python data/download/vqa_v2.py val /home/data/VQAv2/val
```

Equivalent named-argument form:

```bash
python data/download/vqa_v2.py --split val --output-dir /home/data/VQAv2/val
```

Supported split names:

- `train`
- `val`, `valid`, `validation`, `dev`
- `test`

The output directory must be empty. After downloading, point `config/eval.yaml` to the parent dataset root, for example:

```yaml
vqav2:
  dataset_root: "/home/data/VQAv2"
  split: "val"
```
