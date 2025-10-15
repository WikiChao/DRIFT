# DRIFT: Directional Reasoning Injection for Fine-Tuning   MLLMs

[![Paper](https://img.shields.io/badge/Paper-arXiv:{{ARXIV_ID}}-B31B1B.svg)]({{ARXIV_URL}})
[![Project Page](https://img.shields.io/badge/Project%20Page-Live-0A84FF)]({{PROJECT_PAGE_URL}})
[![Demo](https://img.shields.io/badge/Demo-Open%20in%20Colab-F9AB00?logo=googlecolab&logoColor=white)]({{COLAB_URL}})
[![Model](https://img.shields.io/badge/Model-HuggingFace-ff9a00?logo=huggingface)]({{HF_REPO_OR_SPACE_URL}})
[![Dataset](https://img.shields.io/badge/Dataset-Download-2b9348)]({{DATASET_URL}})
[![License](https://img.shields.io/badge/License-{{LICENSE_SHORT}}-black.svg)](./LICENSE)
[![Build](https://img.shields.io/github/actions/workflow/status/{{OWNER}}/{{REPO}}/ci.yml?logo=github)](https://github.com/{{OWNER}}/{{REPO}}/actions)
[![Coverage](https://img.shields.io/codecov/c/github/{{OWNER}}/{{REPO}}?logo=codecov)](https://codecov.io/gh/{{OWNER}}/{{REPO}})
[![Python](https://img.shields.io/badge/Python-{{PY_VERSION_RANGE}}-3776AB?logo=python)](#environment)
[![CUDA](https://img.shields.io/badge/CUDA-{{CUDA_VERSION}}-76B900?logo=nvidia)](#environment)
[![DOI](https://img.shields.io/badge/DOI-{{DOI}}-8A2BE2)]({{DOI_URL}})

<!-- Optional: hero figure -->
<p align="center">
  <img src="{{STATIC_OR_FIGURE_URL}}" alt="Overview figure" width="85%" />
</p>

<p align="center">
  <b>DRIFT transfers reasoning from DeeSeekR1 into QwenVL through gradient guidance.</b>
</p>

---

## Table of Contents
- [Abstract](#abstract)
- [Highlights](#highlights)
- [News](#news)
- [TL;DR / Demo](#tldr--demo)
- [Environment](#environment)
- [Installation](#installation)
- [Datasets](#datasets)
- [Pretrained Models](#pretrained-models)
- [Quick Start](#quick-start)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Ablations](#ablations)
- [Reproducibility Checklist](#reproducibility-checklist)
- [Repository Structure](#repository-structure)
- [Configuration](#configuration)
- [Logging & Experiment Tracking](#logging--experiment-tracking)
- [Export & Inference](#export--inference)
- [Model Card](#model-card)
- [Limitations & Ethics](#limitations--ethics)
- [FAQ](#faq)
- [Cite Us](#cite-us)
- [Acknowledgements](#acknowledgements)
- [License](#license)
- [Maintainers & Contact](#maintainers--contact)
- [Changelog](#changelog)

---

## Abstract
Multimodal large language models (MLLMs) are rapidly advancing, yet their reasoning ability often lags behind that of strong text-only counterparts. Existing methods to bridge this gap rely on supervised fine-tuning over large-scale multimodal reasoning data or reinforcement learning, both of which are resource-intensive. A promising alternative is \textit{model merging}, which interpolates parameters between reasoning-enhanced LLMs and multimodal variants. However, our analysis shows that naive merging is not always a ``free lunch'': its effectiveness varies drastically across model families, with some (e.g., LLaVA, Idefics) benefiting while others (e.g., Qwen) suffer performance degradation. To address this, we propose Directional Reasoning Injection for Fine-Tuning (DRIFT) MLLMs, a lightweight method that transfers reasoning knowledge in the gradient space, without destabilizing multimodal alignment. DRIFT precomputes a reasoning prior as the parameter-space difference between reasoning and multimodal variants, then uses it to bias gradients during multimodal fine-tuning. This approach preserves the simplicity of standard supervised fine-tuning pipelines while enabling efficient reasoning transfer. Extensive experiments on multimodal reasoning benchmarks, including MathVista and MathVerse, demonstrate that DRIFT consistently improves reasoning performance over naive merging and supervised fine-tuning, while matching or surpassing training-heavy methods at a fraction of the cost. 

---

## Highlights
- Novelty: {{KEY_NOVEL_POINT_1}}
- Performance: {{KEY_RESULT_OR_GAIN}}
- Efficiency/Scalability: {{KEY_EFFICIENCY_POINT}}
- Applications: {{KEY_APPLICATIONS}}
- Code: {{KEY_ENGINEERING_OR_DESIGN_DECISIONS}}

---

## News
- {{YYYY-MM-DD}}: {{MAJOR_ANNOUNCEMENT}}
- {{YYYY-MM-DD}}: {{RELEASE_OR_UPDATE}}
- {{YYYY-MM-DD}}: {{BENCHMARK_OR_AWARD}}

---

## TL;DR / Demo
- Colab: [Open Notebook]({{COLAB_URL}})
- Hugging Face Space: [Live Demo]({{HF_SPACE_URL}})
- Minimal example:
```bash
python -m {{PKG_NAME}}.infer --config configs/infer.yaml --input {{EXAMPLE_INPUT}}
```

---

## Environment


- Conda:
```bash
conda create -n {{ENV_NAME}} python={{PY_VERSION}} -y
conda activate {{ENV_NAME}}
pip install -e ".[dev]"  # or pip install -r requirements.txt
```

---

## Installation
```bash
git clone https://github.com/{{OWNER}}/{{REPO}}.git
cd {{REPO}}
pip install -e ".[all]"  # extras: [train,eval,docs]
# optional: pre-commit for style
pre-commit install
```

---

## Datasets
- Primary dataset: [{{DATASET_NAME}}]({{DATASET_URL}})
- Size: {{NUM_SAMPLES}} samples
- License: {{DATA_LICENSE}}
- Auto-download:
```bash
python -m {{PKG_NAME}}.data.prepare --dataset {{DATASET_NAME}} --root data/
```
- Manual:
  - Place files under `data/{{DATASET_NAME}}/` as:
```
data/
└── {{DATASET_NAME}}/
    ├── train/...
    ├── val/...
    └── test/...
```

---

## Pretrained Models
- Checkpoint (best val): [{{CKPT_NAME}}]({{CKPT_URL}}) ({{SIZE_MB}} MB)
- Weights (HF): [{{HF_MODEL_ID}}]({{HF_MODEL_URL}})
- Integrity:
```bash
sha256sum {{CHECKPOINT_FILENAME}}  # expected: {{SHA256}}
```

---

## Quick Start
- Run on a sample:
```bash
python -m {{PKG_NAME}}.infer \
  --config configs/infer.yaml \
  --input assets/examples/{{EXAMPLE_FILE}} \
  --weights {{PATH_OR_URL_TO_WEIGHTS}} \
  --out runs/infer/
```

- Programmatic:
```python
from {{PKG_NAME}} import load_model
model = load_model(weights="{{PATH_OR_URL_TO_WEIGHTS}}")
out = model.predict("{{EXAMPLE_INPUT}}")
print(out)
```

---

## Training
- Single GPU:
```bash
python -m {{PKG_NAME}}.train \
  --config configs/train/{{EXP_NAME}}.yaml \
  trainer.max_epochs={{EPOCHS}} \
  data.root=data/{{DATASET_NAME}}
```

- Multi-GPU (DDP):
```bash
torchrun --nproc_per_node={{NUM_GPUS}} -m {{PKG_NAME}}.train \
  --config configs/train/{{EXP_NAME}}.yaml
```

- Resume/finetune:
```bash
python -m {{PKG_NAME}}.train \
  --config configs/train/{{EXP_NAME}}.yaml \
  ckpt_path={{CKPT_PATH}} \
  model.lr={{LR}}
```

---

## Evaluation
- Reproduce Table {{TABLE_NUMBER}} ({{BENCHMARK_NAME}}):
```bash
python -m {{PKG_NAME}}.eval \
  --config configs/eval/{{EVAL_NAME}}.yaml \
  --weights {{CKPT_PATH_OR_URL}} \
  data.split=test
```
- Expected metrics:
```
Accuracy: {{ACC}} ± {{STD}}
F1: {{F1}}
mAP@50: {{MAP50}}
```
- Submit to leaderboard: {{LINK_OR_INSTRUCTIONS}}

---

## Results
- Main benchmark:
| Method | Backbone | Dataset | Metric | Score | +/− |
|-------:|:--------:|:-------:|:------:|:-----:|:---:|
| {{OURS}} | {{BB}} | {{DATASET}} | {{METRIC}} | {{SCORE}} | {{STD}} |
| {{BASELINE1}} | {{BB1}} | {{DATASET}} | {{METRIC}} | {{S1}} | {{}} |
| {{BASELINE2}} | {{BB2}} | {{DATASET}} | {{METRIC}} | {{S2}} | {{}} |

- Efficiency:
| Model | Params | FLOPs | Throughput | Latency |
|:-----:|-------:|------:|-----------:|--------:|
| {{OURS}} | {{P}}M | {{F}}G | {{T}} samples/s | {{L}} ms |

- Qualitative samples:
<p align="center">
  <img src="assets/figures/qualitative_1.png" width="90%"/>
</p>

---

## Reproducibility Checklist
- Code and configs match paper: Yes/No
- Random seeds fixed: `seed={{SEED}}`
- Exact training/eval scripts provided
- Environment pinned: `requirements.txt` / `environment.yml` / `Dockerfile`
- External data and preprocessing steps documented
- All results derive from included scripts or notebooks
- Artifact DOI: {{DOI}} (Zenodo/ Figshare)

To fully reproduce results:
```bash
bash scripts/reproduce_all.sh  # orchestrates data -> train -> eval -> tables
```

---

## Repository Structure
```
{{REPO}}/
├─ {{PKG_NAME}}/
│  ├─ data/              # dataset loaders and transforms
│  ├─ models/            # architectures and layers
│  ├─ lit/               # Lightning/Trainer modules (optional)
│  ├─ utils/             # common helpers
│  ├─ train.py           # CLI entry for training
│  ├─ eval.py            # CLI entry for evaluation
│  └─ infer.py           # CLI entry for inference
├─ configs/              # YAML configs for exps/evals/ablations
├─ scripts/              # bash utilities for reproduce, download, etc.
├─ assets/               # figures, example inputs/outputs
├─ tests/                # unit/integration tests
├─ docker/               # Dockerfiles
├─ requirements.txt      # or pyproject.toml
├─ environment.yml       # conda env (optional)
├─ CITATION.cff
├─ LICENSE
└─ README.md
```

---

## Configuration
- Override any config via CLI using dot notation:
```bash
python -m {{PKG_NAME}}.train --config configs/train/{{EXP}}.yaml trainer.max_epochs=200 model.lr=3e-4
```
- Example config snippet:
```yaml
# configs/train/base.yaml
seed: 42
data:
  name: {{DATASET_NAME}}
  root: data/{{DATASET_NAME}}
  batch_size: 64
model:
  name: {{MODEL_NAME}}
  backbone: {{BACKBONE}}
  lr: 3e-4
trainer:
  max_epochs: 100
  precision: 16
  devices: 1
```

---

## Logging & Experiment Tracking
- Local logging to `runs/`.
- Optional integrations:
  - Weights & Biases:
    ```bash
    wandb login
    python -m {{PKG_NAME}}.train logger=wandb logger.project={{WANDB_PROJECT}}
    ```
  - TensorBoard:
    ```bash
    tensorboard --logdir runs/
    ```

---

## Export & Inference
- Export to ONNX/TorchScript:
```bash
python -m {{PKG_NAME}}.export --weights {{CKPT}} --format onnx --opset 17
```
- Batch inference on a folder:
```bash
python -m {{PKG_NAME}}.infer --input_dir assets/examples/ --out runs/infer/
```

---

## Model Card
- Intended use: {{INTENDED_USE}}
- Out-of-scope use: {{OUT_OF_SCOPE}}
- Training data: {{DATASETS_AND_SOURCES}}
- Evaluation data: {{EVAL_SETS}}
- Metrics: {{METRICS}}
- Ethical considerations / risks: see [Limitations & Ethics](#limitations--ethics)

---

## Limitations & Ethics
- Known failure modes: {{FAILURE_MODES}}
- Bias and fairness: {{BIAS_CONCERNS}}
- Safety mitigations: {{MITIGATIONS}}
- Data licenses and usage constraints: {{DATA_LICENSE_NOTES}}

---

## FAQ
- Q: {{COMMON_QUESTION}}  
  A: {{CONCISE_ANSWER}}

- Q: {{ANOTHER_QUESTION}}  
  A: {{CONCISE_ANSWER}}

---

## Cite Us
If you find this work useful, please cite:
```bibtex
@article{{
  {{CITE_KEY}},
  title={{{{PROJECT_TITLE}}}},
  author={{{{AUTHORS}}}},
  journal={{{JOURNAL_OR_VENUE}}},  % or 'arXiv preprint arXiv:{{ARXIV_ID}}'
  year={{{{YEAR}}}},
  url={{{{PAPER_OR_PROJECT_URL}}}}
}}
```

Additionally, see [CITATION.cff](./CITATION.cff) for more formats.

---

## Acknowledgements
- {{ADVISORS_COLLABS_FUNDING}}
- This repo uses components from: {{UPSTREAM_REPOS_OR_LIBS}}

---

## License
This project is licensed under the {{LICENSE_LONG}}. See the [LICENSE](./LICENSE) file for details.

---

## Maintainers & Contact
- {{MAINTAINER_NAME}} ({{AFFILIATION}}) — {{EMAIL_OR_TWITTER}}
- {{SECOND_MAINTAINER}} — {{CONTACT}}
- Open an issue for questions, or email us directly.

---

## Changelog
- {{YYYY-MM-DD}} v{{VERSION}}: {{HIGHLIGHTS}}
- {{YYYY-MM-DD}} v{{PREV_VERSION}}: {{HIGHLIGHTS}}

<!--
Tips:
- Keep commands exact, runnable, and minimal.
- Pin versions for reproducibility.
- Provide scripts to fully reproduce paper tables and figures.
- Prefer relative paths and include small example assets.
- Validate configs in CI and test loading pretrained weights.
-->
