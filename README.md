# DRIFT: Directional Reasoning Injection for Fine-Tuning   MLLMs

[![Paper](https://img.shields.io/badge/Paper-arXiv:{{ARXIV_ID}}-B31B1B.svg)]({{ARXIV_URL}})
[![Project Page](https://img.shields.io/badge/Project%20Page-DRIFT-0A84FF)](https://github.com/WikiChao/DRIFT)
[![Model](https://img.shields.io/badge/Model-HuggingFace-ff9a00?logo=huggingface)](https://huggingface.co/ChaoHuangCS/DRIFT-VL-7B)
[![Dataset](https://img.shields.io/badge/Dataset-Download-2b9348)](https://huggingface.co/datasets/ChaoHuangCS/DRIFT-TL-Distill-4K)
[![License](https://img.shields.io/badge/License-MIT-black.svg)](./LICENSE)

<!-- Optional: hero figure -->
<p align="center">
  <img src="asset/method.png" alt="Overview figure" width="85%" />
</p>

<p align="center">
  <b>DRIFT transfers reasoning from DeeSeekR1 into QwenVL through gradient guidance.</b>
</p>


---

## Abstract
Multimodal large language models (MLLMs) are rapidly advancing, yet their reasoning ability often lags behind that of strong text-only counterparts. Existing methods to bridge this gap rely on supervised fine-tuning over large-scale multimodal reasoning data or reinforcement learning, both of which are resource-intensive. A promising alternative is \textit{model merging}, which interpolates parameters between reasoning-enhanced LLMs and multimodal variants. However, our analysis shows that naive merging is not always a "free lunch": its effectiveness varies drastically across model families, with some (e.g., LLaVA, Idefics) benefiting while others (e.g., Qwen) suffer performance degradation. To address this, we propose Directional Reasoning Injection for Fine-Tuning (DRIFT) MLLMs, a lightweight method that transfers reasoning knowledge in the gradient space, without destabilizing multimodal alignment. DRIFT precomputes a reasoning prior as the parameter-space difference between reasoning and multimodal variants, then uses it to bias gradients during multimodal fine-tuning. This approach preserves the simplicity of standard supervised fine-tuning pipelines while enabling efficient reasoning transfer. Extensive experiments on multimodal reasoning benchmarks, including MathVista and MathVerse, demonstrate that DRIFT consistently improves reasoning performance over naive merging and supervised fine-tuning, while matching or surpassing training-heavy methods at a fraction of the cost. 

---

## News
- 2025-10-16 — Initial code release

---

## Environment

DRIFT can be integrated into most LLM/VLM training stacks. This repository provides a reference implementation compatible with [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory).

- Conda:
```bash
# create and activate environment
conda create -n drift python=3.12 -y
conda activate drift
cd LLaMA-Factory
pip install -e ".[torch,metrics]" --no-build-isolation
```

---

## Quick Start
To train the model, you may need to first download the dataset locally:
```python
conda activate drift
git lfs install
cd LLaMA-Factory
git clone https://huggingface.co/datasets/ChaoHuangCS/DRIFT-TL-Distill-4K
```

- Then run on an example script:
```bash
llamafactory-cli train examples/train_full_merge/qwen2_5vl_full_sft_merge.yaml
```


---

## Datasets
Our dataset is available on Hugging Face: [ChaoHuangCS/DRIFT-TL-Distill-4K](https://huggingface.co/datasets/ChaoHuangCS/DRIFT-TL-Distill-4K)

Quick load:
```python
from datasets import load_dataset
ds = load_dataset("ChaoHuangCS/DRIFT-TL-Distill-4K")
print(ds)
```

---

## Models
Our model is available on Hugging Face: [ChaoHuangCS/DRIFT-VL-7B](https://huggingface.co/ChaoHuangCS/DRIFT-VL-7B)

Quick load:
```python
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
proc = AutoProcessor.from_pretrained("ChaoHuangCS/DRIFT-VL-7B", trust_remote_code=True)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained("ChaoHuangCS/DRIFT-VL-7B", torch_dtype="auto", device_map="auto", trust_remote_code=True)
```

---

## Evaluation
We use VLMEvalKit for evaluation. Please follow their instructions: https://github.com/open-compass/VLMEvalKit

Quick start:
```bash
# clone and install
git clone https://github.com/open-compass/VLMEvalKit.git
cd VLMEvalKit
pip install -e .

# then follow the repo's Quick Start to select datasets and model adapters
# example (refer to VLMEvalKit docs for exact flags/model tags):
# python run.py --model {{VLMEvalKit_MODEL_TAG}} --datasets {{DATASET_NAME}}
```

---

## Results
- Main benchmark:

| **Model** | **MathVista** | **MathVision** | **MathVerse** | **WeMath** | **LogicVista** |
|---|---:|---:|---:|---:|---:|
| R1-Onevision-7B | 64.1 | 29.9 | 40.0 | — | 61.8 |
| OpenVLThinker-7B | 65.3 | 23.0 | 38.1 | 35.2 | 44.5 |
| R1-VL-7B | 63.5 | 24.7 | 40.0 | — | — |
| X-REASONER (Liu et al., 2025) | 69.0 | 29.6 | — | — | — |
| QwenVL2.5 (SFT) | 68.7 | 25.1 | 42.0 | 33.3 | 45.6 |
| **DRIFT (Ours)** | **70.3 (+1.6)** | **26.5 (+1.5)** | **43.7 (+1.7)** | **36.9 (+3.6)** | **45.6 (+0.0)** |

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

---

## Acknowledgements
- This project builds on [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory). Thanks to the authors and contributors.
- Evaluation leverages [VLMEvalKit](https://github.com/open-compass/VLMEvalKit).

---

## License
This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

<!--
Tips:
- Keep commands exact, runnable, and minimal.
- Pin versions for reproducibility.
- Provide scripts to fully reproduce paper tables and figures.
- Prefer relative paths and include small example assets.
- Validate configs in CI and test loading pretrained weights.
-->
