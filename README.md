<div align="center">

# Margin-based Vision Transformer (MVT)

[![ICCV](https://img.shields.io/badge/ICCV-2025-blue)](https://arxiv.org/abs/2507.20025)
[![ECCV](https://img.shields.io/badge/ECCV-2024-green)](https://github.com/deepglint/unicom)
[![ICLR](https://img.shields.io/badge/ICLR-2023-orange)](https://github.com/deepglint/unicom)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Models-yellow)](https://huggingface.co/DeepGlint-AI)

</div>

---

## 📰 News

- **[2025.11]** 🎉 MVT-1.5 (RICE) accepted to **ICCV 2025**! [[Paper]](https://arxiv.org/abs/2507.20025) [[Model]](https://huggingface.co/DeepGlint-AI/rice-vit-large-patch14-560)
- **[2024.07]** 🎉 MVT-1.1 (MLCD) accepted to **ECCV 2024**! [[Code]](https://github.com/deepglint/unicom)
- **[2023.01]** 🎉 MVT-1.0 (UNICOM) accepted to **ICLR 2023**! [[Code]](https://github.com/deepglint/unicom)

---

## 🔬 Introduction

The **Margin-based Vision Transformer (MVT)** series represents a family of state-of-the-art vision encoders designed for universal visual representation learning. The latest version, **RICE (Region-based Cluster Discrimination)**, advances visual understanding by processing diverse semantic regions within images using a single forward pass.

### MVT-1.5: RICE (ICCV 2025)

**RICE** introduces a novel approach to visual representation learning that jointly captures:
- **General visual semantics** (objects, scenes)
- **OCR semantics** (text within images)
- **Unified representations** seamlessly integrating both modalities

This enables superior performance across multiple vision tasks including image retrieval, visual question answering, and multimodal understanding.

<div align="center">

![RICE Highlights](https://github.com/user-attachments/assets/e0de38b3-b20a-491e-9382-1839e9968481)

*Figure 1: RICE architecture efficiently processes diverse semantic regions within images using region-based cluster discrimination.*

</div>

---

## 📊 Experiments

RICE demonstrates state-of-the-art performance across multiple vision benchmarks. Using the LLaVA-NeXT framework with a high-resolution tiling strategy (2×2+1 grid), RICE achieves superior results compared to existing vision encoders.

<div align="center">

![Experimental Results](https://github.com/user-attachments/assets/cd66223f-1757-4ff4-859c-19dd25f1246d)

*Table 1: Comprehensive performance comparison of RICE with state-of-the-art vision encoders. Each input image is divided into a 2×2+1 grid of crops matching the pre-training resolution (e.g., 336px, 378px, or 560px).*

</div> 

---

## 🚀 Usage

### Standard Usage

```python
# Install dependencies
# pip install torch transformers
# git clone https://github.com/deepglint/unicom
# cd unicom/mlcd

from vit_rope2d_hf import MLCDVisionModel
from transformers import CLIPImageProcessor
from PIL import Image
import requests
import torch

# Load model and processor
model = MLCDVisionModel.from_pretrained("DeepGlint-AI/rice-vit-large-patch14-560")
processor = CLIPImageProcessor.from_pretrained("DeepGlint-AI/rice-vit-large-patch14-560")

# Load and process an image
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
inputs = processor(images=image, return_tensors="pt")

# Extract visual features
with torch.no_grad():
    outputs = model(**inputs)
features = outputs.last_hidden_state

print(f"Extracted features shape: {features.shape}")
```

### Using HuggingFace Transformers (≥4.51.3)

```python
# pip install torch transformers>=4.51.3

from transformers import AutoProcessor, AutoModel
from PIL import Image
import requests
import torch

# Load model and processor
model = AutoModel.from_pretrained("DeepGlint-AI/rice-vit-large-patch14-560")
processor = AutoProcessor.from_pretrained("DeepGlint-AI/rice-vit-large-patch14-560")

# Load and process an image
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
inputs = processor(images=image, return_tensors="pt")

# Extract visual features
with torch.no_grad():
    outputs = model(**inputs)
features = outputs.last_hidden_state[0]

print(f"Extracted features shape: {features.shape}")
```


---

## 🎨 Visualization

RICE maintains stable semantic focus across sequential frames, demonstrating robust visual understanding and tracking capabilities.

<div align="center">

![Qualitative Results](https://github.com/user-attachments/assets/0ff3b764-c5b6-4a10-a63c-89ccbc99d06b)

*Figure 2: Semantic feature visualization using 2048-resolution images as input to ViT-B/16. Token features are projected onto RGB channels via PCA. Sequential frames (arranged vertically) show consistent attention on salient objects (ice skaters, deers, motorcyclists, cyclists), with stable color patterns maintained throughout the sequence.*

</div>

---

## 📦 Model Zoo

All models are available on Hugging Face for easy integration.

| Model | Resolution | Patch Size | Download |
|-------|------------|------------|----------|
| RICE-ViT-L-14 | 560px | 14 | [🤗 HuggingFace](https://huggingface.co/DeepGlint-AI/rice-vit-large-patch14-560) |
| MLCD-ViT-bigG-14 | 448px | 14 | [🤗 HuggingFace](https://huggingface.co/DeepGlint-AI/mlcd-vit-bigG-patch14-448) |
| MLCD-ViT-L-14 | 336px | 14 | [🤗 HuggingFace](https://huggingface.co/DeepGlint-AI/mlcd-vit-large-patch14-336) |
| MLCD-ViT-B-32 | 224px | 32 | [🤗 HuggingFace](https://huggingface.co/DeepGlint-AI/mlcd-vit-base-patch32-224) |

### Related Repositories

- **MVT-1.1 (MLCD)**: [github.com/deepglint/unicom](https://github.com/deepglint/unicom)
- **MVT-1.0 (UNICOM)**: [github.com/deepglint/unicom](https://github.com/deepglint/unicom)

---

## ❓ FAQ

### Q1: How can I reproduce the ViT-L-14-336px results from the paper?

**A:** RICE-ViT uses **RoPE2D** (Rotary Position Embedding 2D), which provides flexible resolution support. While the `rice-vit-large-patch14-560` model is trained at 560px, you can directly input images at 336px resolution—simply set `crop_size` and `shortest_edge` to 336 in your `preprocessor_config.json`, and the model will process the smaller resolution without any architectural changes.

However, please note that **you cannot exactly reproduce the 336px results from the paper** using the 560px RICE-ViT checkpoint. The results may be slightly lower because the ViT-L-14-336px model in the paper was trained specifically and exclusively at 336px resolution. With the publicly available 560px checkpoint, you can only reproduce the 560px results reported in the paper.

If you need to match 336px performance exactly, you would need a checkpoint that was specifically trained at 336px resolution.

### Q2: Which MLCDVisionModel should I use - the one from LLaVA-NEXT or the one from Transformers?

**A:** Both versions are equivalent and produce the same results. You can use either:

1. **From LLaVA-NEXT** (via `unicom/mlcd/vit_rope2d_hf.py`):
   ```python
   from vit_rope2d_hf import MLCDVisionModel
   model = MLCDVisionModel.from_pretrained("DeepGlint-AI/rice-vit-large-patch14-560")
   ```

2. **From Transformers** (≥4.51.3):
   ```python
   from transformers import AutoModel
   model = AutoModel.from_pretrained("DeepGlint-AI/rice-vit-large-patch14-560")
   ```

The model architecture and weights are identical, so feel free to use whichever is more convenient for your workflow.

---

## 📝 Citation

If you find this work useful, please cite our papers:

### RICE (ICCV 2025)

```bibtex
@inproceedings{yinxie_2025_rice,
  title={Region-based Cluster Discrimination for Visual Representation Learning},
  author={Xie, Yin and Yang, Kaicheng and An, Xiang and Wu, Kun and Zhao, Yongle and Deng, Weimo and Ran, Zimin and Wang, Yumeng and Feng, Ziyong And Roy, Miles And Ismail, Elezi And Deng, Jiankang},
  booktitle={ICCV},
  year={2025}
}
```

### MLCD (ECCV 2024)

```bibtex
@inproceedings{anxiang_2024_mlcd,
  title={Multi-label Cluster Discrimination for Visual Representation Learning},
  author={An, Xiang and Yang, Kaicheng and Dai, Xiangzi and Feng, Ziyong and Deng, Jiankang},
  booktitle={ECCV},
  year={2024}
}
```

### UNICOM (ICLR 2023)

```bibtex
@inproceedings{anxiang_2023_unicom,
  title={Unicom: Universal and Compact Representation Learning for Image Retrieval},
  author={An, Xiang and Deng, Jiankang and Yang, Kaicheng and Li, Jiawei and Feng, Ziyong and Guo, Jia and Yang, Jing and Liu, Tongliang},
  booktitle={ICLR},
  year={2023}
}
```

### Related Work

```bibtex
@inproceedings{anxiang_2022_partialfc,
  author={An, Xiang and Deng, Jiankang and Guo, Jia and Feng, Ziyong and Zhu, XuHan and Yang, Jing and Liu, Tongliang},
  title={Killing Two Birds With One Stone: Efficient and Robust Training of Face Recognition CNNs by Partial FC},
  booktitle={CVPR},
  year={2022}
}

@inproceedings{deng_2019_arcface,
  title={Arcface: Additive angular margin loss for deep face recognition},
  author={Deng, Jiankang and Guo, Jia and Xue, Niannan and Zafeiriou, Stefanos},
  booktitle={CVPR},
  year={2019}
}
```

---

<div align="center">

**[Paper](https://arxiv.org/abs/2507.20025)** | **[Models](https://huggingface.co/DeepGlint-AI)** | **[Code](https://github.com/deepglint/unicom)**

</div>
