

# Margin-based Vision Transformer




## Contents
- [MVT-1.5 RICE](#mvt-15-rice)
- [MVT-1.1 MLCD](#mvt-11-mlcd)
- [MVT-1.0 UNICOM](#mvt-10-unicom)

## MVT-1.5 RICE 

 [[Model]](https://huggingface.co/DeepGlint-AI/rice-vit-large-patch14-560) [[Paper]](https://arxiv.org/abs/2507.20025) 




![68747470733a2f2f63617073756c652d72656e6465722e76657263656c2e6170702f6170693f747970653d776176696e6726636f6c6f723d6772616469656e7426637573746f6d436f6c6f724c6973743d3136266865696768743d3135302673656374696f6e3d6865616](https://github.com/user-attachments/assets/f6f6d512-f11d-4cf5-bbc7-686fa948b1d3)




###  Highlights
![470695215-38e89eea-8a73-4e3f-b43a-fa1ea6e32f0f](https://github.com/user-attachments/assets/e0de38b3-b20a-491e-9382-1839e9968481)


RICE efficiently processes diverse semantic regions
within the image using a single forward pass. The model jointly captures both general visual semantics (objects) and OCR semantics
(texts), seamlessly integrating them into a unified representation.

###  Experiments

![470696193-65b351ac-9399-4dac-8999-b4412286731a](https://github.com/user-attachments/assets/cd66223f-1757-4ff4-859c-19dd25f1246d)


Comprehensive performance comparison of RICE with state-of-the-art vision encoders. For all experiments within the LLaVA-NeXT framework, we adopt a high-resolution tiling strategy: each input image is divided into a 2×2+1 grid of crops, where each crop matches the pre-training resolution of the backbone model (e.g., 336px, 378px, or 560px). 

### How to use

#### 1. Standard Usage

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

#### 2. Using HuggingFace Transformers >= 4.51.3

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

### Visualize Semantic Features

![screenshot-20250725-232729](https://github.com/user-attachments/assets/0ff3b764-c5b6-4a10-a63c-89ccbc99d06b)

Using 2048-resolution images as input to a ViT-B/16 model, we project token features onto RGB channels via
PCA to visualize the semantic structure. Sequential frames (arranged vertically) illustrate the evolution of model attention, consistently
highlighting salient objects across time. The visualization reveals stable color patterns for tracked entities such as ice skaters, deers,
motorcyclists, and cyclists, demonstrating the model’s ability to maintain semantic focus throughout the sequence.


## [MVT-1.1 MLCD](https://github.com/deepglint/unicom)
## [MVT-1.0 UNICOM](https://github.com/deepglint/unicom)

**The authors are from DeepGlint team and Huawei London Research Institute.**

## ModelZoo

| Model | Download |
|-------|-------------|
| RICE-ViT-L-14-560px | [huggingface](https://huggingface.co/DeepGlint-AI/rice-vit-large-patch14-560) |
| MLCD-ViT-bigG-14-448px | [huggingface](https://huggingface.co/DeepGlint-AI/mlcd-vit-bigG-patch14-448) |
| MLCD-ViT-L-14-336px | [huggingface](https://huggingface.co/DeepGlint-AI/mlcd-vit-large-patch14-336) |
| MLCD-ViT-B-32-224px | [huggingface](https://huggingface.co/DeepGlint-AI/mlcd-vit-base-patch32-224) |


## Citation



```latex
@inproceedings{yinxie_2025_rice,
  title={Region-based Cluster Discrimination for Visual Representation Learning},
  author={Xie, Yin and Yang, Kaicheng and An, Xiang and Wu, Kun and Zhao, Yongle and Deng, Weimo and Ran, Zimin and Wang, Yumeng and Feng, Ziyong And Roy, Miles And Ismail, Elezi And Deng, Jiankang},
  booktitle={ICCV},
  year={2025}
}
@inproceedings{anxiang_2024_mlcd,
  title={Multi-label Cluster Discrimination for Visual Representation Learning},
  author={An, Xiang and Yang, Kaicheng and Dai, Xiangzi and Feng, Ziyong and Deng, Jiankang},
  booktitle={ECCV},
  year={2024}
}
@inproceedings{anxiang_2023_unicom,
  title={Unicom: Universal and Compact Representation Learning for Image Retrieval},
  author={An, Xiang and Deng, Jiankang and Yang, Kaicheng and Li, Jiawei and Feng, Ziyong and Guo, Jia and Yang, Jing and Liu, Tongliang},
  booktitle={ICLR},
  year={2023}
}
@inproceedings{anxiang_2022_partialfc,
    author={An, Xiang and Deng, Jiankang and Guo, Jia and Feng, Ziyong and Zhu, XuHan and Yang, Jing and Liu, Tongliang},
    title={Killing Two Birds With One Stone: Efficient and Robust Training of Face Recognition CNNs by Partial FC},
    booktitle={CVPR},
    year={2022},
}
@inproceedings{deng_2019_arcface,
  title={Arcface: Additive angular margin loss for deep face recognition},
  author={Deng, Jiankang and Guo, Jia and Xue, Niannan and Zafeiriou, Stefanos},
  booktitle={CVPR},
  year={2019}
}
```
