# Margin based Vision Transformer


## MVT-1.5 RICE 

 [[Model]](https://huggingface.co/DeepGlint-AI/rice-vit-large-patch14-560) [[Paper]](https://github.com/deepglint/MVT/blob/main/paper.pdf) 


###  Highlights
![470695215-38e89eea-8a73-4e3f-b43a-fa1ea6e32f0f](https://github.com/user-attachments/assets/e0de38b3-b20a-491e-9382-1839e9968481)


RICE efficiently processes diverse semantic regions
within the image using a single forward pass. The model jointly captures both general visual semantics (objects) and OCR semantics
(texts), seamlessly integrating them into a unified representation.

###  Experiments

![470696193-65b351ac-9399-4dac-8999-b4412286731a](https://github.com/user-attachments/assets/cd66223f-1757-4ff4-859c-19dd25f1246d)


Comprehensive performance comparison of RICE with state-of-the-art vision encoders. For all experiments within the LLaVA-NeXT framework, we adopt a high-resolution tiling strategy: each input image is divided into a 2×2+1 grid of crops, where each crop matches the pre-training resolution of the backbone model (e.g., 336px, 378px, or 560px). 

### How to use

```python
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

# Process single image
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
inputs = processor(images=image, return_tensors="pt")

# Get visual features
with torch.no_grad():
    outputs = model(**inputs)
features = outputs.last_hidden_state

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
