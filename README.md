# Margin based Vision Transformer


## MVT-1.5 RICE  


### 💡 Highlights
<img width="3121" height="790" alt="image" src="https://github.com/user-attachments/assets/38e89eea-8a73-4e3f-b43a-fa1ea6e32f0f" />

RICE efficiently processes diverse semantic regions
within the image using a single forward pass. The model jointly captures both general visual semantics (objects) and OCR semantics
(texts), seamlessly integrating them into a unified representation.

### 💡 Experiments

<img width="3182" height="1656" alt="image" src="https://github.com/user-attachments/assets/65b351ac-9399-4dac-8999-b4412286731a" />

Comprehensive performance comparison of RICE with state-of-the-art vision encoders. For all experiments within the LLaVA-NeXT framework, we adopt a high-resolution tiling strategy: each input image is divided into a 2×2+1 grid of crops, where each crop matches the pre-training resolution of the backbone model (e.g., 336px, 378px, or 560px). 



## [MVT-1.1 MLCD](https://github.com/deepglint/unicom)
## [MVT-1.0 UNICOM](https://github.com/deepglint/unicom)

**The authors are from DeepGlint team and Huawei London Research Institute.**

## Citation



```latex
@inproceedings{yinxie_2024_rice,
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
