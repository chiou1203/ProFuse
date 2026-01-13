# ProFuse: Efficient Cross-View Context Fusion for Open-Vocabulary 3D Gaussian Splatting

🌐 [Project](https://chiou1203.github.io/ProFuse/) | 📄 [Paper](https://arxiv.org/abs/2601.04754) |🤗 [Hugging Face Paper](https://huggingface.co/papers/2601.04754) |🤗 [Hugging Face Demo](https://huggingface.co/spaces/remiii25/ProFuse_Open-Vocabulary_Demo) |🔶 [Custom Demo](https://colab.research.google.com/drive/16DRjfcYU_ZAJTzn8J1AnZl3rcwqGGPnJ?copy=true)

![ProFuse framework](assets/圖片64.png)

ProFuse is an efficient context-aware framework for open-vocabulary 3D scene understanding with 3D Gaussian Splatting. The pipeline enhances a direct registration setup with a dense correspondence–guided pre-registration phase, adding minimal overhead and requiring no render-supervised fine-tuning.

## 0. Installation
Clone the repo

```bash
git clone https://github.com/chiou1203/ProFuse.git
cd ProFuse
```

Setup

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install fused-local-corr==0.1.1 pycolmap hydra-core tqdm torchmetrics lpips matplotlib rich plyfile \
            imageio imageio-ffmpeg plotly scikit-learn moviepy==2.1.1 ffmpeg numpy==1.26.4 open_clip_torch
pip install -q --no-deps loguru
```

Install submodules for pre-registration

```bash
cd ProFuse/pre_registration

mkdir -p submodules

git clone --recursive https://github.com/Parskatt/RoMa.git submodules/RoMa
git clone --recursive https://github.com/chiou1203/gaussian-splatting submodules/gaussian-splatting
cd submodules/gaussian-splatting
git checkout profuse-v1

cd ProFuse/pre_registration
pip install -q submodules/gaussian-splatting/submodules/diff-gaussian-rasterization
pip install -q --no-deps submodules/RoMa
```

Install submodules for registration

```bash
cd ProFuse/feature_registration
pip install -q submodules/langsplat-rasterization \
                submodules/segment-anything-langsplat \
                submodules/simple-knn \
                ninja kmeans_pytorch faiss-cpu
```

## 1. Data preparation
For pre-registration, prepare the scene folder like the following:

```text
data_root/
├─ images/
├─ sparse/
└─ language_features/
```

For registration, move the 3DGS scene folder to data root :

```text
data_root/
├─ images/
├─ sparse/
└─ language_features/
└─ GS/
└─ input.ply
```

## 2. Pre-registration
You can run the following script for pre-registration.
Please replace scene_dir with your own dataset directory.
```bash
chmod +x pre_registration.sh 
./pre_registration.sh  --scene_dir /content/ramen
```
After pre-registration is done, the Gaussian scene will be under the out_pre_registration folder, and 3D Context Proposal related metadata will be written into the language_features folder.

## 3. Feature registration
You can run the following script to do feature registration. 
Please make sure both the gs folder and language feature folder with context proposal metadata exists.
```bash
chmod +x registration.sh
./registration.sh
```

## 4. 3D object selection
![](assets/圖片63.png)
(TBA)
## 5. 3D point cloud understanding
![](assets/圖片62.png)
(TBA)
## 6. ToDo list

- [ ] Data preprocessing
- [ ] Evaluation
- [ ] Pretrained checkpoint

## 6. Citation
If you find our work useful, please consider cite it in your work.
```text
@misc{chiou2026profuseefficientcrossviewcontext,
      title={ProFuse: Efficient Cross-View Context Fusion for Open-Vocabulary 3D Gaussian Splatting}, 
      author={Yen-Jen Chiou and Wei-Tse Cheng and Yuan-Fu Yang},
      year={2026},
      eprint={2601.04754},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2601.04754}, 
}
```
