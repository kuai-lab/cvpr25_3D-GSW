# 3D-GSW: 3D Gaussian Splatting for Robust Watermarking

**IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) 2025**

Youngdong Jang, Hyunje Park, Feng Yang, Heeju Ko, Euijin Choo, Sangpil Kim

<span class="links">
  <a  href="https://arxiv.org/abs/2409.13222" rel="nofollow"><img src="https://img.shields.io/badge/cs.CV-2409.13222-b31b1b?logo=arxiv&logoColor=red" alt="ArXiv" style="max-width: 100%;"></a>
  <a href="https://cvpr.thecvf.com/virtual/2025/poster/35252" rel="nofollow"><img src="https://img.shields.io/badge/CVPR-2025-blue" alt="CVPR" style="max-width: 100%;"></a>
  <a href="https://github.com/graphdeco-inria/gaussian-splatting?tab=readme-ov-file" rel="nofollow"><img alt="Pre-trained Models" src="https://img.shields.io/badge/Pretrained Models-gray?logo=Github&label=GitHub&labelColor=black" style="max-width: 100%;"></a>
</span>

This repository is the official implementation associated with the paper "3D-GSW: 3D Gaussian Splatting for Robust Watermarking". 

<a href="https://kuaicv.com/"><img height="50" src="assets/korea_univ_logo.png"> </a> 
<a href="https://deepmind.google/"><img height="50" src="assets/Google_DeepMind_logo.png"> </a>
<a href="https://www.ualberta.ca/en/index.html"><img height="50" src="assets/university-of-alberta-vector-logo.png"> </a> 

**Abstract:** *As 3D Gaussian Splatting (3D-GS) gains significant attention and its commercial usage increases, the need for watermarking technologies to prevent unauthorized use of the 3D-GS models and rendered images has become increasingly important. In this paper, we introduce a robust watermarking method for 3D-GS that secures copyright of both the model and its rendered images. Our proposed method remains robust against distortions in rendered images and model attacks while maintaining high rendering quality. To achieve these objectives, we present Frequency-Guided Densification (FGD), which removes 3D Gaussians based on their contribution to rendering quality, enhancing real-time rendering and the robustness of the message. FGD utilizes Discrete Fourier Transform to split 3D Gaussians in high-frequency areas, improving rendering quality. Furthermore, we employ a gradient mask for 3D Gaussians and design a wavelet-subband loss to enhance rendering quality. Our experiments show that our method embeds the message in the rendered images invisibly and robustly against various attacks, including model distortion. Our method achieves superior performance in both rendering quality and watermark robustness while improving real-time rendering efficiency.*

## Data Preparation
To prepare the dataset and pre-trained weights for training and evaluation, follow these steps:

1. Download the Blender and LLFF dataset from [NeRF Dataset Link](https://drive.google.com/drive/folders/1cK3UDIJqKAAm7zyrxRYVFJ0BRMgrwhh4).
2. Download the Mip-NeRF 360 dataset from [Mip-NeRF 360 Dataset Link](https://jonbarron.info/mipnerf360/). 
3. Extract the downloaded dataset and place it in the `./dataset` directory. Your directory structure should look like this:
    ```
    cvpr25_3D-GSW/
    ├── dataset/
    │   ├── nerf_synthetic/
    │   │   ├── lego/
    │   │   ├── ...
    │   ├── llff/
    │   │   ├── fern/
    │   │   ├── ...
    │   ├── mipnerf360/
    │   │   ├── bonsai/
    │   │   ├── ...
    ├── finetuning_gaussian.py
    └── ...
    ```

4. Download the 3D-GS pre-trained weights from [3D-GS Pretrained Wieghts Link (Mip-NeRF 360)](https://github.com/graphdeco-inria/gaussian-splatting?tab=readme-ov-file).
5. Pre-train the 3D-GS as following as [Original 3D-GS paper Link](https://github.com/graphdeco-inria/gaussian-splatting?tab=readme-ov-file).
7. Place the pre-trained weights in the `./gaussian_models` directory. Your directory structure should look like this:
    ```
    cvpr25_3D-GSW/
    ├── gaussian_models/
    │   ├── nerf_synthetic/
    │   │   ├── lego/
    │   │   ├── ...
    │   ├── llff/
    │   │   ├── fern/
    │   │   ├── ...
    │   ├── mipnerf360/
    │   │   ├── bonsai/
    │   │   ├── ...
    ├── dataset/
    ├── finetuning_gaussian.py
    └── ...
    ```
8. Render the image using pre-trained model. Your final directory structure should look like this:
     ```
    cvpr25_3D-GSW/
    ├── gaussian_models/
    │   ├── nerf_synthetic/
    │   │   ├── lego/
    │   │   │   ├── iteration_30000/
    │   │   │   │   ├── point_cloud.ply
    │   │   │   ├── test/
    │   │   │   │   ├── ours_30000/
    │   │   │   │   │   ├── gt/
    │   │   │   │   │   ├── renders/  
    │   │   │   │   │   │   ├── 00000.png ────> We use it for evaluation !!!!
    │   │   │   │   │   │   ├── ...
    │   │   │   ├── cameras.json
    │   │   │   ├── cfg_args
    │   │   │   ├── ...
    │   │   ├── ...
    │   ├── ...
    ├── dataset/
    ├── finetuning_gaussian.py
    └── ...
    ```
9. Download the weights for perceptual loss from [PerceptualSimilarity Link](https://github.com/SteffenCzolbe/PerceptualSimilarity) and place them in the `./decoder/loss/losses` directory. Your directory structure should now include:
    ```
    cvpr25_3D-GSW/
    ├── decoder/
    │   ├── loss/
    │   │   ├── losses/
    │   │   │   ├── rgb_watson_vgg_trial0.pth
    │   │   │   └── ...
    │   │   └── ...
    ├── gaussian_models/
    ├── dataset/
    ├── finetuning_gaussian.py
    └── ...
    ```
Ensure that the paths in your configuration files are set correctly to match the locations of the dataset, pre-trained weights, and loss weights.

## Cloning the Repository

The repository contains submodules, thus please check it out with 
```
git clone https://github.com/kuai-lab/cvpr25_3D-GSW.git --recursive
```
## Overview

Tested on Ubuntu Linux 20.04 + PyTorch 1.13.1 + CUDA 11.7 in a Python 3.7.13

### Hardware Requirements

- CUDA-ready GPU with Compute Capability 7.0+
- 26 GB VRAM for Mip-NeRF 360 Dataset and 14 GB for NeRF-Synthetic Dataset and LLFF Dataset

### Setup

Our default, provided install method is based on Conda package and environment management:
```
conda env create --file environment/3d_gsw.yaml
conda activate 3d_gsw
pip install ./submodules/diff-gaussian-rasterization
pip install ./submodules/simple-knn
```
or
```
pip install -r environment/requirements.txt
pip install ./submodules/diff-gaussian-rasterization
pip install ./submodules/simple-knn
```

To use DWT, install pytorch wavelet [pytorch_wavelet Link](https://github.com/fbcotter/pytorch_wavelets)

## **Results and Weights**

We provide results and weights for all scenes presented in our paper.
- **Download**: [Link](https://kuaicv.synology.me/weights/cvpr2025/3dgsw/3dgsw_weights.zip)

## Docker image loading and running

We provide docker image and manual to reproduce our paper's result.

- [docker image link](https://kuaicv.synology.me/weights/cvpr2025/3dgsw/3dgsw_s42-m19701105.tar) sm_70–90 (V100, A100, A6000, L40, RTX 6000 Ada, H100, etc.)
- [docker image link (sm120)](https://kuaicv.synology.me/weights/cvpr2025/3dgsw/3dgsw_s42-m19701105-sm120.tar) sm_70–120 (all GPUs above, plus B200, RTX PRO 6000 Blackwell, RTX 50 series) 

[Docker instructions](Docker_RUN.md)

### Fine-tuning

To fine-tune the model to embed a watermark, simply use

```
. ./scripts/finetuning_gaussian.sh 0
```

<details>
<summary><span style="font-weight: bold;">Command Line Arguments for finetuning_gaussian.py</span></summary>

  #### --source_path / -s
  Path of the source directory containing a COLMAP or Synthetic NeRF data set.
  #### --model_path / -m 
  Path where the trained model should be stored (```output/<random>``` by default).
  #### --images / -i
  Alternative subdirectory for COLMAP images (```images``` by default).
  #### --eval
  Add this flag to use a MipNeRF360-style training/test split for evaluation.
  #### --resolution / -r
  Specifies resolution of the loaded images before training. If provided ```1, 2, 4``` or ```8```, uses original, 1/2, 1/4 or 1/8 resolution, respectively. For all other values, rescales the width to the given number while maintaining image aspect. **If not set and input image width exceeds 1.6K pixels, inputs are automatically rescaled to this target.**
  #### --data_device
  Specifies where to put the source image data, ```cuda``` by default, recommended to use ```cpu``` if training on large/high-resolution dataset, will reduce VRAM consumption, but slightly slow down training. Thanks to [HrsPythonix](https://github.com/HrsPythonix).
  #### --white_background / -w
  Add this flag to use white background instead of black (default), e.g., for evaluation of NeRF Synthetic dataset.
  #### --sh_degree
  Order of spherical harmonics to be used (no larger than 3). ```3``` by default.
  #### --convert_SHs_python
  Flag to make pipeline compute forward and backward of SHs with PyTorch instead of ours.
  #### --convert_cov3D_python
  Flag to make pipeline compute forward and backward of the 3D covariance with PyTorch instead of ours.
  #### --debug
  Enables debug mode if you experience erros. If the rasterizer fails, a ```dump``` file is created that you may forward to us in an issue so we can take a look.
  #### --debug_from
  Debugging is **slow**. You may specify an iteration (starting from 0) after which the above debugging becomes active.
  #### --iterations
  Number of total iterations to train for, ```30_000``` by default.
  #### --ip
  IP to start GUI server on, ```127.0.0.1``` by default.
  #### --port 
  Port to use for GUI server, ```6009``` by default.
  #### --epoch
  Epoch for embedding watermark.
  #### --decoder_att
  Decoder attribution path.
  #### --patch
  Patch size for FGD.
  #### --proportion
  Patch selection ratio for FGD.
  #### --exp_name
  Define experiment name.
  #### --lambda_lpips
  Influence of lpips on total loss. 
  #### --lambda_i
  Influence of reconstruction on total loss. 
  #### --lambda_subband
  Influence of high-frequency on total loss. 
  #### --lambda_wm
  Influence of watermark on total loss. 
  #### --alpha
  Influence of gradient mask. 
  #### --seed
  Fix the random seed for reproducing and use the seed to change the message.
</details>

For reproducibility, the five seeds used in our experiments are shared below
<details>
<summary><span style="font-weight: bold;">Seeds</span></summary>
  777, 19701105, 19911108, 19961124, 20240519
</details>

### Rendering
To rendering image, simply use
```
. ./scripts/render.sh 0
```

<details>
<summary><span style="font-weight: bold;">Command Line Arguments for render.py</span></summary>

  #### --source_path / -s
  Path of the source directory containing a COLMAP or Synthetic NeRF data set.
  #### --model_path / -m 
  Path of the pre-trained model directory.
  #### --images / -i
  Alternative subdirectory for COLMAP images (```images``` by default).
  #### --eval
  Add this flag to use a MipNeRF360-style training/test split for evaluation.
  #### --resolution / -r
  Changes the resolution of the loaded images before training. If provided ```1, 2, 4``` or ```8```, uses original, 1/2, 1/4 or 1/8 resolution, respectively. For all other values, rescales the width to the given number while maintaining image aspect. ```1``` by default.
  #### --white_background / -w
  Add this flag to use white background instead of black (default), e.g., for evaluation of NeRF Synthetic dataset.
  #### --skip_train
  Flag to skip rendering the training set.
  #### --skip_test
  Flag to skip rendering the test set.
  #### --iteration 
  Epoch number of the fine-tuned model to use for rendering.
  #### --exp_name 
  Path of the fine-tuned model directory. The rendered images will be saved here.
  #### --wm 
  Load the watermarked model
</details>

### Evaluation
To evaluate the bit accuracy and rendering quality, simply use
```
. ./scripts/metrics.sh 0
```

<details>
<summary><span style="font-weight: bold;">Command Line Arguments for metrics.py</span></summary>

  #### --wm_path 
  Path of the watermarked images.
  #### --gt_image_path 
  Path of the watermarked images rendered by the pre-trained model.
  #### --epoch 
  Epoch number of the fine-tuned model to use for evaluation.
  #### --decoder_att 
  Decoder attribution path.
  #### --seed 
  Same seed for the fine-tuning.
</details>

<section class="section" id="BibTeX">
  <div class="container is-max-desktop content">
    <h2 class="title">BibTeX</h2>
    <pre><code>@inproceedings{jang20253d,
  title={3d-gsw: 3d gaussian splatting for robust watermarking},
  author={Jang, Youngdong and Park, Hyunje and Yang, Feng and Ko, Heeju and Choo, Euijin and Kim, Sangpil},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={5938--5948},
  year={2025}
}</code></pre>
  </div>
</section>
