# 3D-GSW: 3D Gaussian Splatting for Robust Watermarking

**IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) 2025**

Youngdong Jang, Hyunje Park, Feng Yang, Heeju Ko, Euijin Choo, Sangpil Kim

<span class="links">
  <a  href="https://arxiv.org/abs/2409.13222" rel="nofollow"><img src="https://img.shields.io/badge/cs.CV-2409.13222-b31b1b?logo=arxiv&logoColor=red" alt="ArXiv" style="max-width: 100%;"></a>
  <a  href="https://arxiv.org/abs/2409.13222" rel="nofollow"><img alt="Static Badge" src="https://img.shields.io/badge/Paper-gray?logo=data%3Aimage%2Fpng%3Bbase64%2CiVBORw0KGgoAAAANSUhEUgAAADIAAAAuCAYAAABqK0pRAAAKAElEQVRo3u1aWW8b1xlNkfahQG0nLlwg6IIW%2FQnuH%2Bjv6PbsN%2F8Bo08BGsBLU%2BexNYq2iW35wW%2FKQ2HXjrWLnBmSM9xFiqS4WItJStRCUvLXc747Q116QUPJdRIgA1zM8M72Leec77sjvfPOt9u329drE5Fzg8HgV%2F3%2B8MN%2Bv%2F%2Fv%2Ff39eYy5%2F8c4ODiwfh%2Fo7z28r1Zb%2BytM%2Be6Xsfc7luHvDYfDD4%2BOjq5hf%2FXw8DCG47685Q3vxHguOzs9SSSTPkz7fmTgu3t7ez%2BDt7%2FgwO%2Bfc48bPsNYhsFL4Ui8%2BNDnz58L5k88aBSfYY9er6fz0dYfDGRzc2v0e2dnB9fsysbGJh1xRo7gxg9wvoybe7zOGq%2FcXnzxacb29rbu7Y1Gn8iRzc3NH%2FP%2Btw0ROvDs2TMejM0D99J%2B1j6xI%2BtfhSPl1ob0Bodj81P37slfbt4cZeVr7whckblSUxIbu2Ozt2%2FfkWvXr785R%2FggyKoOyKxAqWSAYwiBQP50zj7Pee55jse8fjgYKqlf50i60pBGbzA2e%2BfuXblx48abc6SHG2afzMjMkycyPzcnqURCEp4rszOz4jqOzM%2FOytLCoizML0iQ8vV4eXFR3HhcFjE3PzsnM198Ifl84bU5WcikpLHbfQlaN79Z0BL529ID%2BUdyRoZHx1m7OzUlf%2F7442%2BOIyR7o96QDhTKlltm5DqhFUKSMK2v1U%2FvCHnRB%2B6PDo%2F%2Bp2GRlNq1wi56L17f7nRees7tO3fk6rVrI%2Bd2d3elXF49vSOrpZL4yaS0251Xnl%2BrVCTjB7JSKEoRfPATSSkXV%2FS4kM3inC%2FZIC2ZVEqviQykI81mU3a622OCQGjd%2FOST0XV7J3GE0acRNRjXQDrXKlVZq%2FJ4DQ6VcVzDuVWkeg37qipTEca6yzGIQVISjivxpWXdJ3kM8jtLS%2BJ7niQgApynskUbBcTBvft7%2B6O5z27fHpPfvd09qeBdEznSqtf1hV4sLkEihegmNMJpRFMNiTuS8hL6O5%2FJquOrKyuYdzHnm%2FuSqdE9Pq51YzEp5PL6LF7T7x874uJ5HgaNHWXk7pRcvXp1DFrVSm0yR%2Bq1mr5MjcILGEkaQ1h4MMiDMw4i7iGKNHYXDywXCuJizofjSc2EI0nXU7glHU%2BfxWuZNTrJLEZbNpORJghv8%2Bdfn34qf%2Froo9NxhHgPwAcaTQOyQaCZyKbTmgkaWsjlJIffHP0Dk5GU6yqEUq7JVnxxCRkl1Bx9Du%2F1vaQeR3yg8a3WU9lB43hkcWR6%2BnOZmro3cgTrD2k0mpM50mBGksYADsLHXV4OjzNKWjpAmKQQdVbxarkM%2BMTVSe7TIHcSjtFwOkg4MosRLG2O%2BCykKJxs3aNtfX1jTEz2wB82khM7QgMZOcKDkGBklSswIpdJq8E0iJlje1IpGUey6axmw0DKHTnHZzCLhFALKmXXjET4Lps3efCpZpFboVWaEFp1qJLiGlhOqSMGTjSOGSHMdA6RJuzoSHtrSyNeTAdw0FO5zYSw5D3kBWW4BOmtAuuHwxBaGGlcYxzpjzmSSQWj7h6LPqlV1yYle1XhQJJn8JIgzIiSV7HuKf7VIWTlAPhdyefVEc57IXx4Xnmj1zo6oiwPwuiTIzSaQWP1HsENmc4AwpEAnKiO1KtVzURGlSulL1dsY87IakKdoaMGWgMpg%2BxUJAMjUxeU4K5nnMB81k9rQNLIrF38kuBRFJBoq6MEUACOObI3ufyy8KliwWhGj8ZEXKGRAWuIws5VR0n2Cio%2FocdhMrYMfjh6DdXO1ww5%2BgwWTZsPHrkEx21HSHY%2Bd8yRam3SgtiQfDqjLye%2BTVEzhEzhpSx4EUx4jvJL3LOuqDqxwuNeXsvaQ3hqMPCbGaWzdh3xwEfCct9yJAc%2B8doIWiciexMZSVCBgFHKrk3eSL0IKdYLSvMBsM3ao7DyonO4B4YUcZ7ZZIdAyWam%2BZzh8BhaTsw4b2eAvEkBcnZGVlcrkzmytbGh%2FVUTOGWzuI6CRXIy8uyAo9VgtGfUWEeYDWbL0Qqf1AxFnGLFZ5YiHkV1hPcWoGQ18HJgwS0DGSfZTwUtpp03RuPF1vtVWylsURj9hGsIz8hTxglPSjWdYj9Gp2xHctmc8shWLTaSrEnHqoWMlCfMyHa3i4hktOJS4zuddrj2HowGDTkM51jcWB9IbsKREWd2suACRYE84lzUv%2BXw7HGOuApFmyM%2BeZk45ghhN3GLQtnLM7UwpLxS0pcXtUD5JqqOUSIaSaMYyRWsO5Jh8eR50zUftzmEnPZZoRLyg8TIEa0x7hi0suAW1y12QaxWq5PKb00jSkciY0hsEl4rfbiucMKeSqMFgXAoDFo44UgqoU6bOpRSkWAg3OW4NBuNsRblVfL74pfGE5E9Wu0Fox7LqBCjTwXjcQERY3SDsCK3kHYaQ5Wj5JYKeYVbLmw%2FcuzBoGY53Ef42dBiQUxq89m3CmJDv%2BCMye%2BkHKECcYWXCfFuSOtqtKNGUesB5ukwi1sphFYqXIOktV4QkjnTTAbpkeO6HhnYjiSkgjpkd8ROLBbWkTAj%2B7tSqa5OLr%2BmFrhh45cO60BGI7%2BJmzfW1%2FW6TexpQLlY1FWgj5dzT55wAWZqUUyW5xcNb3RlGYyp1tOnL38qoMgwaFFGuAxm2zKxI37Y9JGkhAOrNAtaNmwiWSuYJf7mS7iWZ8T9MFtOWP1NXYlJsZCRUhXdby0nZYzBcGAtrFq6yrRlnv0Yl8a2%2FJ6oaYzW6TTGtBaGLwotQi0eD9sNX7G9Wlwx58JrymgxVDDYcEIslPi4lo5yXWITmS2MB%2Bm22%2FgglGqbI6snWepG3S4HW3MqD2ESW1jS4YVtOSOuioLud3WlrM2jccCsZeiYp%2BsXo1q694MxPiSpdAiWXUfIRyLBzsjElb0OR%2FywtfZ0TRGuP8IPCrGFRXWKqsTFEslu1vmmnyLEKBaUWha1IAAEs0G4RnFDslvdL%2BbieKb9OaiITqFeq4%2F9fWRjY8JPpr2dbf2uRTiZFZ4pjnkYzcJH%2FLaaLfRgLbxsTSHBjJg236xfeE8a6w8SO0iYjxHRh4tMgMpufXxgxW5vPRuT5An%2FYuW%2B9ksjH8IWhBBgxY1alGiuH35CZTHUvwVGLUx4Pvr7oN43HI4d252v%2FsWq3X5JtWig7dgAx93u8ULLNKuic3HHcd%2FIR%2BzDw%2BGpPmJ3ut2X5jud8TlmMJ8vVre2tgqwtZjNZpcfPPjP32%2FduvXHK1eu%2FBouvBs58hNc3%2F4q%2FqzQbLW6UKUmRosD4tFaWFj45%2BPHj3%2Fz6NGj3z98%2BPAP09PTv7148eIvz549e%2F7MmTM%2FhMlnX%2FmH9Gaz%2BaNer%2Feo0%2Bn43W439bYG33f%2F%2Fv3fXb58%2BYNLly79NBrnz7%2FG0C%2FzTwEXLlz4wblz597HeO8tjvfx7u%2B9qf%2FE%2BC%2FkEJZILUQcyQAAAABJRU5ErkJggg%3D%3D&label=CVPR (Coming Soon)&labelColor=367DBD" alt="CVPR" style="max-width: 100%;"></a>
  <a href="https://kuai-lab.github.io/cvpr20253dgsw/" rel="nofollow"><img alt="Static Badge" src="https://img.shields.io/badge/Website-gray?logo=Google%20chrome&label=Project Page&labelColor=darkorange&logoColor=white" alt="GitHub" style="max-width: 100%;"></a>
  <a href="https://github.com/kuai-lab/cvpr25_3D-GSW/"><img alt="Github" src="https://img.shields.io/badge/Code-gray?logo=Github&label=GitHub&labelColor=black" style="max-width: 100%;"></a>
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
    <pre><code>@misc{jang20253dgsw3dgaussiansplatting,
      title={3D-GSW: 3D Gaussian Splatting for Robust Watermarking}, 
      author={Youngdong Jang and Hyunje Park and Feng Yang and Heeju Ko and Euijin Choo and Sangpil Kim},
      year={2025},
      eprint={2409.13222},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2409.13222}, 
}</code></pre>
  </div>
</section>
