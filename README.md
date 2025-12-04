# Training for Identity, Inference for Controllability: A Unified Approach to Tuning-Free Face Personalization

Official Implementation of "Training for Identity, Inference for Controllability: A Unified Approach to Tuning-Free Face Personalization" by Lianyu Pang, Ji Zhou, Qiping Wang, Baoquan Zhao, Zhenguo Yang, Li Qing and Xudong Mao.

<a href="https://arxiv.org/abs/2512.03964"><img src="https://img.shields.io/badge/arXiv-2512.03964-b31b1b.svg" height=20.5></a>
<a href="https://huggingface.co/pangly/UniID" rel="nofollow"><img src="https://camo.githubusercontent.com/7e8c2f62ecbb426a33a94f034fe9075df62b44301eba0057872464c968299009/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f25463025394625413425393725323048756767696e67253230466163652d5370616365732d626c7565" data-canonical-src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue" style="max-width: 100%;"></a>

![teaser](assets/teaser.jpg)

## Introduction
We introduce **UniID**, a unified tuning-free framework that synergistically combines text embedding and adapter approaches while preserving both identity fidelity and text controllability. Our key insight is that when merging the two branches, they should mutually reinforce only identity information, while non-identity aspects such as scene composition are controlled by the original diffusion model's prior knowledge. Specifically, during training, we employ an identity-focused learning scheme that guides both the text embedding and adapter branches to capture exclusively identity-relevant features. At inference, we introduce a normalized rescaling strategy that recovers the text controllability of the original diffusion model in both branches while enabling their complementary identity signals to mutually reinforce each other. Through this strategic training-inference paradigm, UniID achieves superior identity fidelity while preserving the text controllability of the original model.

![uniid](assets/uniid.jpg)

## Release
- [x] 2025/12/04: We have released the code and model weights!

## Setup
### Setting Up the Environment
To set up the environment, run the following commands:
```bash
conda env create -f environment.yaml
conda activate uniid
```
### Setting Up Accelerate 
Initialize an [Accelerate](https://github.com/huggingface/accelerate/) environment with:
```bash
accelerate config
```

## Download
### Datasets
Our model was trained using the training sets of [CelebA-HQ](https://mmlab.ie.cuhk.edu.hk/projects/CelebA/CelebAMask_HQ.html), [FFHQ-Portrait](https://github.com/NVlabs/ffhq-dataset) and a filtered version of [FaceID-6M](https://github.com/ShuheSH/FaceID-6M/tree/master).

### Face Recognition Model
We use [Partial-FC](https://github.com/deepinsight/insightface/blob/master/recognition/partial_fc) as our face recognition model.

### Model Weights
For the fine-tuned model presented in our paper, please visit our HuggingFace repository:

**[🔗 HuggingFace Repository: UniID](https://huggingface.co/pangly/UniID)**

## Usage

### Running Inference
You can run the `bash_scripts/inference.sh` script to generate images. Before running the script, configure the following parameters in `inference.sh`:
+ Line **3**: `text_branch_checkpoint_dir`. Path to the directory containing the checkpoints of the text branch.
+ Line **4**: `adapter_branch_checkpoint_dir`. Path to the directory containing the checkpoints of the adapter branch.
+ Line **5**: `pretrained_face_recog_model_path`. Path to the face recognition model. Specify this path if you are using the model weights we provide. If you trained the model yourself, you can omit this argument; it will be loaded from the training configuration. 
+ Line **6**: `ref_image_path`. Path to the reference image.
+ Line **7**: `save_dir`. Directory where the generated images will be saved.
+ Line **8**: `prompt`. The input prompt. For convenience, you can also specify a path to a text file containing prompts (one per line) using `--prompt_file`. For example:
    ```
    A photo of a <class word>
    A <class word> eating bread in front of the Eiffel Tower
    A <class word> latte art
    ```
  The token `<class word>` in the prompts will be automatically replaced with `man` or `woman`. 

To run the inference script:
```bash
bash bash_scripts/inference.sh
```
+ The generated images will be saved to the directory `{save_dir}/{prompt}`

+ For a full list of parameters, please refer to `inference.py` and `inference.sh`.


### Training

You can run the `bash_scripts/train_uniid.sh` script to train your own model. Before running the script, configure the following parameters in `train_uniid.sh`:
+ Line **3**: `pretrained_face_recog_model_path`. Path to the face recognition model.
+ Line **4**: `data_path_file`. This file should list the paths to all training images, with one path per line. For example:
    ```
    dataset/CelebA-HQ/train/00000.jpg
    dataset/CelebA-HQ/train/00001.jpg
    ...
    dataset/ffhq/train/00001.jpg
    ```
    You can generate this file using the `write_dataset_to_file` function in `utils.py`.
+ Line **5**: `output_root`. Directory where the trained model will be saved.

To run the training script:
```bash
bash bash_scripts/train_uniid.sh
```
**Notes**:
+ All training arguments are listed in `train_uniid.sh` and are set to the default values used in the paper.
+ Please refer to the script for more details on each parameter.

## Acknowledgements
We would like to express our sincere gratitude to the creators and maintainers of [CelebA-HQ](https://mmlab.ie.cuhk.edu.hk/projects/CelebA/CelebAMask_HQ.html), [FFHQ-Portrait](https://github.com/NVlabs/ffhq-dataset), [FaceID-6M](https://github.com/ShuheSH/FaceID-6M/tree/master), [Diffusers](https://github.com/huggingface/diffusers/tree/main), [Celeb Basis](https://github.com/ygtxr1997/CelebBasis/tree/main) and [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter). Our project builds upon their foundational work and is made possible by their dedication to the open-source community.

## Citation
```
@article{UniID,
  title = {Training for Identity, Inference for Controllability: A Unified Approach to Tuning-Free Face Personalization},
  author = {Lianyu Pang, Ji Zhou, Qiping Wang, Baoquan Zhao, Zhenguo Yang, Qing Li, Xudong Mao},
  journal = {arXiv preprint arXiv:2512.03964},
  year = {2025}
}
```
