# URVFL: Undetectable Data Reconstruction Attack on Vertical Federated Learning

This is the official implementation of ["URVFL: Undetectable Data Reconstruction Attack on Vertical Federated Learning"](https://arxiv.org/abs/2404.19582) (NDSS 2025).

🎯 **Privacy attacks on VFL!**

We are thrilled to present URVFL, a privacy attack algorithm that fundamentally challenges the privacy assumptions in Vertical Federated Learning. Our work not only introduces the novel URVFL and sync attacks but also provides comprehensive implementations of multiple privacy attack baselines: AGN, PCAT, SDAR, FSHA, GRNA, and GIA.

🎓 **Exciting News**: Our paper has been accepted to NDSS 2025! 

🚀 **Current Status**: This repository currently showcases our implementation on CIFAR10 dataset. Stay tuned as we expand to more datasets.

⭐ If you find this work interesting or useful, please consider giving it a star!


## Overview
Launching effective malicious attacks in VFL presents unique challenges: 1) Firstly, given the distributed nature of clients' data features and models, each client rigorously guards its privacy and prohibits direct querying, complicating any attempts to steal data; 2) Existing malicious attacks alter the underlying VFL training task, and are hence easily detected by comparing the received gradients with the ones received in honest training.

 We propose URVFL, a novel attack strategy that evades current detection mechanisms. The key idea is to integrate a *discriminator with auxiliary classifier* that takes a full advantage of the label information and generates malicious gradients to the victim clients: on one hand, label information helps to better characterize embeddings of samples from distinct classes, yielding an improved reconstruction performance; on the other hand, computing malicious gradients with label information better mimics the honest training, making the malicious gradients indistinguishable from the honest ones, and the attack much more stealthy.
## Installation

1. Clone the repository
```bash
git clone https://github.com/duanyiyao/URVFL.git
cd URVFL
```
2. Create the virtual environment
```bash
conda create -n urvfl python=3.9
conda activate urvfl
```
3. Install the required packages:

```bash
pip install -r requirements.txt
```
## Repository Structure

```
├── cifar10/
│   ├── cifar_data_pre.py    # Data preprocessing
│   ├── config_cifar.json    # Configuration file
│   ├── detection_cifar.py   # Detection implementation
│   ├── urvfl_cifar.py       # URVFL implementation
│   └── ...                  # Other attack implementations
│
│── attack_module.py/        # URVFL attack and sync functions
│── baseline_module.py/      # Baseline Methods
│── grna_gia_img.py/         # GRNA and GIA in Image data
│── grna_gia_table.py/       # GRNA and GIA in tabular data
│── defenses.py/             # Detection methods
└── README.md
```

## Configuration


Before running the attacks, you need to:

1. Set up the dataset root path in cifar10/cifar_data_pre.py
2. Adjust hyperparameters in cifar10/config_cifar.json:

- save: Controls pretraining of encoder, shadow model, and decoder
    - Set True for initial training
    - Set False to load the pretrained models and skip to Step 2 (malicious gradient generation)
- sg_defense: Enable/disable SplitGuard detection
- gs_defense: Enable/disable Gradient Scrutinizer detection

Note: When sg_defense or gs_defense is True, the attack stops upon detection. When False, the system records detection scores throughout the attack process.


## Running the attacks
Basic usage:

```bash 
python cifar10/urvfl_cifar.py
```
For detection analysis:

```bash 
python cifar10/detection_cifar.py
```

This script can run AGN, FSHA, URVFL, sync, and normal training under SplitGuard and Gradient Scrutinizer detection.
