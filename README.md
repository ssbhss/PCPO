# PCPO: Prototype-based Cross-domain Prompt Optimization

This repository contains the official codebase of the paper **"PCPO: Prototype-based Cross-domain Prompt Optimization"**.

PCPO is designed to adapt vision-language models like CLIP to target domains under cross-domain settings through prototype-based cross-domain prompt optimization.

---

## 🚀 Repository Structure

The isolated PCPO codebase is structured as follows:

```
PCPO/
├── Dassl.pytorch/             # Core library (submodule codebase)
├── config/                    # Configs for standalone PCPO script (.json & category files)
├── configs/                   # Configs for DASSL framework (.yaml files)
│   ├── datasets/              # Dataset configs (office31, office_home, visda17, domainnet)
│   └── trainers/PCPO/         # PCPO trainer configs (pcpo.yaml)
├── datasets/                  # DASSL dataset registration scripts
├── trainers/                  # DASSL trainers implementation
│   ├── pcpo.py                # Main PCPO Trainer class
│   └── zsclip.py              # ZeroshotCLIP helper classes
├── tools/pcpo/                # Training and simple source training tools
├── scripts/pcpo/              # Runner scripts for VisDA17, Office31, OfficeHome, miniDomainNet
├── network/                   # ResNet backbone definition for source models
├── PCPO.py                    # Standalone evaluation & prototype optimization script
├── classifier.py              # ResNet model classifier definitions
├── run_pcpo.sh                # Main helper bash runner
├── DATA_STRUCTURE.md          # Dataset directory structure details
├── Dataset_Download_Links.md  # Links to download datasets
└── requirements.txt           # Project dependencies
```

---

## 🛠️ Installation

This codebase is built on top of [Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch). Follow these steps to set up the environment:

1. **Set up the Conda environment & install Dassl:**
   Ensure you have PyTorch installed (preferably with CUDA support). Then install `dassl` following the instructions in `Dassl.pytorch`:
   ```bash
   cd Dassl.pytorch
   pip install -r requirements.txt
   python setup.py develop
   cd ..
   ```

2. **Install project requirements:**
   ```bash
   pip install -r requirements.txt
   ```
   *Note: Make sure `clip`, `torch-kmeans` (or `scikit-learn` as fallback), `nvidia-dali` are installed if you plan on using DALI acceleration.*

---

## 📊 Datasets & Data Preparation

PCPO supports 4 main domain adaptation datasets:
* **Office31** (3 domains: amazon, dslr, webcam)
* **OfficeHome** (4 domains: art, clipart, product, real_world)
* **VisDA17** (2 domains: train, test)
* **miniDomainNet** (4 domains: clipart, painting, real, sketch)

Please refer to:
* [DATA_STRUCTURE.md](DATA_STRUCTURE.md) for how to lay out the datasets in the `data/` folder.
* [Dataset_Download_Links.md](Dataset_Download_Links.md) for quick download links and instructions.

---

## 🏃 How to Run PCPO

We provide a convenient bash script `run_pcpo.sh` to run the experiments.

### 1. Train Source Models
Before running domain adaptation experiments, you need to train the models on the source domain:

```bash
# Train amazon source model on Office31 dataset
./run_pcpo.sh train-source office31 amazon

# Train all source models across all datasets
./run_pcpo.sh train-all-source
```
Pre-trained source models will be saved in `pretrained_models/`.

### 2. Run Domain Adaptation Experiments
Use the `single` command to run PCPO domain adaptation:

```bash
# Run Office31: Amazon -> DSLR
./run_pcpo.sh single office31 amazon dslr

# Run OfficeHome: Art -> Clipart
./run_pcpo.sh single office_home art clipart

# Run VisDA17: Train (synthetic) -> Test (real)
./run_pcpo.sh single visda17 train test
```

### 3. Run Full Datasets & Summarize Results
```bash
# Run all adaptation scenarios for Office31
./run_pcpo.sh dataset office31

# Run all adaptation scenarios across all datasets
./run_pcpo.sh all

# Summarize results for a specific dataset
./run_pcpo.sh summarize office31

# Summarize all results
./run_pcpo.sh summarize
```

---

## 📝 Standalone Optimization Execution

If you wish to run the standalone prototype optimization script directly without using DASSL:
```bash
python PCPO.py --domain visda17 --source_domain train --target_domain test
```
*(Make sure to adjust the dataset paths and pre-trained checkpoint paths in the configuration json located in the `config/` directory)*
