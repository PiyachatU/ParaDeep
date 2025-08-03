<img src="https://github.com/user-attachments/assets/eb2f6678-281d-4ec3-b9c3-d8325fb787d3" alt="Alt Text" style="width:29%; height:auto;">

# ParaDeep: Sequence-Based Paratope Prediction with BiLSTM-CNN

**ParaDeep** is a chain-aware, sequence-only deep learning model for predicting **paratope residues** (antigen-binding sites) directly from antibody sequences.  
It uses a lightweight **BiLSTM-CNN architecture** and supports both **one-hot** and **embedding** encodings for **L** and **H** chains respectively.

It requires only amino acid sequences — no structural input or large pretrained models. Predictions are per-residue, human-readable, and designed for practical use in early-stage antibody discovery and analysis.

---

## Installation & Setup Instructions

Follow these steps to install and run ParaDeep on your machine:


### 1. Clone the Repository
        git clone https://github.com/PiyachatU/ParaDeep.git
        cd ParaDeep
### 2. Set Up a Python Environment (Recommended)
### Option A: Using Conda:
        conda create -n paradeep python=3.9
        conda activate paradeep
### Option B: Using venv:
        python -m venv paradeep_env
        source paradeep_env/bin/activate   # macOS/Linux
        paradeep_env\Scripts\activate      # Windows
### 3. Install Python Dependencies
### From the project root folder, run:
    cd paradeep
    pip install -r requirements.txt
### 4. Ready to Predict
    python paradeep.py \
    --input data/sample_input.csv \
    --modelH models/Best_Model_H.pt \
    --modelL models/Best_Model_L.pt
### 5. View usage instructions at any time
    python paradeep.py --help

Expected output:

```
usage: paradeep.py [-h] --input INPUT --modelH MODELH --modelL MODELL
                   [--output OUTPUT] [--no-vis]

ParaDeep: Chain-aware paratope prediction using BiLSTM-CNN

options:
  -h, --help       show this help message and exit
  --input INPUT    Input file (.csv, .fasta, .txt)
  --modelH MODELH  Path to trained H-chain model (.pt)
  --modelL MODELL  Path to trained L-chain model (.pt)
  --output OUTPUT  Path to output .csv
  --no-vis         Disable per-sequence visualization
```

## Try ParaDeep in Colab

Click below to open the notebook in Google Colab and run ParaDeep:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](
https://colab.research.google.com/github/PiyachatU/ParaDeep/blob/main/ParaDeep_Colab.ipynb)


