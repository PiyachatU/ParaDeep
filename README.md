# ParaDeep: Sequence-Based Paratope Prediction with BiLSTM-CNN

**ParaDeep** is a lightweight deep learning model for predicting paratope residues (antigen-binding sites) from antibody sequences. It uses a BiLSTM-CNN architecture with learnable embeddings and requires only amino acid sequences — no structural input or large pretrained models.

The framework includes pretrained models for heavy (H), light (L), and combined (HL) chains. Predictions are per-residue, human-readable, and designed for practical use in early-stage antibody discovery and analysis.

---

## Why Is ParaDeep Interpretable?

- Chain-aware modeling (H, L, HL)
- Task-specific learnable embeddings
- Transparent BiLSTM + CNN architecture
- Per-residue binary output (0 = non-binding, 1 = binding)
- Input/output in simple, domain-friendly CSV format

---

## Installation & Setup Instructions

Follow these steps to install and run ParaDeep on your machine:


### 1. Clone the Repository
   If you're using Git:
   
        ```bash
        git clone https://github.com/YOUR_USERNAME/ParaDeep.git
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
    cd ParaDeep
    pip install -r requirements.txt

### 4. Ready to Predict!
    python predict.py \
    --model-path saved_models/ParaDeep_HL.pt \
    --input data/sample_input.csv

## ParaDeep in Google Colab
> 📝 To use ParaDeep in Google Colab, please remember to  
> **File → Save a copy in Drive** before running any cells.

🔗 [Click here to open ParaDeep in Colab](https://colab.research.google.com/github/PiyachatU/ParaDeep/blob/main/ParaDeep_Colab.ipynb)

