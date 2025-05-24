# ParaDeep: Sequence-Based Paratope Prediction with BiLSTM-CNN

**ParaDeep** is a lightweight deep learning model for predicting paratope residues (antigen-binding sites) from antibody sequences. It uses a BiLSTM-CNN architecture with learnable embeddings and requires only amino acid sequences — no structural input or large pretrained models.

The framework includes pretrained models for heavy (H), light (L), and combined (HL) chains. Predictions are per-residue, human-readable, and designed for practical use in early-stage antibody discovery and analysis.

---

## Why Is ParaDeep Interpretable?

- ✅ Chain-aware modeling (H, L, HL)
- ✅ Task-specific learnable embeddings
- ✅ Transparent BiLSTM + CNN architecture
- ✅ Per-residue binary output (0 = non-binding, 1 = binding)
- ✅ Input/output in simple, domain-friendly CSV format

---


ParaDeep/
├── predict.py # Main inference script
├── model/
│ └── model.py # BiLSTM-CNN model definition
├── utils/
│ └── preprocessing.py # Sequence tokenizer + encoder
├── saved_models/
│ ├── ParaDeep_H.pt
│ ├── ParaDeep_L.pt
│ └── ParaDeep_HL.pt
├── data/
│ └── sample_input.csv # Example antibody sequences
├── output/
│ └── (auto-saved predictions)
├── requirements.txt
└── README.md
