# utils/preprocessing.py

import torch

# -----------------------------
# Configuration
# -----------------------------
MAX_SEQ_LEN = 150  # Adjust if your model uses a different length
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
aa_to_index = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
aa_to_index["X"] = len(AMINO_ACIDS)  # Unknown amino acid

vocab_size = len(aa_to_index)

# -----------------------------
# Tokenization + Padding
# -----------------------------
def tokenize_and_pad(seq, max_len=MAX_SEQ_LEN):
    """
    Convert amino acid sequence to tensor of indices with fixed length.
    Unknown or padding residues are assigned to index of 'X'.
    """
    tokens = [aa_to_index.get(aa, aa_to_index["X"]) for aa in seq]
    if len(tokens) < max_len:
        tokens += [aa_to_index["X"]] * (max_len - len(tokens))  # Pad with 'X'
    else:
        tokens = tokens[:max_len]  # Truncate
    return torch.tensor(tokens, dtype=torch.long)
