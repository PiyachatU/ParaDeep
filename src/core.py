# paradeep.py

import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

from src.model import MultiKernelBiLSTM_Embedding, MultiKernelBiLSTM_OneHot
from src.io_utils import load_sequences
from src.viz_utils import plot_binding_predictions

# -----------------------------
# Constants
# -----------------------------
MAX_SEQ_LEN = 130
BATCH_SIZE = 1
EMBED_DIM = 21
HIDDEN_DIM = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: idx for idx, aa in enumerate(AMINO_ACIDS)}
AA_TO_IDX["X"] = len(AA_TO_IDX)  # unknown AA = 20

# -----------------------------
# Utility Functions
# -----------------------------
def truncate_or_pad(seq):
    if len(seq) > MAX_SEQ_LEN:
        return seq[:MAX_SEQ_LEN]
    return seq + "X" * (MAX_SEQ_LEN - len(seq))

def one_hot_encode(seq):
    vecs = []
    for aa in seq:
        vec = [0] * len(AA_TO_IDX)
        idx = AA_TO_IDX.get(aa, AA_TO_IDX["X"])
        vec[idx] = 1
        vecs.append(vec)
    return torch.tensor(vecs, dtype=torch.float32)

def encode_sequence(seq):
    idxs = [AA_TO_IDX.get(aa, AA_TO_IDX["X"]) for aa in seq]
    return torch.tensor(idxs, dtype=torch.long)

def ensure_output_folder(path="output"):
    os.makedirs(path, exist_ok=True)

# -----------------------------
# Per-sequence Prediction
# -----------------------------
def predict_on_sequences(df, model, encoding="embedding", tag="H", visualize=True, plot_dir="output/plots"):
    model.eval()
    results = []

    for i, row in tqdm(df.iterrows(), total=len(df), desc=f"Predicting {tag}"):
        seq_id = row["Seq_ID"]
        seq_raw = row["Seq_cap"]
        seq = truncate_or_pad(seq_raw)

        if encoding == "embedding":
            x = encode_sequence(seq).unsqueeze(0).to(DEVICE)
        else:
            x = one_hot_encode(seq).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            probs = model(x).squeeze(0).cpu().numpy()
            preds = (probs >= 0.5).astype(int).tolist()

        valid_len = min(len(seq_raw), MAX_SEQ_LEN)

        for pos, (res, pred) in enumerate(zip(seq[:valid_len], preds[:valid_len]), start=1):
            results.append({
                "Seq_ID": seq_id,
                "Residue_Position": pos,
                "Residue": res,
                f"{tag}_Prediction": pred,
                f"{tag}_Probability": round(probs[pos - 1], 4)
            })

        # Save visualization
        if visualize:
            plot_binding_predictions(
                seq_id=seq_id,
                residues=list(seq[:valid_len]),
                probs=probs[:valid_len],
                threshold=0.5,
                output_dir=plot_dir,
                chain_type=tag
            )

    return pd.DataFrame(results)

# -----------------------------
# Main Dispatcher
# -----------------------------
def predict_paradeep(input_path,
                     model_H_path,
                     model_L_path,
                     kernel_H='Full',
                     kernel_L='Full',
                     output_path="output/predictions.csv",
                     visualize=True,
                     plot_dir="output/plots"):

    # Load input
    df = load_sequences(input_path)
    ensure_output_folder(os.path.dirname(output_path))

    if "Chain_Type" not in df.columns:
        print("[Warning] No 'Chain_Type' column found in the input file.")
        print("[Warning] All sequences will be treated as heavy chains (H).")
        print(f"[Warning] The predictive model that will be used: {os.path.basename(model_H_path)}")
        df["Chain_Type"] = "H"
    else:
        df["Chain_Type"] = df["Chain_Type"].astype(str).str.upper()
        invalid = df[~df["Chain_Type"].isin(["H", "L"])]
        if not invalid.empty:
            print(f"[Warning] {len(invalid)} invalid 'Chain_Type' entries found. Defaulting them to 'H'.")
            df.loc[~df["Chain_Type"].isin(["H", "L"]), "Chain_Type"] = "H"

    df_H = df[df["Chain_Type"] == "H"].reset_index(drop=True)
    df_L = df[df["Chain_Type"] == "L"].reset_index(drop=True)

    print(f"\nLoaded {len(df)} sequences.")
    print(f"  - Heavy chains (H): {len(df_H)}")
    print(f"  - Light chains (L): {len(df_L)}")

    results = []

    # --- Heavy Chain Prediction ---
    if not df_H.empty:
        print(f"\nLoading H model: {model_H_path}")
        model_H = MultiKernelBiLSTM_Embedding(
            vocab_size=21,  # match pretrained H-chain model
            emb_dim=EMBED_DIM,
            hidden_dim=HIDDEN_DIM,
            conv_kernel_size=kernel_H,
            seq_len=MAX_SEQ_LEN
        )
        model_H.load_state_dict(torch.load(model_H_path, map_location=DEVICE))
        model_H.to(DEVICE)

        df_H_pred = predict_on_sequences(df_H, model_H, encoding="embedding", tag="H",
                                  visualize=visualize, plot_dir=plot_dir)
        results.append(df_H_pred)
    else:
        print("No heavy chains found.")

    # --- Light Chain Prediction ---
    if not df_L.empty:
        print(f"\nLoading L model: {model_L_path}")
        model_L = MultiKernelBiLSTM_OneHot(
            input_dim=len(AA_TO_IDX),
            hidden_dim=HIDDEN_DIM,
            conv_kernel_size=kernel_L,
            seq_len=MAX_SEQ_LEN
        )
        model_L.load_state_dict(torch.load(model_L_path, map_location=DEVICE))
        model_L.to(DEVICE)

        df_L_pred = predict_on_sequences(df_L, model_L, encoding="onehot", tag="L",
                                  visualize=visualize, plot_dir=plot_dir)
        results.append(df_L_pred)
    else:
        print("No light chains found.")

    # --- Combine and Save ---
    if results:
        df_all = pd.concat(results, axis=0).sort_values(by=["Seq_ID", "Residue_Position"])
        df_all.to_csv(output_path, index=False)
        print(f"\nPrediction complete. Results saved to:\n{output_path}")
    else:
        print("No predictions were made. Check your input file.")
