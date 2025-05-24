# predict.py

import os
import torch
import pandas as pd
import argparse
from datetime import datetime
from model.model import MultiKernelBiLSTM_Embedding
from utils.preprocessing import tokenize_and_pad, vocab_size

# -----------------------------
# Configuration
# -----------------------------
EMBEDDING_DIM = 21
HIDDEN_DIM = 64
MAX_SEQ_LEN = 150
DEVICE = torch.device("cpu")

# -----------------------------
# Map filename → kernel + tag
# -----------------------------
MODEL_MAP = {
    "ParaDeep_H.pt":  {"kernel": 9, "tag": "H"},
    "ParaDeep_L.pt":  {"kernel": 81, "tag": "L"},
    "ParaDeep_HL.pt": {"kernel": 21, "tag": "HL"}
}

# -----------------------------
# Create output folder if needed
# -----------------------------
def ensure_output_folder(folder="output"):
    if not os.path.exists(folder):
        os.makedirs(folder)

# -----------------------------
# Run predictions and filter out padding ('X') residues
# -----------------------------
def run_prediction(df, model, tag):
    results = []
    model.eval()
    with torch.no_grad():
        for seq_id, seq in zip(df["Seq_ID"], df["Seq_cap"]):
            x = tokenize_and_pad(seq, MAX_SEQ_LEN).unsqueeze(0).to(DEVICE)
            probs = model(x).squeeze(0).cpu().numpy()
            binary = (probs >= 0.5).astype(int).tolist()

            # Only use the original residues (not padded)
            for i, (res, pred) in enumerate(zip(seq, binary), start=1):
                results.append({
                    "Seq_ID": seq_id,
                    "Residue_Position": i,
                    "Residue": res,
                    f"{tag}_pred": pred
                })
    return pd.DataFrame(results)

# -----------------------------
# Main script
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ParaDeep paratope prediction")
    parser.add_argument("--model-path", required=True, help="Path to .pt model (e.g., saved_models/ParaDeep_HL.pt)")
    parser.add_argument("--input", required=True, help="CSV file with columns: Seq_ID, Seq_cap")
    args = parser.parse_args()

    # Resolve kernel size and tag
    model_filename = os.path.basename(args.model_path)
    model_info = MODEL_MAP.get(model_filename)
    if model_info is None:
        raise ValueError(f"Unknown model: {model_filename}. Please update MODEL_MAP.")

    kernel_size = model_info["kernel"]
    tag = model_info["tag"]

    # Timestamped output file name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_filename = f"{model_filename.replace('.pt', '')}_predictions_{timestamp}.csv"
    output_path = os.path.join("output", output_filename)

    # Load model
    model = MultiKernelBiLSTM_Embedding(
        vocab_size=vocab_size,
        emb_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        conv_kernel_size=kernel_size,
        seq_len=MAX_SEQ_LEN
    )
    model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
    model.to(DEVICE)

    # Load input and run prediction
    print(f"Input file: {args.input}")
    print(f"Model: {model_filename} (Kernel size: {kernel_size}, Tag: {tag})")
    df_input = pd.read_csv(args.input)
    df_output = run_prediction(df_input, model, tag)

    # Save results
    ensure_output_folder("output")
    df_output.to_csv(output_path, index=False)
    print(f"Predictions saved to: {output_path}")
