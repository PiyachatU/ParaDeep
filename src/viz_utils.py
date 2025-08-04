import os
import matplotlib.pyplot as plt

def plot_binding_predictions(seq_id, residues, probs, threshold=0.5, output_dir="output/plots", chain_type="L"):
    os.makedirs(output_dir, exist_ok=True)

    # Use only actual residues and their predictions
    valid_len = len(residues)
    probs = probs[:valid_len]
    positions = list(range(1, valid_len + 1))

    # Indices above threshold
    above_thresh = [i for i, p in enumerate(probs) if p >= threshold]

    plt.figure(figsize=(max(12, valid_len * 0.15), 4))
    plt.bar(positions, probs, color='skyblue', label='Binding Probability')
    plt.axhline(y=threshold, color='red', linestyle='--', label=f'Threshold ({threshold})')

    for i in above_thresh:
        plt.plot(positions[i], probs[i], 'ro')  # red dot
        plt.text(positions[i], 1.05, residues[i], ha='center', va='bottom',
                 fontsize=9, fontweight='bold', rotation=0)

    plt.ylim(0, 1.15)
    plt.title(f"ParaDeep {chain_type}_Chain Prediction - {seq_id}")
    plt.xlabel("Residue Position")
    plt.ylabel("Binding Probability")
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(output_dir, f"{seq_id}_{chain_type}_Chain_prediction.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
