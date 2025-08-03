# paradeep.py
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import argparse
from datetime import datetime
from src.core import predict_paradeep

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ParaDeep: Chain-aware paratope prediction using BiLSTM-CNN"
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_output = f"output/predictions_{timestamp}.csv"
    default_plot_dir = f"output/plots_{timestamp}"

    parser.add_argument("--input", required=True, help="Input file (.csv, .fasta, .txt)")
    parser.add_argument("--modelH", required=True, help="Path to trained H-chain model (.pt)")
    parser.add_argument("--modelL", required=True, help="Path to trained L-chain model (.pt)")
    parser.add_argument("--output", default=default_output, help="Path to output .csv")
    parser.add_argument("--no-vis", action="store_true", help="Disable per-sequence visualization")

    args = parser.parse_args()

    predict_paradeep(
        input_path=args.input,
        model_H_path=args.modelH,
        model_L_path=args.modelL,
        kernel_H='Full',
        kernel_L='Full',
        output_path=args.output,
        visualize=not args.no_vis,
        plot_dir=default_plot_dir
    )
