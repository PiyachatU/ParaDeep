from Bio import SeqIO
import pandas as pd
import sys

MAX_SEQ_LEN = 130


def load_csv(path):
    df = pd.read_csv(path)
    assert "Seq_ID" in df.columns and "Seq_cap" in df.columns, "Missing required columns"
    if "Chain_Type" not in df.columns:
        print("[Error] No 'Chain_Type' column found in CSV. Aborting.")
        sys.exit(1)
    else:
        df["Chain_Type"] = df["Chain_Type"].astype(str).str.upper()
        invalid = df[~df["Chain_Type"].isin(["H", "L"])]
        if not invalid.empty:
            print(f"[Warning] {len(invalid)} invalid 'Chain_Type' entries. Defaulting to 'H'.")
            df.loc[~df["Chain_Type"].isin(["H", "L"]), "Chain_Type"] = "H"

    long_seqs = df[df['Seq_cap'].apply(len) > MAX_SEQ_LEN]
    if not long_seqs.empty:
        print(f"[Error] {len(long_seqs)} sequences exceed {MAX_SEQ_LEN} residues. Aborting.")
        sys.exit(1)

    return df


def load_fasta(path):
    records = list(SeqIO.parse(path, "fasta"))
    seq_ids = []
    sequences = []
    chain_types = []

    for record in records:
        header = record.id
        parts = header.split("|")
        seq = str(record.seq)
        if len(seq) > MAX_SEQ_LEN:
            print(f"[Error] Sequence '{parts[0]}' exceeds {MAX_SEQ_LEN} residues. Aborting.")
            sys.exit(1)

        seq_ids.append(parts[0])
        sequences.append(seq)
        if len(parts) < 2:
            print(f"[Error] No chain type found in header for sequence '{parts[0]}'. Aborting.")
            sys.exit(1)
        chain_types.append(parts[1].strip().upper())

    return pd.DataFrame({
        "Seq_ID": seq_ids,
        "Seq_cap": sequences,
        "Chain_Type": chain_types
    })


def load_txt(path):
    seq_ids = []
    sequences = []
    chain_types = []

    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) == 3:
                seq_id, chain, seq = parts
            elif len(parts) == 2:
                print(f"[Error] No chain type found for sequence on line {i+1}. Aborting.")
                sys.exit(1)
            else:
                print(f"[Error] Invalid format on line {i+1}. Aborting.")
                sys.exit(1)

            if len(seq) > MAX_SEQ_LEN:
                print(f"[Error] Sequence '{seq_id}' exceeds {MAX_SEQ_LEN} residues. Aborting.")
                sys.exit(1)

            seq_ids.append(seq_id)
            sequences.append(seq)
            chain_types.append(chain.strip().upper())

    return pd.DataFrame({
        "Seq_ID": seq_ids,
        "Seq_cap": sequences,
        "Chain_Type": chain_types
    })


def load_sequences(path):
    if path.endswith(".csv"):
        return load_csv(path)
    elif path.endswith(".fasta") or path.endswith(".fa"):
        return load_fasta(path)
    elif path.endswith(".txt"):
        return load_txt(path)
    else:
        raise ValueError("Unsupported file format")
