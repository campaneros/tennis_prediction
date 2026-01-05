#!/usr/bin/env python3
"""
Allena un LSTM a due strati per stimare la probabilità di tenere il servizio
su ogni game, usando le sequenze generate da prepare_wimbledon_hold_sequences.py.

Output opzionale: un CSV con le probabilità per game da usare come feature
in un BDT successivo (es. per predire il vincitore del punto o del match).
"""

import argparse
import os
import pickle
from typing import Dict, List, Tuple

import numpy as np

try:
    import torch
    from torch import nn
    from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
    from torch.utils.data import DataLoader, Dataset, random_split
except ModuleNotFoundError as exc:  # pragma: no cover - handled at runtime
    raise SystemExit(
        "PyTorch non è installato. Installa torch per eseguire questo script "
        "(es. pip install torch)."
    ) from exc


def _load_sequences(path: str) -> Tuple[List[Dict], List[str]]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} non trovato. Esegui prima prepare_wimbledon_hold_sequences.py"
        )
    with open(path, "rb") as f:
        payload = pickle.load(f)
    sequences = payload["sequences"]
    # Scarta sequenze vuote (può capitare se un game aveva dati inconsistenti)
    sequences = [s for s in sequences if len(s.get("features", [])) > 0]
    return sequences, payload["feature_names"]


class GameHoldDataset(Dataset):
    def __init__(self, sequences: List[Dict]):
        self.sequences = sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        item = self.sequences[idx]
        feats = torch.tensor(item["features"], dtype=torch.float32)
        label = torch.tensor(item["label_hold"], dtype=torch.float32)
        meta = (
            item["match_id"],
            item["set_no"],
            item["game_no"],
            item["server_start"],
        )
        return feats, label, meta


def collate_batch(batch):
    sequences, labels, metas = zip(*batch)
    lengths = torch.tensor([seq.shape[0] for seq in sequences], dtype=torch.long)
    padded = pad_sequence(sequences, batch_first=True)  # pad con 0
    labels = torch.stack(labels)
    return padded, lengths, labels, metas


class HoldLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, padded_seq, lengths):
        packed = pack_padded_sequence(padded_seq, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)
        # Prende l'ultimo stato valido per ogni sequenza
        idx = (lengths - 1).view(-1, 1, 1).expand(out.size(0), 1, out.size(2))
        last_hidden = out.gather(1, idx).squeeze(1)
        logits = self.head(last_hidden).squeeze(1)
        return logits


def train_model(
    sequences: List[Dict],
    feature_names: List[str],
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = "cpu",
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.1,
    balanced: bool = False,
):
    dataset = GameHoldDataset(sequences)
    val_size = max(1, int(0.1 * len(dataset)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

    model = HoldLSTM(
        input_size=len(feature_names),
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)
    # Calcola pos_weight se richiesto (per riequilibrare hold vs break)
    pos_weight = None
    if balanced:
        labels_all = torch.tensor([item["label_hold"] for item in sequences], dtype=torch.float32)
        pos = labels_all.sum().item()
        neg = len(labels_all) - pos
        if pos > 0:
            pos_weight = torch.tensor([neg / pos], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for padded, lengths, labels, _ in train_loader:
            padded, lengths, labels = padded.to(device), lengths.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(padded, lengths)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * padded.size(0)

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for padded, lengths, labels, _ in val_loader:
                padded, lengths, labels = padded.to(device), lengths.to(device), labels.to(device)
                logits = model(padded, lengths)
                loss = criterion(logits, labels)
                val_loss += loss.item() * padded.size(0)
                preds = (torch.sigmoid(logits) >= 0.5).long()
                val_correct += (preds == labels.long()).sum().item()
                val_total += labels.size(0)

        print(
            f"Epoch {epoch}/{epochs} | train_loss={train_loss/len(train_ds):.4f} "
            f"| val_loss={val_loss/len(val_ds):.4f} | val_acc={val_correct/val_total:.3f}"
        )

    return model


def export_probabilities(model, sequences: List[Dict], feature_names: List[str], out_csv: str, device: str):
    rows = []
    model.eval()
    with torch.no_grad():
        for item in sequences:
            feats = torch.tensor(item["features"], dtype=torch.float32).unsqueeze(0).to(device)
            lengths = torch.tensor([item["features"].shape[0]], dtype=torch.long).to(device)
            logit = model(feats, lengths)
            prob = torch.sigmoid(logit).item()
            rows.append(
                {
                    "match_id": item["match_id"],
                    "set_no": item["set_no"],
                    "game_no": item["game_no"],
                    "server_start": item["server_start"],
                    "hold_label": item["label_hold"],
                    "hold_prob": prob,
                }
            )
    import pandas as pd

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"Salvato CSV con probabilità per game: {out_csv} (righe: {len(df)})")


def main():
    parser = argparse.ArgumentParser(description="Train 2-layer LSTM per probabilità di hold.")
    parser.add_argument("--data", default="data/wimbledon_hold_sequences.pkl", help="Percorso sequenze preparate.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--export-csv", default="", help="Se impostato, salva le probabilità per game in questo CSV.")
    parser.add_argument("--hidden-size", type=int, default=64, help="Dimensione hidden LSTM.")
    parser.add_argument("--num-layers", type=int, default=2, help="Numero layer LSTM.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout tra layer LSTM.")
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=1.0,
        help="Frazione di sequenze da usare per training (per run veloce, es. 0.1).",
    )
    parser.add_argument(
        "--balanced",
        action="store_true",
        help="Usa pos_weight per bilanciare hold vs break (utile se il modello collassa sul prior).",
    )
    args = parser.parse_args()

    sequences, feature_names = _load_sequences(args.data)
    if args.sample_frac < 1.0:
        n = max(1, int(len(sequences) * args.sample_frac))
        sequences = sequences[:n]
        print(f"Sample frac {args.sample_frac}: uso {n} sequenze")

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    print(f"Sequenze: {len(sequences)} | Feature: {len(feature_names)} | Device: {device}")

    model = train_model(
        sequences,
        feature_names,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        balanced=args.balanced,
    )

    model_path = "models/lstm_hold_model.pt"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(
        {"model_state": model.state_dict(), "feature_names": feature_names},
        model_path,
    )
    print(f"Modello salvato in {model_path}")

    if args.export_csv:
        export_probabilities(model, sequences, feature_names, args.export_csv, device=device)


if __name__ == "__main__":
    main()
