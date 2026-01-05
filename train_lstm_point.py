#!/usr/bin/env python3
"""
Allena un LSTM a due strati punto-per-punto per predire la probabilità che P1
vinca il punto (PointWinner==1). Esporta un CSV con una riga per punto.
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
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit("PyTorch non installato. Installa torch per usare questo script.") from exc


def _load_sequences(path: str) -> Tuple[List[Dict], List[str]]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} non trovato. Esegui prima prepare_wimbledon_point_sequences.py"
        )
    with open(path, "rb") as f:
        payload = pickle.load(f)
    sequences = payload["sequences"]
    sequences = [s for s in sequences if len(s.get("features", [])) > 0]
    return sequences, payload["feature_names"]


class PointDataset(Dataset):
    def __init__(self, sequences: List[Dict]):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx: int):
        item = self.sequences[idx]
        feats = torch.tensor(item["features"], dtype=torch.float32)
        labels = torch.tensor(item["labels"], dtype=torch.float32)
        meta = item["meta"]
        return feats, labels, meta, item["match_id"]


def collate_batch(batch):
    feats_list, labels_list, meta_list, match_ids = zip(*batch)
    lengths = torch.tensor([x.shape[0] for x in feats_list], dtype=torch.long)
    feats_padded = pad_sequence(feats_list, batch_first=True)
    labels_padded = pad_sequence(labels_list, batch_first=True)
    return feats_padded, labels_padded, lengths, meta_list, match_ids


class PointLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, padded_seq, lengths):
        packed = pack_padded_sequence(padded_seq, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)
        logits = self.head(out).squeeze(-1)  # shape (batch, seq_len)
        return logits


def train_model(
    sequences: List[Dict],
    feature_names: List[str],
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 1e-3,
    device: str = "cpu",
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.1,
    balanced: bool = False,
):
    dataset = PointDataset(sequences)
    val_size = max(1, int(0.1 * len(dataset)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

    model = PointLSTM(
        input_size=len(feature_names),
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    # Calcolo pos_weight se richiesto
    pos_weight = None
    if balanced:
        all_labels = []
        for seq in sequences:
            all_labels.append(seq["labels"])
        labels_concat = torch.tensor(np.concatenate(all_labels), dtype=torch.float32)
        pos = labels_concat.sum().item()
        neg = len(labels_concat) - pos
        if pos > 0:
            pos_weight = torch.tensor([neg / pos], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for feats_padded, labels_padded, lengths, _, _ in train_loader:
            feats_padded, labels_padded, lengths = feats_padded.to(device), labels_padded.to(device), lengths.to(device)
            optimizer.zero_grad()
            logits = model(feats_padded, lengths)
            # maschera per lunghezze
            mask = torch.arange(labels_padded.size(1), device=device).unsqueeze(0) < lengths.unsqueeze(1)
            loss = criterion(logits[mask], labels_padded[mask])
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for feats_padded, labels_padded, lengths, _, _ in val_loader:
                feats_padded, labels_padded, lengths = feats_padded.to(device), labels_padded.to(device), lengths.to(device)
                logits = model(feats_padded, lengths)
                mask = torch.arange(labels_padded.size(1), device=device).unsqueeze(0) < lengths.unsqueeze(1)
                loss = criterion(logits[mask], labels_padded[mask])
                val_loss += loss.item()
                preds = (torch.sigmoid(logits[mask]) >= 0.5).long()
                val_correct += (preds.view(-1) == labels_padded[mask].long().view(-1)).sum().item()
                val_total += labels_padded[mask].numel()
        print(
            f"Epoch {epoch}/{epochs} | train_loss={train_loss/len(train_loader):.4f} "
            f"| val_loss={val_loss/len(val_loader):.4f} | val_acc={val_correct/val_total:.3f}"
        )
    return model


def export_point_probs(model, sequences: List[Dict], feature_names: List[str], out_csv: str, device: str):
    rows = []
    model.eval()
    with torch.no_grad():
        for seq in sequences:
            feats = torch.tensor(seq["features"], dtype=torch.float32).unsqueeze(0).to(device)
            lengths = torch.tensor([seq["features"].shape[0]], dtype=torch.long).to(device)
            logits = model(feats, lengths).squeeze(0)  # shape (seq_len,)
            probs = torch.sigmoid(logits).cpu().numpy()
            for (match_id, set_no, game_no, point_no, server), prob, label in zip(seq["meta"], probs, seq["labels"]):
                rows.append(
                    {
                        "match_id": match_id,
                        "SetNo": set_no,
                        "GameNo": game_no,
                        "PointNumber": point_no,
                        "PointServer": server,
                        "PointWinner": int(label == 1),
                        "p1_point_prob": float(prob),
                    }
                )
    import pandas as pd

    df = pd.DataFrame(rows)
    df.sort_values(["match_id", "SetNo", "GameNo", "PointNumber"], inplace=True)
    df.to_csv(out_csv, index=False)
    print(f"Salvato CSV punto-per-punto: {out_csv} (righe: {len(df)})")


def main():
    parser = argparse.ArgumentParser(description="Train LSTM punto-per-punto (P1 vince il punto).")
    parser.add_argument("--data", default="data/wimbledon_point_sequences.pkl", help="Pickle generato da prepare_wimbledon_point_sequences.py")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--sample-frac", type=float, default=1.0, help="Frazione di sequenze per run veloce.")
    parser.add_argument("--balanced", action="store_true", help="Usa pos_weight per bilanciare classi.")
    parser.add_argument("--export-csv", default="data/lstm_point_probs.csv", help="CSV output punto-per-punto.")
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

    os.makedirs("models", exist_ok=True)
    torch.save({"model_state": model.state_dict(), "feature_names": feature_names}, "models/lstm_point_model.pt")
    print("Modello salvato in models/lstm_point_model.pt")

    if args.export_csv:
        export_point_probs(model, sequences, feature_names, args.export_csv, device=device)


if __name__ == "__main__":
    main()
