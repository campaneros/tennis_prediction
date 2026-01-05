"""
Runner gender-aware per `scripts.point_predictors` senza toccare i file esistenti.
Filtra i match_id: uomini 1000-1999, donne 2000-3000 (pattern ...-<id>).

Esempi:
python run_point_predictors_gender.py --files data/2019-wimbledon-points.csv --model-out models/bdt_male.pkl --model-type bdt --gender male
python run_point_predictors_gender.py --files data/2019-wimbledon-points.csv --model-out models/mlp_female.pt --model-type mlp --gender female
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import re
import sys
import types
from pathlib import Path
from typing import Iterable


def _load_point_module():
    """Carica scripts.point_predictors evitando scripts/__init__.py."""
    repo_root = Path(__file__).resolve().parent
    pkg_path = repo_root / "scripts"

    scripts_pkg = types.ModuleType("scripts")
    scripts_pkg.__path__ = [str(pkg_path)]
    sys.modules.setdefault("scripts", scripts_pkg)

    target_path = pkg_path / "point_predictors.py"
    spec = importlib.util.spec_from_file_location("scripts.point_predictors", target_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Impossibile caricare {target_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["scripts.point_predictors"] = module
    spec.loader.exec_module(module)
    return module


def parse_args(argv: Iterable[str] | None = None):
    p = argparse.ArgumentParser(description="Train CPU BDT/MLP per genere (match_id ranges).")
    p.add_argument("--files", nargs="+", required=True, help="CSV dei punti (pattern ...-<id>).")
    p.add_argument("--model-out", required=True, help="Path salvataggio modello.")
    p.add_argument("--model-type", choices=["bdt", "mlp"], default="bdt")
    p.add_argument("--gender", choices=["male", "female", "both"], default="both", help="Filtro match_id.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def _filter_by_gender(df, gender: str):
    """Estrae suffisso numerico da match_id e filtra per genere."""
    match_ids = df["match_id"].astype(str)
    num = match_ids.apply(lambda s: int(re.search(r"-(\d+)$", s).group(1)) if re.search(r"-(\d+)$", s) else None)
    df = df[num.notnull()].copy()
    num = num[num.notnull()].astype(int)

    if gender == "male":
        mask = (num >= 1000) & (num < 2000)
    elif gender == "female":
        mask = (num >= 2000) & (num <= 3000)
    else:
        mask = num == num  # tutti

    filtered = df[mask.values]
    if filtered.empty:
        raise ValueError(f"Nessun match per genere={gender}. Controlla i range id (uomini 1000-1999, donne 2000-3000).")
    return filtered


def main(argv: Iterable[str] | None = None):
    args = parse_args(argv)
    mod = _load_point_module()

    df = mod.prepare_dataframe(args.files)
    df = _filter_by_gender(df, args.gender)
    print(f"[gender-filter] genere={args.gender}, punti={len(df)}, partite uniche={df['match_id'].nunique()}")

    bundle = mod.build_enhanced_dataset(df)
    os.makedirs(os.path.dirname(args.model_out) or ".", exist_ok=True)

    if args.model_type == "bdt":
        model, metrics = mod.train_bdt(bundle, random_state=args.seed)
        mod.joblib.dump({"model": model, "features": mod.MATCH_FEATURE_COLUMNS + mod.EXTRA_FEATURE_COLUMNS}, args.model_out)
        print(f"[bdt] saved to {args.model_out}")
    else:
        model, scaler, metrics = mod.train_mlp(bundle, random_state=args.seed)
        mod.torch.save(
            {"state_dict": model.state_dict(), "scaler": scaler, "features": mod.MATCH_FEATURE_COLUMNS + mod.EXTRA_FEATURE_COLUMNS},
            args.model_out,
        )
        print(f"[mlp] saved to {args.model_out}")

    print("[metrics]", metrics)


if __name__ == "__main__":
    main()
