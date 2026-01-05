"""
Runner per `scripts.point_predictors` che evita l'esecuzione di `scripts/__init__.py`
(che al momento genera SyntaxError). Usa solo CPU.

Esempio:
python run_point_predictors.py --files data/train.csv --model-out models/bdt_point.pkl --model-type bdt
python run_point_predictors.py --files data/train.csv --model-out models/mlp_point.pt --model-type mlp
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


def main():
    repo_root = Path(__file__).resolve().parent
    pkg_path = repo_root / "scripts"

    # Crea un modulo package "scripts" fittizio per abilitare import relativi
    scripts_pkg = types.ModuleType("scripts")
    scripts_pkg.__path__ = [str(pkg_path)]
    sys.modules.setdefault("scripts", scripts_pkg)

    # Carica scripts.point_predictors senza eseguire scripts/__init__.py
    target_path = pkg_path / "point_predictors.py"
    spec = importlib.util.spec_from_file_location("scripts.point_predictors", target_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Impossibile caricare {target_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["scripts.point_predictors"] = module
    spec.loader.exec_module(module)

    # Esegue il main del modulo (usa sys.argv)
    module.main()


if __name__ == "__main__":
    main()
