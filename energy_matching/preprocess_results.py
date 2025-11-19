from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Extract top-k% positive samples from results.json.")
    parser.add_argument("--results", type=str, default="results.json")
    parser.add_argument("--output", type=str, default="positive_samples.json")
    parser.add_argument("--top-percent", type=float, default=0.05)
    args = parser.parse_args()

    top_ratio = max(1e-6, min(1.0, args.top_percent))

    with open(args.results, "r", encoding="utf-8") as f:
        data = json.load(f)

    selected = []
    for entry in data["results"]:
        scenarios = sorted(entry["scenarios"], key=lambda s: s["profit_eur"], reverse=True)
        top_n = max(1, math.ceil(top_ratio * len(scenarios)))
        filtered = [sc for sc in scenarios[:top_n] if sc["profit_eur"] > 0.0]
        for rank, s in enumerate(filtered):
            selected.append(
                {
                    "demand_index": entry["index"],
                    "demand_profile": entry["demand_profile"],
                    "count_feasible": entry["count_feasible"],
                    "selected_rank": rank + 1,
                    "selected_from": len(scenarios),
                    "m_mw": s["m_mw"],
                    "n_r": s["n_r"],
                    "n_storage_fixed": s["n_storage_fixed"],
                    "profit_eur": s["profit_eur"],
                    "prod": s["prod"],
                    "soc": s["soc"],
                    "charge": s["charge"],
                    "discharge": s["discharge"],
                }
            )

    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "source_file": args.results,
        "description": f"Top {top_ratio*100:.1f}% positive profit configurations per demand profile",
        "count_profiles": len(data["results"]),
        "count_positive_samples": len(selected),
        "samples": selected,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    print(f"Saved {len(selected)} samples (top {top_ratio*100:.1f}%) to {args.output}")


if __name__ == "__main__":
    main()
