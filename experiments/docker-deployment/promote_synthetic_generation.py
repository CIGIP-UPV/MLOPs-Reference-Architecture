from __future__ import annotations

import json

from industrial_mlops.db import load_training_dataset
from industrial_mlops.orchestration import manual_promote
from industrial_mlops.registry import get_production_version, train_and_register


def main() -> None:
    previous = get_production_version()
    frame = load_training_dataset(limit=5000)
    candidate = train_and_register(
        frame,
        reason="ota-continuity-candidate",
        run_name="ota-continuity-candidate",
    )
    promotion = manual_promote(
        candidate["model_version"],
        reason="ota-continuity-experiment",
        issued_by="ota-continuity-experiment",
    )
    current = get_production_version()
    print(
        json.dumps(
            {
                "previous_production": previous,
                "candidate": candidate,
                "promotion": promotion,
                "current_production": current,
            },
            sort_keys=True,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
