from __future__ import annotations

import json

from industrial_mlops.orchestration import bootstrap_platform


if __name__ == "__main__":
    summary = bootstrap_platform()
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))
