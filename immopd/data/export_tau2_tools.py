"""Export the tool schemas tau2-bench sends to the agent (``data/tau2/tools_<domain>.json``).

  python -m immopd.data.export_tau2_tools --out data/tau2      (tau2-bench python)
"""
from __future__ import annotations

import argparse
import json
import os


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/tau2")
    ap.add_argument("--domains", default="airline,retail,telecom")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    from tau2.registry import registry
    for d in a.domains.split(","):
        schemas = [t.openai_schema for t in registry.get_env_constructor(d)().get_tools()]
        json.dump(schemas, open(os.path.join(a.out, "tools_%s.json" % d), "w"), indent=1, ensure_ascii=False)
        print(d, len(schemas), "tools")


if __name__ == "__main__":
    main()
