"""Tables 1, 2, 9 and 10 of the paper (Markdown)."""
import json
from pathlib import Path

D = json.loads((Path(__file__).resolve().parent / "data" / "tables.json").read_text())
TITLES = {"table1_4b": "Table 1, Qwen3-4B", "table1_1p7b": "Table 1, Qwen3-1.7B", "table2_merge_timing": "Table 2",
          "table9_4b_delta": "Table 9 (gamma = 0.6)", "table10_1p7b_gamma_delta": "Table 10 (gamma, delta)"}
for key, title in TITLES.items():
    print("\n**%s**\n\n| | %s |\n|---|%s" % (title, " | ".join(D["columns"]), "---:|" * len(D["columns"])))
    for r in D[key]:
        print("| %s | %s | %.1f |" % (r["label"], " | ".join("%.1f" % x for x in r["scores"]), r["norm"]))
