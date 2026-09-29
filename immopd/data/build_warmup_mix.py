"""Mix teacher samples of the five domains by token budget (SFT warm-up baseline).

  python -m immopd.data.build_warmup_mix --out ${DATA_DIR}/sft/warmup_4b.jsonl \
      --law 2.2 --med 62 --tau 100 --fin 0.36 --if 0.14 \
      --law-pool <law.jsonl> --med-pool <med.jsonl> --tau-pool <tau.jsonl> --fin-pool <fin.jsonl> --if-pool <if.jsonl>

Budgets are in million tokens. The pools are the filtered teacher samples
(``immopd.training.filter_samples`` / ``tau2_filter``); validation prompts are removed.
"""
import argparse
import json
import os
import random

from immopd.common.io import repo_root

DOMS = ("law", "med", "tau", "fin", "if")


def validation_prompts() -> set:
    path = os.path.join(repo_root(), "data", "validation", "pool.jsonl")
    return {json.loads(l)["input"].strip() for l in open(path)}


def load_pool(d: str, path: str, val: set):
    rows, tokens = [], 0
    for ln in open(path):
        r = json.loads(ln)
        if d == "tau":
            t = (r.get("n_prompt_tokens") or 0) + len(r.get("completion") or "") // 3
        else:
            msgs = r["messages"]
            if msgs[0]["content"].strip() in val:
                continue
            t = sum(len(x.get("content") or "") for x in msgs) // 3
        rows.append(ln if ln.endswith("\n") else ln + "\n")
        tokens += t
    return rows, tokens


def main():
    ap = argparse.ArgumentParser()
    for d in DOMS:
        ap.add_argument("--" + d, type=float, default=0.0, help="million tokens")
        ap.add_argument("--%s-pool" % d, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    val = validation_prompts()
    rng = random.Random(a.seed)
    out = []
    for d in DOMS:
        if getattr(a, d) <= 0:
            continue
        rows, tokens = load_pool(d, getattr(a, "%s_pool" % d), val)
        k = int(round(getattr(a, d) * 1e6 / (tokens / len(rows))))
        picked = []
        while len(picked) < k:
            rng.shuffle(rows)
            picked.extend(rows[: k - len(picked)])
        out.extend(picked)
        print("%-4s %d rows" % (d, k))
    rng.shuffle(out)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        f.writelines(out)
    print("wrote", a.out, len(out))


if __name__ == "__main__":
    main()
