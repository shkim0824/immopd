"""Prefix budgets for the Tool Use continuation: N(alpha) = round(alpha * mean tokens per assistant turn).

  python -m immopd.eval.prefix.token_stats --results <dir>/tau2_telecom_base.json --alphas 0.1,0.3,0.5,0.7
"""
import argparse
import json
import statistics


def mean_turn_tokens(results_path: str) -> float:
    sims = json.load(open(results_path))["simulations"]
    toks = [m["usage"]["completion_tokens"] for s in sims for m in (s.get("messages") or [])
            if m.get("role") == "assistant" and (m.get("usage") or {}).get("completion_tokens") is not None]
    return statistics.mean(toks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--alphas", default="0.1,0.3,0.5,0.7")
    a = ap.parse_args()
    mean = mean_turn_tokens(a.results)
    print(json.dumps({"mean_tokens_per_turn": mean,
                      "N": {x: int(round(float(x) * mean)) for x in a.alphas.split(",")}}, indent=1))


if __name__ == "__main__":
    main()
