"""Keep teacher samples of Tool Use contexts as SFT rows (SFT warm-up data).

  python -m immopd.training.tau2_filter --gen-dir <dir> --pool ${DATA_DIR}/pools/tau_train.jsonl \
      --ref ${DATA_DIR}/sft/tau2_sft.jsonl --out <sft.jsonl> --keep 3

A sample is kept if it finished, has one thinking block and takes the same kind of action as the reference
turn: the same tool if the reference calls a tool, a text reply otherwise.
"""
from __future__ import annotations

import argparse
import difflib
import glob
import json
import os
import re
from collections import Counter

CJK = re.compile(r"[぀-ヿ一-鿿가-힯]")
TOOL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))


def tool_name(call: str):
    try:
        return json.loads(call).get("name")
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-dir", required=True)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--tools-dir", default=os.path.join(REPO, "data", "tau2"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep", type=int, default=3)
    ap.add_argument("--sim", type=float, default=0.85)
    ap.add_argument("--max-chars", type=int, default=20000)
    a = ap.parse_args()
    tools = {d: {t["function"]["name"] for t in json.load(open(os.path.join(a.tools_dir, "tools_%s.json" % d)))}
             for d in ("airline", "retail", "telecom")}
    pool = {r["id"]: r for r in map(json.loads, open(a.pool, encoding="utf-8"))}
    ref = {(r.get("source_dialog_id"), r.get("turn_index")): r for r in map(json.loads, open(a.ref, encoding="utf-8"))}
    stats = Counter()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fo:
        for f in sorted(glob.glob(os.path.join(a.gen_dir, "shard*.jsonl"))):
            for line in open(f, encoding="utf-8"):
                g = json.loads(line)
                pr = pool[g["id"]]
                rr = ref.get((pr.get("source_dialog_id"), pr.get("turn_index")))
                if rr is None:
                    continue
                m = TOOL_RE.search(rr.get("completion") or "")
                want = tool_name(m.group(1)) if m else None
                kept = []
                for s in g.get("samples") or []:
                    txt = (s.get("text") or "").strip()
                    if s.get("finish") != "stop" or len(txt) > a.max_chars or CJK.search(txt):
                        continue
                    if txt.count("<think>") != 1 or txt.count("</think>") != 1 or not txt.startswith("<think>"):
                        continue
                    after = txt.split("</think>", 1)[1]
                    calls = TOOL_RE.findall(after)
                    if want is not None:
                        if len(calls) != 1 or tool_name(calls[0]) != want or want not in tools[pr["sub_domain"]]:
                            continue
                    elif calls or not after.strip():
                        continue
                    kept.append(txt)
                kept.sort(key=len)
                chosen = []
                for t in kept:
                    if all(difflib.SequenceMatcher(None, t[:2000], c[:2000]).ratio() < a.sim for c in chosen):
                        chosen.append(t)
                    if len(chosen) >= a.keep:
                        break
                stats["prompts"] += 1
                stats["rows"] += len(chosen)
                for t in chosen:
                    fo.write(json.dumps({"prompt_text": pr["prompt_text"], "completion": t, "sub_domain": pr["sub_domain"],
                                         "source_dialog_id": pr.get("source_dialog_id"), "turn_index": pr.get("turn_index"),
                                         "n_prompt_tokens": pr.get("n_prompt_tokens")}, ensure_ascii=False) + "\n")
    print(json.dumps(dict(stats)))


if __name__ == "__main__":
    main()
