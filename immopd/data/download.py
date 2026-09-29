"""External models, datasets and benchmark repositories.

  python -m immopd.data.download --list
  python -m immopd.data.download --dest ${DATA_DIR}/raw [--only models,medical,law,finance,if,tau2,sft]

Models go to ``${MODEL_DIR}``; datasets and repositories to ``<dest>/{hf,domains,third_party}``.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

SOURCES = [
    ("models", "model", "Qwen/Qwen3-4B-Base", "Qwen3-4B-Base"),
    ("models", "model", "Qwen/Qwen3-1.7B-Base", "Qwen3-1.7B-Base"),
    ("sft", "dataset", "open-thoughts/OpenThoughts3-1.2M", "hf/open-thoughts__OpenThoughts3-1.2M"),
    ("medical", "dataset", "GBaker/MedQA-USMLE-4-options", "domains/med_medqa"),
    ("medical", "dataset", "bigbio/med_qa", "domains/med_medqa_5opt"),
    ("law", "dataset", "casehold/casehold", "domains/law_casehold"),
    ("law", "dataset", "reglab/barexam_qa", "domains/law_barexam_qa"),
    ("finance", "git", "https://github.com/czyssrs/FinQA", "third_party/FinQA"),
    ("finance", "git", "https://github.com/NExTplusplus/TAT-QA", "third_party/TAT-QA"),
    ("if", "dataset", "nvidia/Nemotron-RL-Ultra-Training-Blends", "hf/nvidia__Nemotron-RL-Ultra-Training-Blends"),
    ("tau2", "dataset", "inclusionAI/AReaL-tau2-data", "hf/inclusionAI__AReaL-tau2-data"),
    ("tau2", "git", "https://github.com/sierra-research/tau2-bench", "third_party/tau2-bench"),
]


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dest", default=os.path.join(os.environ.get("DATA_DIR", "data"), "raw"))
    p.add_argument("--models", default=os.environ.get("MODEL_DIR", "models"))
    p.add_argument("--only", default="")
    p.add_argument("--list", action="store_true")
    a = p.parse_args(argv)
    groups = set(a.only.split(",")) if a.only else None
    for group, kind, ident, dest in SOURCES:
        if groups and group not in groups:
            continue
        target = os.path.join(a.models if kind == "model" else a.dest, dest)
        if a.list:
            print("%-8s %-8s %-50s %s" % (group, kind, ident, target))
            continue
        if os.path.exists(target) and os.listdir(target):
            continue
        os.makedirs(os.path.dirname(target), exist_ok=True)
        if kind == "git":
            subprocess.check_call(["git", "clone", "--depth", "1", ident, target])
        else:
            from huggingface_hub import snapshot_download
            snapshot_download(ident, repo_type=kind, local_dir=target)
    return 0


if __name__ == "__main__":
    sys.exit(main())
