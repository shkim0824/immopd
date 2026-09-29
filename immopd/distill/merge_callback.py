"""In-training merge of IM-MOPD (Algorithm 1): theta <- theta + delta * sum_{i in M} tau_i.

After the MOPD update of a merge step the checkpoint is saved and evaluated on the validation set
(``immopd.distill.driver``), which writes ``DECISION_<step>.json``. Before the next update the selected task
vectors are added to the fp32 master weights held by DeepSpeed, and the bf16 model weights are refreshed from
them. Optimizer state is kept. The merged weights are also saved as ``postmerge-<step>``.
"""
from __future__ import annotations

import json
import os
import shutil
import time

import torch
from transformers import TrainerCallback

from immopd.distill.runtime import is_main
from immopd.merging.delta_store import DeltaStore

PROBE = "layers.0.self_attn.q_proj.weight"


def atomic_json(path: str, obj) -> None:
    tmp = path + ".tmp.%d" % os.getpid()
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


class IterativeMergeCallback(TrainerCallback):
    def __init__(self, out_dir: str, steps, decision_dir: str | None = None, plan: dict | None = None,
                 wait_s: float = 4 * 3600, stop_at_decision: bool = True):
        self.out_dir, self.steps = out_dir, set(int(s) for s in steps)
        self.dir = decision_dir or os.path.join(out_dir, "decisions")
        self.wait_s, self.stop_at_decision = wait_s, stop_at_decision
        self.applied = set()
        self.stores = {int(k): DeltaStore(v) for k, v in (plan or {}).items()}
        self.fixed = plan is not None
        if self.fixed:
            assert set(self.stores) == self.steps, (sorted(self.stores), sorted(self.steps))
        os.makedirs(self.dir, exist_ok=True)

    def on_train_begin(self, args, state, control, model=None, **kw):
        params = list(model.named_parameters())
        for st in self.stores.values():
            st.check_model(model)
        assert all(hasattr(p, "_hp_mapping") for _, p in params), "DeepSpeed ZeRO fp32 master weights are required"
        name, p0 = next((n, p) for n, p in params if n.endswith(PROBE))
        err = float((p0.get_full_hp_param().to(p0.dtype).float() - p0.data.float()).abs().max())
        assert err < 1e-3, f"master and model weights differ on {name}: {err}"
        if is_main():
            print(f"[immopd] merge steps {sorted(self.steps)}, decisions in {self.dir}", flush=True)

    def on_step_begin(self, args, state, control, model=None, **kw):
        s = state.global_step
        if s not in self.steps or s in self.applied:
            return
        if self.fixed:
            adds = self.stores[s].info.get("adds", {})
            self._merge(model, s, {"picked": list(adds), "adds": adds}, self.stores[s])
        else:
            dec = self._wait_decision(s)
            if dec.get("delta_dir"):
                self._merge(model, s, dec, DeltaStore(dec["delta_dir"]))
            elif is_main():
                self._log({"step": s, "merged": [], "recovery": dec.get("recovery")})
        self.applied.add(s)

    def on_step_end(self, args, state, control, **kw):
        s = state.global_step
        pending = s in self.steps and s not in self.applied and not self.fixed
        if pending and self.stop_at_decision and not os.path.exists(self._decision(s)):
            if is_main():
                self._request(s)
            control.should_training_stop = True
            control.should_save = True
        return control

    def _decision(self, s: int) -> str:
        return os.path.join(self.dir, f"DECISION_{s}.json")

    def _request(self, s: int) -> None:
        atomic_json(os.path.join(self.dir, f"REQUEST_{s}.json"),
                    {"step": s, "checkpoint": os.path.join(self.out_dir, f"checkpoint-{s}")})

    def _log(self, rec: dict) -> None:
        print("[immopd] " + json.dumps(rec), flush=True)
        with open(os.path.join(self.out_dir, "merge_log.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")

    def _wait_decision(self, s: int) -> dict:
        path = self._decision(s)
        if not os.path.exists(path):
            torch.cuda.empty_cache()
            if is_main():
                self._request(s)
            t0 = time.time()
            while not os.path.exists(path):
                os.listdir(self.dir)
                if time.time() - t0 > self.wait_s:
                    raise RuntimeError(f"no decision for step {s} after {self.wait_s / 3600:.1f} h")
                time.sleep(10)
        for _ in range(60):
            try:
                return json.load(open(path))
            except (json.JSONDecodeError, OSError):
                time.sleep(2)
        raise RuntimeError(f"unreadable decision {path}")

    @torch.no_grad()
    def _merge(self, model, s: int, dec: dict, store: DeltaStore) -> None:
        store.check_model(model)
        err = None
        for n, p in model.named_parameters():
            d = store.get(n).to(p.device, torch.float32)
            before = p.get_full_hp_param() if n.endswith(PROBE) else None
            if p._hp_mapping is not None:
                a = p._hp_mapping.lp_fragment_address
                hp = p._hp_mapping.get_hp_fragment()
                hp.data.add_(torch.narrow(d.flatten(), 0, a.start, a.numel).to(hp.dtype))
            full = p.get_full_hp_param()
            if before is not None:
                err = float((full - before - d).abs().max())
            p.data.copy_(full.to(p.dtype))
            del d, full
        torch.cuda.empty_cache()
        if is_main():
            self._log({"step": s, "merged": dec.get("picked"), "adds": store.info.get("adds"),
                       "recovery": dec.get("recovery"), "max_abs_err": err,
                       "postmerge": self._save_postmerge(model, s)})

    def _save_postmerge(self, model, s: int):
        out = os.path.join(self.out_dir, f"postmerge-{s}")
        if os.path.exists(os.path.join(out, "MERGED_OK")):
            return out
        try:
            tmp = out + ".saving"
            shutil.rmtree(tmp, ignore_errors=True)
            shutil.rmtree(out, ignore_errors=True)
            sd = {n: p.detach().to("cpu") for n, p in model.named_parameters()}
            model.save_pretrained(tmp, state_dict=sd, safe_serialization=True, max_shard_size="5GB")
            src = os.path.join(self.out_dir, f"checkpoint-{s}")
            for f in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json",
                      "added_tokens.json", "chat_template.jinja", "generation_config.json"):
                if os.path.isfile(os.path.join(src, f)):
                    shutil.copy(os.path.join(src, f), os.path.join(tmp, f))
            open(os.path.join(tmp, "MERGED_OK"), "w").write("ok")
            os.replace(tmp, out)
            return out
        except Exception as e:  # noqa: BLE001
            print(f"[immopd] post-merge snapshot for step {s} failed: {e!r}", flush=True)
            return None
