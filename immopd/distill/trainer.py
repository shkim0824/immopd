"""MOPDTrainer: trl ``GRPOTrainer`` whose advantages come from the domain teachers.

Each step samples student rollouts, sends every rollout to the teacher of its domain for per-token
log-probabilities, and updates the student with the clipped log-ratio advantage.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
from trl import GRPOTrainer

from immopd.distill import loss as L
from immopd.distill.teacher_client import TeacherPool


def zero_reward(completions=None, **kw) -> List[float]:
    return [0.0] * len(completions)


zero_reward.__name__ = "zero"


class MOPDTrainer(GRPOTrainer):
    def __init__(self, *args, teacher_pool: TeacherPool, a_max: float = 5.0,
                 num_generations_override: Optional[int] = None, **kwargs):
        super().__init__(*args, reward_funcs=[zero_reward], **kwargs)
        self.teacher_pool = teacher_pool
        self.a_max = a_max
        if num_generations_override is not None:
            self.num_generations = num_generations_override
        self._mopd_metrics: Dict[str, List[float]] = {}

    def _generate_and_score_completions(self, inputs):
        out = super()._generate_and_score_completions(inputs)
        device = out["completion_ids"].device
        prompt_ids, prompt_mask = out["prompt_ids"], out["prompt_mask"]
        completion_ids, completion_mask = out["completion_ids"], out["completion_mask"]
        B, T = completion_ids.shape
        student_logps = out.get("old_per_token_logps")
        if student_logps is None:
            input_ids = torch.cat([prompt_ids, completion_ids], 1)
            attn = torch.cat([prompt_mask, completion_mask], 1)
            with torch.no_grad():
                student_logps, _ = self._get_per_token_logps_and_entropies(self.model, input_ids, attn, T,
                                                                           self.args.per_device_train_batch_size)
            out["old_per_token_logps"] = student_logps
        domains = [x["domain"] for x in inputs]
        assert len(domains) == B, (len(domains), B)
        seqs, starts = [], []
        for i in range(B):
            p = prompt_ids[i][prompt_mask[i].bool()].tolist()
            c_len = int(completion_mask[i].sum().item())
            c_full = completion_ids[i].tolist()
            c = c_full[:c_len if c_len > 0 else len(c_full)]
            seqs.append(p + c)
            starts.append(len(p))
        res = self.teacher_pool.prefill_batch(domains, seqs, starts)
        teacher_logps = torch.zeros((B, T), dtype=torch.float32, device=device)
        for i, lp in enumerate(res):
            n = min(len(lp), T)
            teacher_logps[i, :n] = torch.tensor(lp[:n], dtype=torch.float32, device=device)
        mask = completion_mask.float()
        out["advantages"] = L.pg_advantages(teacher_logps, student_logps.float(), mask, self.a_max)
        out["teacher_logps"] = teacher_logps
        kl = L.reverse_kl_estimate(teacher_logps, student_logps.float(), mask)
        a = out["advantages"][mask.bool()]
        m = self._mopd_metrics
        m.setdefault("mopd/student_teacher_kl", []).append(self.accelerator.gather(kl.mean()).mean().item())
        m.setdefault("mopd/adv_mean", []).append(a.mean().item() if a.numel() else 0.0)
        m.setdefault("mopd/adv_abs_mean", []).append(a.abs().mean().item() if a.numel() else 0.0)
        m.setdefault("mopd/adv_clipped_frac", []).append(
            (a.abs() >= self.a_max - 1e-6).float().mean().item() if a.numel() else 0.0)
        for d in set(domains):
            m.setdefault(f"mopd/frac_{d}", []).append(sum(1 for x in domains if x == d) / B)
        return out

    def log(self, logs, *a, **k):
        for key, vals in self._mopd_metrics.items():
            if vals:
                logs[key] = sum(vals) / len(vals)
        self._mopd_metrics = {}
        logs.update(self.teacher_pool.flush_stats())
        super().log(logs, *a, **k)
