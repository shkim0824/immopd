"""MOPD policy-gradient signal: per-token advantage A_t = clip(log pi_teacher(y_t) - log pi_student(y_t), +-a_max)."""
from __future__ import annotations

import torch


def pg_advantages(teacher_logps: torch.Tensor, student_logps: torch.Tensor, mask: torch.Tensor,
                  a_max: float = 5.0) -> torch.Tensor:
    adv = (teacher_logps.detach() - student_logps.detach()).clamp(-a_max, a_max)
    return adv * mask


def reverse_kl_estimate(teacher_logps: torch.Tensor, student_logps: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.float()
    return ((student_logps - teacher_logps) * m).sum(-1) / m.sum(-1).clamp(min=1.0)
