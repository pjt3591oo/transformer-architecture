"""LoRA(Low-Rank Adaptation). 사전학습된 Linear는 고정하고 저랭크 행렬 B·A만 학습합니다."""
import math
import torch
from torch import nn

'''
LoRALinear

입력 x · [..., in]
   ├─ 고정된 원래 Linear · base          → W₀x
   └─ LoRA 경로
        ├─ Dropout
        ├─ A · in → r차원   (작은 랜덤값으로 초기화)
        ├─ B · r → out차원  (0으로 초기화)
        └─ scaling = alpha / r 곱하기  → (alpha/r)·BAx
   ↓
W₀x + (alpha/r)·BAx

B가 0에서 시작하므로 학습 첫 단계의 출력은 원래 모델과 정확히 같습니다.
'''


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=4, alpha=8, dropout=0.0):
        super().__init__()
        if r < 1:
            raise ValueError('LoRA rank r은 1 이상이어야 합니다.')
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout)
        self.A = nn.Parameter(torch.empty(r, base.in_features, device=base.weight.device))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + (self.dropout(x) @ self.A.T @ self.B.T) * self.scaling

    @torch.no_grad()
    def merged(self):
        # W = W₀ + (alpha/r)·BA 를 하나의 Linear로 합칩니다. 추론 시 추가 계산이 없습니다.
        layer = nn.Linear(self.base.in_features, self.base.out_features,
                          bias=self.base.bias is not None, device=self.base.weight.device)
        layer.weight.copy_(self.base.weight + (self.B @ self.A) * self.scaling)
        if self.base.bias is not None:
            layer.bias.copy_(self.base.bias)
        return layer


def apply_lora(model, targets=('qkv', 'proj'), r=4, alpha=8, dropout=0.0):
    """이름이 targets 중 하나로 끝나는 nn.Linear를 LoRALinear로 교체하고 나머지는 모두 고정합니다."""
    for p in model.parameters():
        p.requires_grad_(False)
    names = [name for name, m in model.named_modules()
             if isinstance(m, nn.Linear) and name.split('.')[-1] in targets]
    if not names:
        raise ValueError(f'LoRA를 적용할 Linear가 없습니다: {targets}')
    for name in names:
        parent_name, _, child = name.rpartition('.')
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child, LoRALinear(getattr(parent, child), r, alpha, dropout))
    return names


def lora_state_dict(model):
    """학습한 A, B 행렬만 꺼냅니다. 베이스 모델 없이 이 작은 파일만 저장하면 됩니다."""
    return {k: v.detach().cpu() for k, v in model.state_dict().items()
            if k.endswith('.A') or k.endswith('.B')}


def merge_lora(model):
    """모든 LoRALinear를 병합된 nn.Linear로 되돌립니다. 결과 모델은 원래 MiniGPT와 같은 구조입니다."""
    names = [name for name, m in model.named_modules() if isinstance(m, LoRALinear)]
    for name in names:
        parent_name, _, child = name.rpartition('.')
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child, getattr(parent, child).merged())
    return model
