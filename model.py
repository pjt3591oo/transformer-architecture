"""작은 decoder-only Transformer. Attention 계산은 직접 구현합니다."""
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class Config:
    vocab_size: int
    context: int = 64
    dim: int = 64
    heads: int = 4
    layers: int = 2
    dropout: float = 0.0

'''
model architecture 

입력 토큰 ID · [B, T]
   ↓
토큰 임베딩 · token_embedding
   └─ 각 토큰 ID를 64차원 벡터로 변환
   +
학습 가능한 위치 임베딩 · position_embedding
   └─ 각 위치를 64차원 벡터로 변환
   ↓
토큰 표현 · [B, T, 64]
   ↓
디코더 블록 X 2 · blocks
   │
   ├─ Layer Normalization · norm1
   │
   ├─ Masked Multi-Head Self-Attention · attention
   │    ├─ 선형 변환으로 Q, K, V 생성 · qkv
   │    ├─ 각각 4개 헤드로 분리
   │    │    └─ 헤드당 16차원 · [B, 4, T, 16]
   │    ├─ Q X Kᵀ로 어텐션 점수 계산
   │    ├─ √16으로 나누어 점수 크기 조정
   │    ├─ 미래 토큰 위치를 -∞로 마스킹
   │    ├─ Softmax로 어텐션 가중치 계산
   │    ├─ 어텐션 가중치에 Dropout
   │    ├─ 가중치 X V로 정보 가중합
   │    ├─ 4개 헤드 결과 결합 · [B, T, 64]
   │    └─ 출력 선형 변환 · proj
   │
   ├─ 잔차 연결
   │    └─ 블록 입력 + 어텐션 결과
   │
   ├─ Layer Normalization · norm2
   │
   ├─ Feed-Forward Network · ffn
   │    ├─ 선형 변환 · 64 → 256차원
   │    ├─ GELU 활성화 함수
   │    ├─ 선형 변환 · 256 → 64차원
   │    └─ Dropout
   │
   └─ 잔차 연결
        └─ FFN 진입 전 입력 + FFN 결과
   ↓
최종 Layer Normalization · norm
   ↓
어휘 출력층 · lm_head
   └─ 64차원 → vocab_size차원
   ↓
어휘별 점수 · logits
   └─ [B, T, vocab_size]
'''

'''
입력 프롬프트의 토큰 ID
   ↓
generate()
   ├─ eval(): Dropout 비활성화
   └─ no_grad(): 기울기 계산 비활성화
   ↓
아래 과정을 count번 반복
   │
   ├─ 최근 context개 토큰 선택 · 기본 최대 64개
   │
   ├─ MiniGPT.forward() 실행
   │    └─ 임베딩 → 블록 2개 → 정규화 → 어휘 출력층
   │
   ├─ 마지막 입력 위치의 logits 선택
   │    └─ logits[:, -1, :]
   │
   ├─ 다음 토큰 ID 선택
   │    ├─ temperature == 0
   │    │    └─ 가장 높은 점수의 ID 선택 · argmax
   │    └─ temperature > 0
   │         ├─ logits를 temperature로 나누기
   │         ├─ top_k > 0이면 상위 후보만 남기기
   │         ├─ Softmax로 확률 계산
   │         └─ 확률에 따라 ID 하나 추출 · multinomial
   │
   └─ 선택한 ID를 입력 토큰 뒤에 추가
   ↓
프롬프트 + 생성된 토큰 ID 반환
   ↓
토크나이저의 decode()        ← 외부에서 호출
   ↓
최종 텍스트
'''

class CausalAttention(nn.Module):
    def __init__(self, c):
        super().__init__()
        if c.dim % c.heads:
            raise ValueError('dim은 heads로 나누어 떨어져야 합니다.')
        self.heads = c.heads
        self.head_dim = c.dim // c.heads
        self.qkv = nn.Linear(c.dim, 3 * c.dim, bias=False)
        self.proj = nn.Linear(c.dim, c.dim, bias=False)
        self.dropout = nn.Dropout(c.dropout)
        self.register_buffer('mask', torch.tril(torch.ones(c.context, c.context, dtype=torch.bool)))

    def forward(self, x):
        batch, length, dim = x.shape
        # 한 번의 Linear 계산을 세 부분으로 나누면 각각 XW_Q, XW_K, XW_V입니다.
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(batch, length, self.heads, self.head_dim).transpose(1, 2)
                   for t in (q, k, v)]  # [B, H, T, D/H]
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~self.mask[:length, :length], float('-inf'))
        weights = F.softmax(scores, dim=-1)  # 각 query가 참고하는 key 위치의 비율
        mixed = self.dropout(weights) @ v
        mixed = mixed.transpose(1, 2).contiguous().view(batch, length, dim)
        return self.proj(mixed)


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.norm1 = nn.LayerNorm(c.dim)
        self.attention = CausalAttention(c)
        self.norm2 = nn.LayerNorm(c.dim)
        # FFN: Feed-Forward Network.
        self.ffn = nn.Sequential(nn.Linear(c.dim, c.dim * 4), nn.GELU(),
                                 nn.Linear(c.dim * 4, c.dim), nn.Dropout(c.dropout))

    def forward(self, x):
        # Pre-LN 배치: 원 논문의 Post-LN과 정규화 위치가 다릅니다.
        x = x + self.attention(self.norm1(x))
        return x + self.ffn(self.norm2(x))


class MiniGPT(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.config = c
        self.token_embedding = nn.Embedding(c.vocab_size, c.dim)
        self.position_embedding = nn.Embedding(c.context, c.dim)
        self.blocks = nn.Sequential(*[Block(c) for _ in range(c.layers)])
        self.norm = nn.LayerNorm(c.dim)
        self.lm_head = nn.Linear(c.dim, c.vocab_size, bias=False)
        self.apply(self._init)

    @staticmethod
    def _init(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, ids, targets=None):
        _, length = ids.shape
        if not 1 <= length <= self.config.context:
            raise ValueError('입력 길이는 1 이상 context 이하여야 합니다.')
        positions = torch.arange(length, device=ids.device)
        x = self.token_embedding(ids) + self.position_embedding(positions)
        logits = self.lm_head(self.norm(self.blocks(x)))  # [B, T, vocab_size]
        # CrossEntropy는 내부에서 log-softmax를 계산하므로 logits를 바로 전달합니다.
        loss = None if targets is None else F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    # @torch.no_grad()는 gradient 계산을 하지 않도록 합니다. 
    # 학습 시에는 필요 없지만, 추론 시에는 속도를 높이고 메모리 절약
    @torch.no_grad()
    def generate(self, ids, count=120, temperature=0.7, top_k=20):
        self.eval()
        for _ in range(count):
            # 교육용 구현: KV cache 없이 최근 context를 매번 다시 계산합니다.
            logits, _ = self(ids[:, -self.config.context:])
            logits = logits[:, -1, :]

            if temperature == 0:
                nxt = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k > 0:
                    cutoff = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1:]
                    logits = logits.masked_fill(logits < cutoff, float('-inf'))
                nxt = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            ids = torch.cat((ids, nxt), dim=1)
        print(ids)
        return ids
