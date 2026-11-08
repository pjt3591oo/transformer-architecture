# 직접 만드는 작은 Transformer: 구현 → 학습 → 텍스트 생성

Q·K·V, 마스킹, Softmax, 가중합을 직접 계산하는 **decoder-only 언어 모델** 실습입니다. 사전학습된 모델을 다운로드하지 않고 무작위 가중치부터 학습합니다. PyTorch는 텐서 연산, 자동 미분, 기본 층, 최적화에 사용합니다. `nn.Transformer`나 `nn.MultiheadAttention`은 사용하지 않습니다.

문자 단위 모델로 동작을 이해하는 것이 목표입니다. 짧은 합성 문장의 패턴을 학습하며, 범용 한국어 대화 능력이나 사실 지식을 갖춘 챗봇은 아닙니다.

## 1. 파일

| 파일 | 내용 |
| --- | --- |
| model.py | 직접 구현한 인과적 멀티 헤드 어텐션, 블록, 모델, 생성 루프 |
| run.py | 학습, 검증, 체크포인트 저장/불러오기, 생성, 동작 확인 CLI |
| lora.py | 직접 구현한 LoRA: LoRALinear, 모델에 붙이기, A·B만 저장하기, 병합 |
| make_corpus.py | 360개의 실습용 문장 생성 |
| make_finetune_corpus.py | LoRA 파인튜닝용 문장 90개 생성 |
| finetune.txt | LoRA 파인튜닝에 사용한 UTF-8 데이터 |
| corpus.txt | 실제 학습에 사용한 UTF-8 데이터 |
| requirements.txt | 필요한 패키지 |
| artifacts/model.pt | 실제 학습으로 만든 체크포인트. 검증 loss가 가장 낮았던 평가 시점 |
| artifacts/metrics.json | 실제 실행 환경, 파라미터 수, loss, 시간, 학습 전후 결과 |
| artifacts/samples.txt | 학습 전후 문장 생성 결과 |
| artifacts/train.txt, validation.txt | 실제 분리된 학습/검증 문장 |
| artifacts/lora/lora.pt | LoRA 어댑터. A·B 행렬과 설정값만 저장 (약 15KB) |
| artifacts/lora/metrics.json, samples.txt | LoRA 파인튜닝의 loss 기록과 전후 생성 결과 |
| RUN_RESULT.md | 제공된 체크포인트의 실행 결과 요약 |

## 2. 설치

Python 3.10 이상 환경에서 압축을 풀고 이 폴더로 이동합니다. 실제 검증 환경은 RUN_RESULT.md를 참고하세요.

```bash
python -m venv .venv
```

Linux / macOS / WSL:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

CPU 전용 Linux / Windows / WSL 실습:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cpu

uv sync
```

macOS:

```bash
python -m pip install -r requirements.txt

uv sync
```

GPU를 사용하려면 [PyTorch 공식 설치 선택기](https://pytorch.org/get-started/locally/)에서 운영체제와 CUDA 환경에 맞는 명령으로 설치하세요. `--device auto`는 CUDA, MPS, CPU 순으로 사용 가능한 장치를 선택합니다. 제공 결과는 CPU에서 검증했습니다.

## 3. 먼저 저장된 모델로 생성하기

체크포인트를 함께 제공하므로 재학습 없이 바로 실행할 수 있습니다.

```bash
python run.py generate --device cpu --prompt "나는 오늘 " --tokens 120

uv run run.py generate --device cpu --prompt "나는 오늘 " --tokens 120
```

출력에는 입력 prompt와 새로 생성한 문자열이 함께 표시됩니다. 이 모델에서 토큰 하나는 Unicode 문자 하나입니다. 공백과 줄바꿈도 토큰입니다. 따라서 `--tokens 120`은 새 문자 120개를 의미합니다.

탐욕적 선택과 확률적 선택 비교:

```bash
python run.py generate --prompt "멍개는 " --temperature 0 --tokens 100
python run.py generate --prompt "멍개는 " --temperature 0.8 --top-k 20 --seed 7 --tokens 100

uv run run.py generate --prompt "멍개는 " --temperature 0 --tokens 100
uv run run.py generate --prompt "멍개는 " --temperature 0.8 --top-k 20 --seed 7 --tokens 100
```

`temperature=0`은 매번 최고 점수 토큰을 고릅니다. 양수이면 점수를 온도로 나누고 확률적으로 뽑습니다. `top-k=20`은 점수가 높은 후보만 남기며, `top-k=0`은 후보 제한을 끕니다. 작은 모델에서는 높은 온도가 문법 오류를 늘릴 수 있습니다.

학습 어휘에 없는 문자를 prompt로 넣으면 명시적인 오류가 발생합니다. 이 구현은 UNK 토큰으로 조용히 바꾸지 않습니다.

## 4. 직접 처음부터 학습하기

제공된 체크포인트를 유지하려면 별도 출력 폴더를 지정하세요.

```bash
python run.py check
python make_corpus.py
python run.py train --device cpu --steps 1200 --out my-run
python run.py generate --device cpu --checkpoint my-run/model.pt --prompt "나는 오늘 "

uv run run.py check
uv run make_corpus.py
uv run run.py train --device cpu --steps 1200 --out my-run
uv run run.py generate --device cpu --checkpoint my-run/model.pt --prompt "나는 오늘 "
```

`train`은 항상 무작위 가중치로 새 학습을 시작합니다. 학습 재개 기능은 구현하지 않았습니다. `--out`을 생략하면 `artifacts` 아래의 같은 이름 파일들을 새 실행 결과로 덮어씁니다.

CPU 기본 설정은 2개 층, 4개 헤드, 임베딩 64차원, 문맥 64문자, 배치 16입니다. 빠른 점검만 하려면 `--steps 100 --eval-every 50`으로 줄일 수 있지만 문장 품질은 낮을 수 있습니다.

## 5. 메모와 코드 연결

| 메모의 개념 | 코드 |
| --- | --- |
| 토큰을 벡터로 변환 | token_embedding |
| 위치 정보를 추가 | position_embedding |
| Q, K, V 만들기 | qkv(x).chunk(3, dim=-1) |
| Query와 Key의 내적 | q @ k.transpose(-2, -1) |
| 점수 스케일 조정 | / sqrt(head_dim) |
| 미래 위치 차단 | masked_fill(..., -inf) |
| 참고 비율 계산 | softmax(scores, dim=-1) |
| Value 가중합 | weights @ v |
| 여러 헤드 결합 및 변환 | transpose, view, proj |
| 피드포워드 | Linear → GELU → Linear |
| 다음 토큰 후보 점수 | lm_head |

`X: [B,T,64]` → `Q/K/V: [B,4,T,16]` → `점수: [B,4,T,T]` → `가중합: [B,4,T,16]` → `헤드 결합: [B,T,64]` → `최종 logits: [B,T,어휘 수]` 순서입니다.

학습 가능한 위치 임베딩, Pre-LN, GELU를 사용합니다. 원래 인코더-디코더 트랜스포머를 그대로 재현한 것이 아니라, 인코더와 Cross-Attention 없이 다음 토큰을 예측하는 작은 GPT 형태입니다.

## 6. 어떻게 학습하나?

텍스트를 문자 ID로 바꾸고, 무작위 위치에서 `context+1`개를 가져옵니다. 마지막 한 문자를 제외한 배열이 입력이고, 첫 문자를 제외한 배열이 정답입니다.

예를 들어 `나는 오늘`에서는 다음처럼 한 문자씩 이동합니다.

| 위치 | 입력 문자 | 정답 문자 |
| --- | --- | --- |
| 0 | 나 | 는 |
| 1 | 는 | 공백 |
| 2 | 공백 | 오 |
| 3 | 오 | 늘 |

모델이 각 위치에서 다음 문자에 대한 logits를 계산하면 CrossEntropyLoss로 정답과 비교합니다. `loss.backward()`가 각 파라미터의 기울기를 계산하고, `optimizer.step()`이 가중치를 갱신합니다. 이 과정이 반복되며 데이터에 자주 나타나는 연결 패턴을 배우게 됩니다.

CrossEntropyLoss에는 Softmax를 적용하지 않은 logits를 전달합니다. Softmax는 생성 시 후보 확률을 만드는 데 사용합니다.

검증은 기울기를 계산하지 않고, 학습에 사용하지 않은 문장들의 다음 문자 예측 loss를 측정합니다. 10개 고정 무작위 배치의 평균이므로 검증 데이터 전체를 빠짐없이 평가하는 값은 아닙니다. 평가 때마다 같은 배치를 사용합니다.

## 7. 데이터의 범위와 해석

원본 문장은 주어 6개 × 시간 표현 6개 × 행동 10개를 조합한 360개입니다. 외부 웹 문서나 기존 모델 출력은 사용하지 않았습니다.

중복 문장을 제거한 후 문장 단위로 90% 학습, 10% 검증으로 분리합니다. 완전히 같은 문장이 양쪽에 들어가지는 않지만, 단어와 문장 틀은 공유합니다. **검증 loss 감소는 이 작은 합성 분포를 배웠다는 증거이며 일반적인 한국어 능력의 증거가 아닙니다.**

분리된 각 문장 묶음을 줄바꿈으로 연결한 뒤 학습 창을 뽑습니다. 창은 문장 경계를 넘어갈 수 있습니다. 별도 BOS/EOS 없이 줄바꿈도 다음 문자로 학습하며, 생성은 지정한 문자 수에서 끝납니다.

학습 어휘는 학습 분할에서만 만듭니다. 검증 데이터에만 등장한 문자가 있으면 오류로 알려 줍니다. 사용자 데이터를 적용할 때는 충분히 다양한 반복 출현 문자를 확보하거나 tokenizer/UNK 정책을 확장하세요.

## 8. 자신의 데이터로 바꾸기

UTF-8 텍스트 파일에 줄마다 문장을 넣습니다. 서로 다른 문장 10개 이상이 필요하며, 학습과 검증 분할 모두 문맥 길이보다 길어야 합니다. 작은 숫자만 충족하기보다 충분한 문장을 준비하는 것이 좋습니다.

```bash
python run.py train --data my-text.txt --out custom-run --prompt "학습 데이터에 있는 문장" --steps 3000
python run.py generate --checkpoint custom-run/model.pt --prompt "학습 데이터에 있는 문장"
```

사용자 데이터에 맞게 prompt를 바꾸세요. 재학습으로 새 문자 어휘와 새 모델을 만듭니다. 기존 체크포인트에 새 패턴을 덧붙이려면 9장의 LoRA 파인튜닝을 사용하세요.

## 9. LoRA로 파인튜닝하기

LoRA(Low-Rank Adaptation)는 사전학습된 가중치 W₀를 고정하고, 작은 행렬 두 개 A, B만 학습해 보정값을 더하는 방법입니다.

```text
W = W₀ + (α/r) · B · A

W₀ : [out, in]   고정
B  : [out, r]    학습, 0으로 초기화
A  : [r, in]     학습, 작은 랜덤값으로 초기화
```

- `r`(rank)은 A, B의 안쪽 차원입니다. 보정값이 가질 수 있는 자유도와 파라미터 수를 정합니다.
- `α/r`은 LoRA 출력에 곱하는 고정 상수입니다. r을 바꿔도 출력 크기가 비슷하게 유지되도록 합니다.
- B가 0에서 시작하므로 LoRA를 붙인 직후의 모델은 원래 모델과 출력이 같습니다.
- 학습이 끝나면 `W₀ + (α/r)·BA`를 하나의 Linear로 병합하므로 추론 비용이 늘지 않습니다.

LLM 전용 기법이 아닙니다. 행렬로 된 가중치를 가진 사전학습 모델이라면 어디에나 붙일 수 있고, `apply_lora`도 모델 종류와 관계없이 이름이 맞는 `nn.Linear`를 찾아 교체합니다.

### 9.1 실행

```bash
python make_finetune_corpus.py
python run.py finetune --device cpu
python run.py generate --device cpu --lora artifacts/lora/lora.pt --prompt "멍개는 "

uv run make_finetune_corpus.py
uv run run.py finetune --device cpu
uv run run.py generate --device cpu --lora artifacts/lora/lora.pt --prompt "멍개는 "
```

`finetune`은 `artifacts/model.pt`를 불러와 고정하고, 어텐션의 `qkv`, `proj` 층에 LoRA를 붙여 `finetune.txt`로 A, B만 학습합니다. 검증 loss가 가장 낮은 시점의 A, B를 `artifacts/lora/lora.pt`에 저장합니다. 베이스 모델 파일은 바뀌지 않습니다.

`generate`에 `--lora`를 주면 어댑터를 베이스 모델에 병합해서 생성하고, 빼면 원래 모델로 생성합니다.

| 옵션 | 기본값 | 의미 |
| --- | --- | --- |
| `--base` | artifacts/model.pt | 고정할 사전학습 체크포인트 |
| `--data` | finetune.txt | 파인튜닝 데이터 |
| `--out` | artifacts/lora | 어댑터 저장 폴더 |
| `--targets` | qkv,proj | LoRA를 붙일 Linear 이름. `qkv,proj,0,2`로 주면 FFN에도 붙입니다. |
| `--rank` | 4 | r |
| `--alpha` | 8 | α |
| `--lr` | 0.005 | 사전학습보다 높게 잡습니다. 학습할 값이 적고 B가 0에서 시작하기 때문입니다. |
| `--steps` | 600 | 학습 단계 수 |

### 9.2 파인튜닝 데이터

사전학습 데이터는 한 줄에 한 문장입니다. 파인튜닝 데이터는 `멍개는`으로 시작하는 줄에 두 문장이 이어지는 새 패턴입니다.

```text
멍개는 아침에 책을 읽는다. 저녁에 코드를 작성한다.
```

문자 단위 모델이므로 **베이스 모델의 어휘에 있는 문자만** 사용할 수 있습니다. 파인튜닝할 때는 데이터로 어휘를 새로 만들지 않고 체크포인트의 어휘를 그대로 씁니다. 제공된 `artifacts/model.pt`는 `잠을 잔다.`가 추가되기 전에 학습되어 `잠`, `잔`이 어휘에 없으므로, 이 문장은 파인튜닝 데이터에서 뺐습니다.

### 9.3 실제 실행 결과

| 항목 | 값 |
| --- | --- |
| LoRA 적용 층 | qkv, proj × 2블록 (4개 층) |
| 학습 파라미터 | 3,072 / 114,816 (2.68%) |
| 어댑터 크기 | 약 15KB |
| 학습 | 600단계, CPU 6.9초, 300단계 선택 |
| 새 데이터 검증 loss | 0.6557 → 0.2424 |
| 원래 데이터 검증 loss | 0.3760 → 0.6492 |

```text
파인튜닝 전: 멍개는 내일 운동을 한다.
파인튜닝 후: 멍개는 아침에 책을 읽는다. 저녁에 운동을 한다.
```

새 패턴은 배웠지만 원래 데이터의 검증 loss가 올라갔습니다. 새 패턴을 배우면서 기존 분포를 잊은 것입니다(catastrophic forgetting). 생성 결과에서도 `학생은 아침에 …. 저녁에 …`처럼 멍개가 아닌 주어에 새 패턴이 번졌습니다. 모델이 작고 파인튜닝 데이터가 한 가지 패턴뿐이라 두드러지며, 원래 데이터를 섞어 학습하면 줄일 수 있습니다.

### 9.4 개념과 코드 연결

| 개념 | 코드 |
| --- | --- |
| W₀ 고정 | `p.requires_grad_(False)` |
| A, B 생성 | `nn.Parameter(torch.empty(r, in))`, `nn.Parameter(torch.zeros(out, r))` |
| α/r | `self.scaling = alpha / r` |
| W₀x + (α/r)·BAx | `LoRALinear.forward` |
| 대상 층 교체 | `apply_lora` |
| A, B만 최적화 | `AdamW([p for p in model.parameters() if p.requires_grad])` |
| A, B만 저장 | `lora_state_dict` |
| W = W₀ + (α/r)·BA 병합 | `LoRALinear.merged`, `merge_lora` |

어댑터는 `LoRALinear` 4개의 A, B 묶음과 설정값(`targets`, `rank`, `alpha`)입니다. 특정 베이스 모델을 기준으로 한 차이값이므로, 베이스 모델을 다시 학습하면 어댑터도 다시 학습해야 합니다. 현재 구현은 한 번에 어댑터 하나만 적용합니다.

## 10. 확인한 동작과 교육용 생략

`python run.py check`는 미래 토큰을 바꿔도 앞 위치의 출력이 변하지 않는지, 출력 텐서 크기, 역전파의 유한한 기울기, 문맥 길이를 넘는 생성을 확인합니다. LoRA에 대해서는 붙인 직후 출력이 원래 모델과 같은지, A·B만 학습되는지, 병합 전후 출력이 같은지 확인합니다.

KV cache, FlashAttention, GQA, MoE, 분산 학습은 넣지 않았습니다. 생성할 때 매번 최근 64문자를 재계산하고 그 창 안에서 위치를 0부터 부여합니다. 긴 문맥을 유지하는 최적화된 LLM 서빙 방식과는 차이가 있습니다. 이 구현에서 어텐션을 직접 수정하며 GQA 등을 실험할 수 있습니다.

CPU 스레드 기본값은 2입니다. 작은 모델은 스레드를 과하게 늘리면 느려질 수 있습니다. 실행 시간과 확률적 출력은 하드웨어/라이브러리 버전에 따라 달라질 수 있습니다.

## 참고

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [PyTorch CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
- [PyTorch multinomial](https://docs.pytorch.org/docs/stable/generated/torch.multinomial.html)

학습 데이터와 코드는 본 실습을 위해 작성했습니다.
