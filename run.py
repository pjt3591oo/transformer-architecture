"""python run.py train / finetune / generate / check"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import time
import torch
from model import Config, MiniGPT
from lora import apply_lora, lora_state_dict, merge_lora

ROOT = Path(__file__).resolve().parent


def device_for(name):
    if name == 'auto':
        if torch.cuda.is_available():
            return 'cuda'
        if torch.backends.mps.is_available():
            return 'mps'
        return 'cpu'
    return name


def encode(text, chars):
    lookup = {ch: i for i, ch in enumerate(chars)}
    missing = sorted(set(text) - set(lookup))
    if missing:
        raise ValueError(f'학습 어휘에 없는 문자가 있습니다: {missing}. 다른 prompt 또는 더 넓은 학습 데이터를 사용하세요.')
    return [lookup[ch] for ch in text]


def sample(model, chars, prompt, device, count=120, temperature=0.7, top_k=20, seed=42):
    torch.manual_seed(seed)
    ids = torch.tensor([encode(prompt, chars)], dtype=torch.long, device=device)
    out = model.generate(ids, count, temperature, top_k)[0].tolist()
    return ''.join(chars[i] for i in out)


def split_data(path, context, seed, chars=None):
    # 중복 문장을 제거한 다음 문장 단위로 분리합니다.
    lines = list(dict.fromkeys(line.strip() for line in path.read_text(encoding='utf-8').splitlines() if line.strip()))
    if len(lines) < 10:
        raise ValueError('서로 다른 문장을 줄 단위로 10개 이상 넣어 주세요.')
    random.Random(seed).shuffle(lines)
    cut = min(len(lines) - 1, max(1, int(len(lines) * 0.9)))
    train_text = '\n'.join(lines[:cut]) + '\n'
    val_text = '\n'.join(lines[cut:]) + '\n'
    # 어휘는 학습 데이터로만 만듭니다. 검증에만 등장하는 문자가 있으면 알려 줍니다.
    # 파인튜닝처럼 기존 어휘를 써야 할 때는 chars를 넘겨받습니다.
    chars = chars or sorted(set(train_text))
    train = torch.tensor(encode(train_text, chars), dtype=torch.long)
    val = torch.tensor(encode(val_text, chars), dtype=torch.long)
    if min(len(train), len(val)) <= context:
        raise ValueError('학습/검증 텍스트가 context보다 길어야 합니다. 데이터를 늘리거나 --context를 줄이세요.')
    return chars, train, val, lines[:cut], lines[cut:]


def batch(data, size, context, device, generator):
    starts = torch.randint(len(data) - context, (size,), generator=generator)
    x = torch.stack([data[i:i + context] for i in starts])
    y = torch.stack([data[i + 1:i + context + 1] for i in starts])
    return x.to(device), y.to(device)


@torch.no_grad()
def evaluate(model, data, size, context, device):
    model.eval()
    generator = torch.Generator().manual_seed(12345)
    losses = []
    for _ in range(10):
        x, y = batch(data, size, context, device, generator)
        losses.append(model(x, y)[1].item())
    return sum(losses) / len(losses)


def train(args):
    device = device_for(args.device)
    torch.manual_seed(args.seed)
    chars, tr, val, train_lines, val_lines = split_data(args.data, args.context, args.seed)
    encode(args.prompt, chars)
    config = Config(len(chars), args.context, args.dim, args.heads, args.layers)
    model = MiniGPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    generator = torch.Generator().manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    before = sample(model, chars, args.prompt, device)
    metrics = []
    start = time.monotonic()
    best_val = float('inf')
    best_step = 0
    for step in range(args.steps + 1):
        if step % args.eval_every == 0 or step == args.steps:
            row = {'step': step,
                   'train_loss': evaluate(model, tr, args.batch_size, args.context, device),
                   'val_loss': evaluate(model, val, args.batch_size, args.context, device)}
            metrics.append(row)
            print(json.dumps(row), flush=True)
            if row['val_loss'] < best_val:
                best_val, best_step = row['val_loss'], step
                # CPU 텐서를 저장하므로 다른 장치에서도 다시 읽을 수 있습니다.
                torch.save({'config': asdict(config), 'chars': chars, 'step': step,
                            'model': {k: v.detach().cpu() for k, v in model.state_dict().items()}},
                           args.out / 'model.pt')
        if step == args.steps:
            break
        model.train()
        x, y = batch(tr, args.batch_size, args.context, device, generator)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    saved = torch.load(args.out / 'model.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(saved['model'])
    after = sample(model, chars, args.prompt, device)
    report = {'device': device, 'torch_version': str(torch.__version__),
              'parameters': sum(p.numel() for p in model.parameters()),
              'config': asdict(config), 'steps': args.steps, 'seed': args.seed,
              'seconds': round(time.monotonic() - start, 2), 'best_step': best_step,
              'before': before, 'after': after, 'metrics': metrics,
              'train_lines': len(train_lines), 'val_lines': len(val_lines)}
    (args.out / 'metrics.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    (args.out / 'train.txt').write_text('\n'.join(train_lines) + '\n', encoding='utf-8')
    (args.out / 'validation.txt').write_text('\n'.join(val_lines) + '\n', encoding='utf-8')
    (args.out / 'samples.txt').write_text('학습 전\n' + before + '\n\n학습 후\n' + after + '\n', encoding='utf-8')
    print(f'\nparameters={report["parameters"]:,}, best_step={best_step}, seconds={report["seconds"]}')
    print('\n학습 전:\n' + before + '\n\n학습 후:\n' + after)


def load_base(path, device):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    model = MiniGPT(Config(**saved['config'])).to(device)
    model.load_state_dict(saved['model'])
    return model, saved


def load_lora(model, path, device):
    adapter = torch.load(path, map_location='cpu', weights_only=True)
    apply_lora(model, tuple(adapter['targets']), adapter['rank'], adapter['alpha'])
    model.to(device)
    missing = set(adapter['lora']) - set(model.state_dict())
    if missing:
        raise ValueError(f'어댑터와 베이스 모델 구조가 맞지 않습니다: {sorted(missing)[:3]}')
    model.load_state_dict(adapter['lora'], strict=False)
    return merge_lora(model)


def finetune(args):
    device = device_for(args.device)
    torch.manual_seed(args.seed)
    model, saved = load_base(args.base, device)
    chars, context = saved['chars'], saved['config']['context']
    _, tr, val, train_lines, val_lines = split_data(args.data, context, args.seed, chars)
    # 원래 데이터의 검증셋으로 기존 능력을 얼마나 잊는지(catastrophic forgetting) 확인합니다.
    base_val = torch.tensor(encode(args.base_validation.read_text(encoding='utf-8'), chars), dtype=torch.long)
    encode(args.prompt, chars)
    before = sample(model, chars, args.prompt, device)

    targets = tuple(args.targets.split(','))
    replaced = apply_lora(model, targets, args.rank, args.alpha, args.dropout)
    model.to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f'LoRA 적용: {replaced}')
    print(f'학습 파라미터 {trainable:,} / 전체 {total:,} ({100 * trainable / total:.2f}%)')

    # 고정된 파라미터는 옵티마이저에 넣지 않으므로 Adam 상태도 LoRA 파라미터 몫만 생깁니다.
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    generator = torch.Generator().manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    metrics = []
    start = time.monotonic()
    best_val = float('inf')
    best_step = 0
    for step in range(args.steps + 1):
        if step % args.eval_every == 0 or step == args.steps:
            row = {'step': step,
                   'train_loss': evaluate(model, tr, args.batch_size, context, device),
                   'val_loss': evaluate(model, val, args.batch_size, context, device),
                   'base_val_loss': evaluate(model, base_val, args.batch_size, context, device)}
            metrics.append(row)
            print(json.dumps(row), flush=True)
            if row['val_loss'] < best_val:
                best_val, best_step = row['val_loss'], step
                # 베이스 모델은 그대로 두고 A, B 행렬만 저장합니다.
                torch.save({'base': args.base.name, 'targets': list(targets), 'rank': args.rank,
                            'alpha': args.alpha, 'step': step, 'lora': lora_state_dict(model)},
                           args.out / 'lora.pt')
        if step == args.steps:
            break
        model.train()
        x, y = batch(tr, args.batch_size, context, device, generator)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    model, _ = load_base(args.base, device)
    model = load_lora(model, args.out / 'lora.pt', device)
    after = sample(model, chars, args.prompt, device)
    report = {'device': device, 'base': str(args.base.name), 'targets': list(targets),
              'rank': args.rank, 'alpha': args.alpha, 'trainable_parameters': trainable,
              'total_parameters': total, 'adapter_bytes': (args.out / 'lora.pt').stat().st_size,
              'steps': args.steps, 'seed': args.seed, 'seconds': round(time.monotonic() - start, 2),
              'best_step': best_step, 'before': before, 'after': after, 'metrics': metrics,
              'train_lines': len(train_lines), 'val_lines': len(val_lines)}
    (args.out / 'metrics.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    (args.out / 'samples.txt').write_text('파인튜닝 전\n' + before + '\n\n파인튜닝 후\n' + after + '\n', encoding='utf-8')
    print(f'\ntrainable={trainable:,}/{total:,}, best_step={best_step}, seconds={report["seconds"]}')
    print('\n파인튜닝 전:\n' + before + '\n\n파인튜닝 후:\n' + after)


def generate(args):
    device = device_for(args.device)
    model, saved = load_base(args.checkpoint, device)
    if args.lora:
        model = load_lora(model, args.lora, device)
    print(sample(model, saved['chars'], args.prompt, device, args.tokens,
                 args.temperature, args.top_k, args.seed))


def check():
    torch.manual_seed(7)
    model = MiniGPT(Config(20, context=16, dim=32, heads=4, layers=2)).eval()
    x = torch.randint(20, (2, 12))
    changed = x.clone()
    changed[:, 6:] = (changed[:, 6:] + 1) % 20
    a, _ = model(x)
    b, _ = model(changed)
    assert torch.allclose(a[:, :6], b[:, :6], atol=1e-6), '미래 토큰 정보 누출'
    assert a.shape == (2, 12, 20)
    _, loss = model(x[:, :-1], x[:, 1:])
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    out = model.generate(x[:1], 7, temperature=0)
    assert out.shape == (1, 19)  # context를 넘어도 sliding window로 생성

    # LoRA: B=0으로 시작하므로 적용 직후 출력이 같고, A·B만 학습되며, 병합 후에도 출력이 같아야 합니다.
    model.zero_grad(set_to_none=True)
    apply_lora(model, ('qkv', 'proj'), r=2, alpha=4)
    c, _ = model(x)
    assert torch.allclose(a, c, atol=1e-6), 'LoRA 초기 출력이 원래 모델과 다름'
    _, loss = model(x[:, :-1], x[:, 1:])
    loss.backward()
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable and all(n.endswith(('.A', '.B')) for n in trainable)
    assert all(p.grad is None for n, p in model.named_parameters() if n not in trainable)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n.endswith('.B'):
                p.normal_()
    d, _ = model.eval()(x)
    merged, _ = merge_lora(model)(x)
    assert not torch.allclose(a, d, atol=1e-3) and torch.allclose(d, merged, atol=1e-5), 'LoRA 병합 결과 불일치'
    print('PASS: causal mask, logits shape, finite gradients, generation beyond context, LoRA init/freeze/merge')


def main():
    p = argparse.ArgumentParser(description='직접 구현한 작은 Transformer 학습/생성')
    sub = p.add_subparsers(dest='command', required=True)
    t = sub.add_parser('train')
    t.add_argument('--data', type=Path, default=ROOT / 'corpus.txt')
    t.add_argument('--out', type=Path, default=ROOT / 'artifacts')
    t.add_argument('--steps', type=int, default=1200)
    t.add_argument('--eval-every', type=int, default=200)
    t.add_argument('--batch-size', type=int, default=16)
    t.add_argument('--context', type=int, default=64)
    t.add_argument('--dim', type=int, default=64)
    t.add_argument('--heads', type=int, default=4)
    t.add_argument('--layers', type=int, default=2)
    t.add_argument('--lr', type=float, default=0.001)
    t.add_argument('--prompt', default='나는 오늘 ')
    f = sub.add_parser('finetune')
    f.add_argument('--base', type=Path, default=ROOT / 'artifacts' / 'model.pt')
    f.add_argument('--base-validation', type=Path, default=ROOT / 'artifacts' / 'validation.txt')
    f.add_argument('--data', type=Path, default=ROOT / 'finetune.txt')
    f.add_argument('--out', type=Path, default=ROOT / 'artifacts' / 'lora')
    f.add_argument('--targets', default='qkv,proj', help='LoRA를 붙일 Linear 이름. 예: qkv,proj 또는 qkv,proj,0,2(FFN)')
    f.add_argument('--rank', type=int, default=4)
    f.add_argument('--alpha', type=float, default=8)
    f.add_argument('--dropout', type=float, default=0.0)
    f.add_argument('--steps', type=int, default=600)
    f.add_argument('--eval-every', type=int, default=100)
    f.add_argument('--batch-size', type=int, default=16)
    f.add_argument('--lr', type=float, default=0.005)
    f.add_argument('--prompt', default='멍개는 ')
    g = sub.add_parser('generate')
    g.add_argument('--checkpoint', type=Path, default=ROOT / 'artifacts' / 'model.pt')
    g.add_argument('--lora', type=Path, help='LoRA 어댑터 파일. 지정하면 베이스 모델에 병합해서 생성합니다.')
    g.add_argument('--prompt', default='나는 오늘 ')
    g.add_argument('--tokens', type=int, default=120)
    g.add_argument('--temperature', type=float, default=0.7)
    g.add_argument('--top-k', type=int, default=20)
    sub.add_parser('check')
    for command in (t, f, g):
        command.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto')
        command.add_argument('--seed', type=int, default=42)
        command.add_argument('--threads', type=int, default=2)
    args = p.parse_args()
    if args.command == 'check':
        torch.set_num_threads(2)
        check()
        return
    if args.threads < 1 or not args.prompt:
        p.error('threads는 양수이고 prompt는 비어 있지 않아야 합니다.')
    torch.set_num_threads(args.threads)
    if args.command == 'train':
        if any(getattr(args, key) < 1 for key in ['steps', 'eval_every', 'batch_size', 'context', 'dim', 'heads', 'layers']) or args.lr <= 0:
            p.error('학습 관련 숫자는 모두 양수여야 합니다.')
        train(args)
    elif args.command == 'finetune':
        if any(getattr(args, key) < 1 for key in ['steps', 'eval_every', 'batch_size', 'rank']) \
                or args.lr <= 0 or args.alpha <= 0 or not 0 <= args.dropout < 1:
            p.error('파인튜닝 관련 숫자는 양수여야 하고 dropout은 0 이상 1 미만이어야 합니다.')
        finetune(args)
    else:
        if args.tokens < 0 or args.temperature < 0 or args.top_k < 0:
            p.error('tokens, temperature, top-k는 0 이상이어야 합니다.')
        generate(args)


if __name__ == '__main__':
    main()
