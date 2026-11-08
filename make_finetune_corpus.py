"""LoRA 파인튜닝용 문장을 생성합니다.

사전학습 corpus.txt는 한 줄에 한 문장입니다. 여기서는 '멍개는'으로 시작하는 줄이
'아침에 ... 저녁에 ...' 두 문장으로 이어지는 새 패턴을 가르칩니다.
새 문자를 쓰면 기존 문자 어휘로 인코딩할 수 없으므로 corpus.txt에 있는 문자만 사용합니다.
"""
from itertools import permutations
from pathlib import Path

# make_corpus.py를 import하면 corpus.txt를 다시 쓰므로 같은 목록을 그대로 둡니다.
# 제공된 artifacts/model.pt는 '잠을 잔다.'가 추가되기 전에 학습되어 '잠', '잔'이 어휘에 없습니다.
actions = [
  '책을 읽는다.',
  '코드를 작성한다.',
  '모델을 학습한다.',
  '문장을 생성한다.',
  '컴퓨터를 공부한다.',
  '운동을 한다.',
  '음악을 듣는다.',
  '문제를 해결한다.',
  '데이터를 확인한다.',
  '블로그를 작성한다.',
]

text = '\n'.join(f'멍개는 아침에 {a} 저녁에 {b}' for a, b in permutations(actions, 2)) + '\n'
path = Path(__file__).with_name('finetune.txt')
path.write_text(text, encoding='utf-8')
print(f'{path.name}: {len(text)} characters, {len(text.splitlines())} sentences')
