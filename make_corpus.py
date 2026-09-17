"""외부 데이터 없이 실습용 한국어 문장을 생성합니다."""
from itertools import product
from pathlib import Path

subjects = ['나는', '우리는', '멍개는', '친구는', '학생은', '개발자는']
times = ['오늘', '내일', '아침에', '저녁에', '주말에', '매일']
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
  '잠을 잔다.'
]

text = '\n'.join(f'{s} {t} {a}' for s, t, a in product(subjects, times, actions)) + '\n'
path = Path(__file__).with_name('corpus.txt')
path.write_text(text, encoding='utf-8')
print(f'{path.name}: {len(text)} characters, {len(text.splitlines())} sentences')
