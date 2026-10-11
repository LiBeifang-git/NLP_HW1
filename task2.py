import re

import numpy as np
import pandas as pd
from gensim.models import Word2Vec

from common import OUT, ROOT, SEED, fit_lr, load_data


def tokenize(text):
    return re.findall(r'[a-z]+', text.lower())


def mean_vectors(texts, vectors):
    rows = []
    details = []
    for text in texts:
        words = tokenize(text)
        valid = [vectors[word] for word in words if word in vectors]
        if valid:
            rows.append(np.mean(valid, axis=0))
        else:
            # 没有词被覆盖时，空列表无法求平均；使用100维零向量。
            rows.append(np.zeros(100, dtype=np.float32))
        details.append({
            'tokens': len(words),
            'known_tokens': len(valid),
            'oov_rate': 1 - len(valid) / len(words) if words else 0,
            'zero_vector': len(valid) == 0,
        })
    total = sum(row['tokens'] for row in details)
    known = sum(row['known_tokens'] for row in details)
    coverage = {
        'token_coverage': known / total,
        'zero_documents': sum(row['zero_vector'] for row in details),
    }
    return np.asarray(rows, dtype=np.float32), coverage, details


def load_glove(path, needed):
    vectors = {}
    with open(path, encoding='utf-8') as file:
        for line in file:
            word, _, values = line.partition(' ')
            if word in needed:
                vectors[word] = np.fromstring(values, sep=' ', dtype=np.float32)
    return vectors


def main():
    parts = load_data()
    for method in ['glove', 'ag', 'nyt']:
        if method == 'glove':
            # 查找公开向量不会训练模型；只加载需要的词以节省内存。
            needed = set()
            for part in parts.values():
                for text in part.text:
                    needed.update(tokenize(text))
            vectors = load_glove(ROOT / 'models/glove.6B.100d.txt', needed)
            extra = {'source': 'Stanford glove.6B.100d'}
        else:
            if method == 'ag':
                texts = pd.read_csv(ROOT / 'data/ag.csv').text
            else:
                texts = parts['train'].text
            sentences = [tokenize(text) for text in texts]
            model = Word2Vec(
                sentences=sentences, vector_size=100, window=5, min_count=2,
                workers=1, sg=1, epochs=5, seed=SEED,
            )
            vectors = model.wv
            extra = {
                'corpus_documents': len(sentences),
                'corpus_tokens': sum(map(len, sentences)),
                'vocabulary_size': len(vectors),
            }

        name = 'glove' if method == 'glove' else 'w2v_' + method
        folder = OUT / name
        folder.mkdir(parents=True, exist_ok=True)
        features = {}
        coverage = {}
        for split, part in parts.items():
            features[split], coverage[split], details = mean_vectors(part.text, vectors)
            pd.DataFrame(details).assign(row_id=part.row_id).to_csv(
                folder / f'{split}_coverage.csv', index=False
            )
        fit_lr(name, parts, features, {'dimension': 100, 'coverage': coverage, **extra})


if __name__ == '__main__':
    main()
