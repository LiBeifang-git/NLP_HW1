import platform

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from common import ROOT, OUT, SEED, load_data, metrics, save_json
from task2 import load_glove, mean_vectors, tokenize

FRACTIONS = [0.1, 0.25, 0.5, 1.0]
SAMPLE_SEEDS = [42, 43, 44]
TARGET = OUT / 'training_size'


def plot_results():
    table = pd.read_csv(TARGET / 'summary.csv')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for method, label in [('bow_count', 'Word Frequency'), ('glove', 'GloVe mean')]:
        rows = table[table.method == method]
        for ax, score in zip(axes, ['accuracy', 'macro_f1']):
            ax.errorbar(rows.fraction * 100, rows[score + '_mean'] * 100,
                        yerr=rows[score + '_std'] * 100, marker='o', capsize=4, label=label)
    for ax, label in zip(axes, ['Test Accuracy (%)', 'Test Macro-F1 (%)']):
        ax.set_xlabel('Training data used (%)')
        ax.set_ylabel(label)
        ax.set_xticks([10, 25, 50, 100])
        ax.grid(alpha=.25)
        ax.legend()
    fig.tight_layout()
    (ROOT / 'report/figures').mkdir(parents=True, exist_ok=True)
    fig.savefig(ROOT / 'report/figures/training_size.png', dpi=200)
    plt.close(fig)


def main():
    parts = load_data()
    TARGET.mkdir(parents=True, exist_ok=True)
    # 固定预训练向量不依赖本地样本量，可预先计算后按训练子集取行。
    needed = {w for part in parts.values() for text in part.text for w in tokenize(text)}
    vectors = load_glove(ROOT / 'models/glove.6B.100d.txt', needed)
    glove = {split: mean_vectors(part.text, vectors)[0] for split, part in parts.items()}
    runs, searches, subsets, predictions = [], [], {}, []
    for fraction in FRACTIONS:
        # 100%时没有抽样变化，只运行一次，不把相同结果当作三次独立重复。
        seeds = SAMPLE_SEEDS if fraction < 1 else [SEED]
        for seed in seeds:
            if fraction == 1:
                indices = np.arange(len(parts['train']))
            else:
                indices, _ = train_test_split(
                    np.arange(len(parts['train'])), train_size=fraction,
                    random_state=seed, stratify=parts['train'].label,
                )
                indices.sort()
            subset = parts['train'].iloc[indices]
            run_id = f'{int(fraction * 100)}_{seed}'
            subsets[run_id] = subset.row_id.tolist()
            vectorizer = CountVectorizer()
            bow = {'train': vectorizer.fit_transform(subset.text)}
            for split in ['val', 'test']:
                bow[split] = vectorizer.transform(parts[split].text)
            for method, features in [('bow_count', bow), ('glove', {
                'train': glove['train'][indices], 'val': glove['val'], 'test': glove['test'],
            })]:
                best_f1 = -1
                for c in [0.1, 1.0, 10.0]:
                    model = LogisticRegression(C=c, max_iter=1500, tol=1e-3, random_state=SEED)
                    model.fit(features['train'], subset.label)
                    val = metrics(parts['val'].label, model.predict(features['val']))
                    searches.append({'method': method, 'fraction': fraction, 'seed': seed, 'C': c, **val})
                    if val['macro_f1'] > best_f1:
                        best_model, best_f1 = model, val['macro_f1']
                pred = best_model.predict(features['test'])
                scores = metrics(parts['test'].label, pred)
                row = {'method': method, 'fraction': fraction, 'seed': seed,
                       'train_n': len(subset), 'C': best_model.C, 'val_macro_f1': best_f1, **scores}
                runs.append(row)
                predictions.extend({'method': method, 'fraction': fraction, 'seed': seed,
                                    'row_id': int(row_id), 'label': label, 'prediction': prediction}
                                   for row_id, label, prediction in zip(parts['test'].row_id, parts['test'].label, pred))
                print(row, flush=True)
    table = pd.DataFrame(runs)
    table.to_csv(TARGET / 'runs.csv', index=False)
    pd.DataFrame(searches).to_csv(TARGET / 'validation_search.csv', index=False)
    pd.DataFrame(predictions).to_csv(TARGET / 'test_predictions.csv', index=False)
    summary = table.groupby(['method', 'fraction'], as_index=False).agg(
        train_n=('train_n', 'first'), repeats=('seed', 'count'),
        accuracy_mean=('accuracy', 'mean'), accuracy_std=('accuracy', 'std'),
        macro_f1_mean=('macro_f1', 'mean'), macro_f1_std=('macro_f1', 'std'),
    )
    # 全量点只运行一次；图中不画误差条，不能解读为总体没有不确定性。
    summary[['accuracy_std', 'macro_f1_std']] = summary[['accuracy_std', 'macro_f1_std']].fillna(0)
    summary.to_csv(TARGET / 'summary.csv', index=False)
    save_json(TARGET / 'subsets.json', subsets)
    save_json(TARGET / 'protocol.json', {
        'fractions': FRACTIONS, 'sampling_seeds': SAMPLE_SEEDS, 'split_seed': SEED,
        'full_data_runs': 1, 'std': 'sample standard deviation; full-data error bar omitted',
        'selection': 'C selected on fixed validation set; test never used for selection',
        'python': platform.python_version(), 'numpy': np.__version__, 'sklearn': sklearn.__version__,
    })
    plot_results()


if __name__ == '__main__':
    main()
