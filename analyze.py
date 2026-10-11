import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import LABELS, OUT, ROOT, load_data, save_json

METHODS = ['bow_binary', 'bow_count', 'glove', 'w2v_ag', 'w2v_nyt', 'bert']
NAMES = ['Binary', 'Count', 'GloVe', 'W2V AG', 'W2V NYT', 'BERT']


def plot_confusion(name, target):
    matrix = pd.read_csv(OUT / name / 'test_confusion.csv', index_col=0).to_numpy()
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(matrix, cmap='Blues')
    ax.set_xticks(range(3), LABELS)
    ax.set_yticks(range(3), LABELS)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(name)
    for i in range(3):
        for j in range(3):
            color = 'white' if matrix[i, j] > matrix.max() / 2 else 'black'
            ax.text(j, i, str(matrix[i, j]), ha='center', va='center', color=color)
    fig.tight_layout()
    fig.savefig(target / f'{name}_confusion.png', dpi=180)
    plt.close(fig)


def main():
    target = ROOT / 'report/figures'
    target.mkdir(parents=True, exist_ok=True)
    parts = load_data()
    save_json(OUT / 'data_summary.json', {
        split: {'n': len(part), 'classes': part.label.value_counts().to_dict()}
        for split, part in parts.items()
    })
    rows = []
    for name in METHODS:
        result = json.loads((OUT / name / 'metrics.json').read_text())
        rows.append({'method': name,
                     'val_accuracy': result['val']['accuracy'],
                     'val_macro_f1': result['val']['macro_f1'],
                     'test_accuracy': result['test']['accuracy'],
                     'test_macro_f1': result['test']['macro_f1']})
        if name != 'glove':
            plot_confusion(name, target)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / 'results.csv', index=False)
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(table))
    ax.bar(x - .18, table.test_accuracy * 100, .36, label='Accuracy')
    ax.bar(x + .18, table.test_macro_f1 * 100, .36, label='Macro-F1')
    ax.set_xticks(x, NAMES)
    ax.set_ylim(0, 105)
    ax.set_ylabel('Test score (%)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(target / 'metrics.png', dpi=180)
    plt.close(fig)

    history = pd.read_json(OUT / 'bert/history.json')
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].plot(history.epoch, history.train_loss, 'o-', label='Train')
    axes[0].plot(history.epoch, history.val_loss, 'o-', label='Validation')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[1].plot(history.epoch, history.macro_f1, 'o-')
    axes[1].set_ylabel('Validation Macro-F1')
    for ax in axes:
        ax.set_xlabel('Epoch')
        ax.set_xticks(history.epoch)
    fig.tight_layout()
    fig.savefig(target / 'bert_training.png', dpi=180)
    plt.close(fig)
    if (OUT / 'training_size/summary.csv').exists():
        from training_size import plot_results
        plot_results()
    print(table.to_string(index=False))


if __name__ == '__main__':
    main()
