import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs'
SEED = 42
LABELS = ['business', 'politics', 'sports']


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, indent=2)
    path.write_text(text, encoding='utf-8')


def load_data():
    df = pd.read_csv(ROOT / 'data/nyt.csv')
    df['row_id'] = np.arange(len(df))
    df = df.dropna(subset=['text', 'label']).reset_index(drop=True)

    train, remaining = train_test_split(
        np.arange(len(df)), test_size=0.2, random_state=SEED, stratify=df.label
    )
    val, test = train_test_split(
        remaining, test_size=0.5, random_state=SEED, stratify=df.iloc[remaining].label
    )
    parts = {}
    for name, indices in [('train', train), ('val', val), ('test', test)]:
        parts[name] = df.iloc[indices].reset_index(drop=True)

    save_json(OUT / 'split.json', {
        'seed': SEED,
        'row_ids': {name: part.row_id.tolist() for name, part in parts.items()},
    })
    return parts


def metrics(y, pred):
    return {
        'accuracy': accuracy_score(y, pred),
        'macro_f1': f1_score(y, pred, labels=LABELS, average='macro', zero_division=0),
    }


def export(name, parts, predictions, metadata):
    folder = OUT / name
    folder.mkdir(parents=True, exist_ok=True)
    result = {'method': name, **metadata}

    for split, pred in predictions.items():
        part = parts[split]
        result[split] = metrics(part.label, pred)
        table = part[['row_id', 'text', 'label']].copy()
        table['prediction'] = pred
        table['correct'] = table.label == table.prediction
        table['word_count'] = table.text.str.split().str.len()
        table.to_csv(folder / f'{split}_predictions.csv', index=False)

        report = classification_report(
            part.label, pred, labels=LABELS, output_dict=True, zero_division=0
        )
        pd.DataFrame(report).T.to_csv(folder / f'{split}_per_class.csv')
        matrix = confusion_matrix(part.label, pred, labels=LABELS)
        pd.DataFrame(matrix, index=LABELS, columns=LABELS).to_csv(
            folder / f'{split}_confusion.csv'
        )

    save_json(folder / 'metrics.json', result)
    print(name, result['test'], flush=True)
    return result


def fit_lr(name, parts, features, metadata):
    history = []
    best_score = -1
    for c in [0.1, 1.0, 10.0]:
        model = LogisticRegression(C=c, max_iter=1500, tol=1e-3, random_state=SEED)
        model.fit(features['train'], parts['train'].label)
        scores = metrics(parts['val'].label, model.predict(features['val']))
        history.append({'C': c, **scores})
        if scores['macro_f1'] > best_score:
            best_model = model
            best_score = scores['macro_f1']

    predictions = {}
    for split in ['val', 'test']:
        predictions[split] = best_model.predict(features[split])
    result = export(name, parts, predictions, {
        'selected_C': best_model.C,
        'validation_search': history,
        **metadata,
    })
    return best_model, result
