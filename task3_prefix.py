from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from transformers import AutoTokenizer

from common import ROOT, SEED, export, load_data


def main():
    parts = load_data()
    tokenizer = AutoTokenizer.from_pretrained(ROOT / 'models/bert-base-uncased')
    prefixes = {}
    for split, part in parts.items():
        encoded = tokenizer(
            part.text.tolist(), add_special_tokens=False,
            truncation=True, max_length=62, return_offsets_mapping=True,
        )

        prefixes[split] = [text[:offsets[-1][1]] for text, offsets in
                           zip(part.text, encoded['offset_mapping'])]

    for binary, name in [(True, 'prefix_binary'), (False, 'prefix_count')]:
        vectorizer = CountVectorizer(binary=binary)
        train_x = vectorizer.fit_transform(prefixes['train'])
        # 沿用全文主实验验证集选出的C，不在测试集上调参。
        model = LogisticRegression(C=0.1, max_iter=1500, tol=1e-3, random_state=SEED)
        model.fit(train_x, parts['train'].label)
        predictions = {
            split: model.predict(vectorizer.transform(prefixes[split]))
            for split in ['val', 'test']
        }
        export(name, parts, predictions, {
            'C': 0.1, 'content_wordpieces': 62,
            'vocabulary_size': len(vectorizer.vocabulary_),
        })


if __name__ == '__main__':
    main()
