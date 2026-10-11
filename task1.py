import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from common import OUT, fit_lr, load_data


def main():
    parts = load_data()
    methods = [
        ('bow_binary', CountVectorizer(binary=True)),
        ('bow_count', CountVectorizer()),
    ]
    for name, vectorizer in methods:
        features = {'train': vectorizer.fit_transform(parts['train'].text)}
        for split in ['val', 'test']:
            features[split] = vectorizer.transform(parts[split].text)
        model, _ = fit_lr(name, parts, features, {
            'vocabulary_size': len(vectorizer.vocabulary_),
        })

        terms = vectorizer.get_feature_names_out()
        rows = []
        for label, weights in zip(model.classes_, model.coef_):
            for i in weights.argsort()[-25:][::-1]:
                rows.append({'label': label, 'word': terms[i], 'coefficient': weights[i]})
        pd.DataFrame(rows).to_csv(OUT / name / 'top_words.csv', index=False)


if __name__ == '__main__':
    main()
