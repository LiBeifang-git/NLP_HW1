import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import get_linear_schedule_with_warmup

from common import LABELS, OUT, ROOT, SEED, export, load_data, metrics, save_json

MAX_LENGTH = 64
EPOCHS = 3
BATCH_SIZE = 16


class TextDataset(Dataset):
    def __init__(self, part, tokenizer):
        self.encoded = tokenizer(
            part.text.tolist(), truncation=True, padding='max_length', max_length=MAX_LENGTH
        )
        self.labels = [LABELS.index(label) for label in part.label]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        item = {}
        for key, values in self.encoded.items():
            item[key] = torch.tensor(values[index])
        item['labels'] = torch.tensor(self.labels[index])
        return item


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    predictions = []
    total_loss = 0
    count = 0
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        output = model(**batch)
        size = len(batch['labels'])
        total_loss += output.loss.item() * size
        count += size
        class_ids = output.logits.argmax(dim=-1).cpu().tolist()
        predictions.extend(LABELS[i] for i in class_ids)
    return predictions, total_loss / count


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    parts = load_data()
    folder = OUT / 'bert'
    folder.mkdir(parents=True, exist_ok=True)
    source = ROOT / 'models/bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(source)
    model = AutoModelForSequenceClassification.from_pretrained(
        source, num_labels=3, id2label=dict(enumerate(LABELS)),
        label2id={label: i for i, label in enumerate(LABELS)},
    ).to(device)
    loaders = {}
    for split, part in parts.items():
        dataset = TextDataset(part, tokenizer)
        loaders[split] = DataLoader(
            dataset, batch_size=BATCH_SIZE, shuffle=(split == 'train')
        )

        encoded = tokenizer(part.text.tolist(), truncation=False, verbose=False)
        lengths = [len(ids) for ids in encoded['input_ids']]
        pd.DataFrame({
            'row_id': part.row_id,
            'wordpiece_length': lengths,
            'truncated': np.array(lengths) > MAX_LENGTH,
        }).to_csv(folder / f'{split}_lengths.csv', index=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    total_steps = EPOCHS * len(loaders['train'])
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(0.1 * total_steps), total_steps
    )
    history = []
    best_f1 = -1
    print(f'device={device}, epochs={EPOCHS}', flush=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        loss_sum = 0
        count = 0
        for step, batch in enumerate(loaders['train'], start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            output = model(**batch)
            output.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            size = len(batch['labels'])
            loss_sum += output.loss.item() * size
            count += size
            if step % 50 == 0:
                print(f'epoch={epoch} step={step} loss={loss_sum / count:.4f}', flush=True)

        pred, val_loss = evaluate(model, loaders['val'], device)
        scores = metrics(parts['val'].label, pred)
        history.append({
            'epoch': epoch, 'train_loss': loss_sum / count, 'val_loss': val_loss, **scores,
        })
        save_json(folder / 'history.json', history)
        if scores['macro_f1'] > best_f1:
            best_f1 = scores['macro_f1']
            best_epoch = epoch
            model.save_pretrained(folder / 'best_model')
            tokenizer.save_pretrained(folder / 'best_model')

    del model, optimizer, scheduler
    model = AutoModelForSequenceClassification.from_pretrained(folder / 'best_model').to(device)
    predictions = {}
    for split in ['val', 'test']:
        predictions[split], _ = evaluate(model, loaders[split], device)
    export('bert', parts, predictions, {
        'pretrained_model': 'google-bert/bert-base-uncased',
        'max_length': MAX_LENGTH, 'epochs': EPOCHS, 'selected_epoch': best_epoch,
        'batch_size': BATCH_SIZE, 'learning_rate': 2e-5, 'weight_decay': 0.01,
        'warmup_ratio': 0.1, 'seed': SEED, 'device': device,
    })


if __name__ == '__main__':
    main()
