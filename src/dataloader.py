import torch
from torch.utils.data import DataLoader


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, input_len, text, labels=None, aug=False):
        self.text = text
        self.labels = labels
        self.aug = aug
        self.tokenizer = tokenizer
        self.input_len = input_len

    def __getitem__(self, idx):
        t = self.text[idx]
        if self.aug:
            t = augment(t)
        else:
            t = [t]

        encodings = self.tokenizer(t, padding="max_length", truncation=True, max_length=self.input_len)
        item = {key: torch.tensor(val)[0] for key, val in encodings.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx]).float()
        return item

    def __len__(self):
        return len(self.text)


def get_loader(tokenizer, input_len, x, y=None, batch_size=8, shuffle=False):
    loader = DataLoader(TrainDataset(tokenizer, input_len, x, y),
                        batch_size=batch_size, shuffle=shuffle)
    return loader
