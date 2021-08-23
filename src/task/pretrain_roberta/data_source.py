import itertools

import torch


def collate_fn(batch_data):
    return batch_data


class DataSource(torch.utils.data.Dataset):

    def __init__(self, config, tokenizer, docs, stage):
        # Attributes
        self.max_seq_len = config.max_seq_len
        # Other attributes
        self.tokenizer = tokenizer
        self.stage = stage
        self.statistics = {"n_docs": 0, "n_sents": 0, "n_tokens": 0}

        # Load dataset
        self.docs = docs

        # Calculate basic statistics
        self.statistics["n_docs"] = len(self.docs)
        for doc in self.docs:
            self.statistics["n_sents"] += len(doc)
            for sent in doc:
                self.statistics["n_tokens"] += len(sent)

    def __len__(self):
        return len(self.docs)

    def __getitem__(self, idx):
        doc = self.docs[idx]
        
        seq = list(itertools.chain(*doc))
        seq = [self.tokenizer.cls_token_id] + seq[:self.max_seq_len - 1]

        return seq
