# coding=utf-8
# Copyright 2021 rinna Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
