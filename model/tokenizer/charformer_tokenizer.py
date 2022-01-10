from typing import Dict, Union, List, Optional

import numpy as np


class CharformerTokenizer:
    def __init__(self):
        self.mappings = {
            "[PAD]": 0,
            "[MASK]": 1,
            "[S]": 2,
            "[/S]": 3,
            "[SEP]": 4,
            "[CLS]": 5,
            "[UNK]": 6,
        }
        self.mappings_revers = {self.mappings[k]: k for k in self.mappings}
        self.spacial_ids = [self.mappings[k] for k in self.mappings]
        self.spacial_tokens = [k for k in self.mappings]
        self.vocab_size = 256 + len(self.spacial_ids)
        self.mask_token_id = 1

    def __len__(self):
        return self.vocab_size

    def tokenize(self, text1, text2=None, truncation=True, max_length=1024, ds_factor=4):
        input_ids = []
        attention_mask = []
        if text2 is None:
            text2 = [''] * len(text1)
        for t1, t2 in zip(text1, text2):
            text = t1 + t2
            out = np.zeros(max_length, dtype=np.int)
            attention = np.zeros(max_length, dtype=np.bool)
            out[:ds_factor] = self.mappings['[CLS]']
            attention[:ds_factor] = True
            for i, b in enumerate(bytearray(text, encoding="utf8")):
                i = i + ds_factor
                if truncation and i == max_length:
                    break
                out[i] = b + len(self.spacial_ids)
                attention[i] = True
            input_ids.append(out)
            attention_mask.append(attention)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
