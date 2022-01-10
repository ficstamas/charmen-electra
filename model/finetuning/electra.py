import torch
import torch.nn as nn
from transformers.models.electra.modeling_electra import ElectraForSequenceClassification
import torch
from transformers import TrainingArguments, Trainer, ElectraConfig, DataCollatorWithPadding, BatchEncoding, AutoTokenizer, BertTokenizerFast
from datasets import DatasetDict, Dataset, load_metric
from typing import Tuple, Any, List
from utils.dataset.opinhubank.opinhubank import read_data
import numpy as np
import pandas as pd

_metric_accuracy = load_metric("accuracy")


class DataCollator:
    def __init__(self, tokenizer, max_length=256, mlm_probability=0.15, special_token_indices=[]):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.special_token_indices = special_token_indices
        self.padding: DataCollatorWithPadding = DataCollatorWithPadding(self.tokenizer, padding='max_length',
                                                                        max_length=max_length)

    def __call__(self, examples: List[dict]):
        padded = self.padding(examples)
        return padded


class Electra:
    _dataset: DatasetDict
    _tokenized_dataset: DatasetDict

    def __init__(self, tokenizer="SZTAKI-HLT/hubert-base-cc", data_path="", seed=0):
        self.tokenizer_name = tokenizer
        self.data_path = data_path
        self.seed = seed

    def read_data(self, max_length=256, binary=False):
        train, test, dev = read_data(self.data_path, binary=binary)

        self._dataset = DatasetDict({
            'train': Dataset.from_dict(train),
            'test': Dataset.from_dict(test),
            'validation': Dataset.from_dict(dev),
        })

    def tokenizer(self, max_length=256, binary=False) -> Tuple[Any, DataCollator]:
        self.read_data(max_length, binary=binary)
        tokenizer: BertTokenizerFast = AutoTokenizer.from_pretrained(self.tokenizer_name)

        def tokenize_function(sample):
            # return tokenizer.tokenize(sample['sentence'], truncation=True, max_length=max_length)
            return tokenizer.batch_encode_plus(sample['sentence'], truncation=True, max_length=max_length)

        self._tokenized_dataset = self._dataset.map(tokenize_function, batched=True)
        data_collator = DataCollator(tokenizer=tokenizer, max_length=max_length, special_token_indices=tokenizer.all_special_ids)

        return tokenizer, data_collator

    def get_config(self, max_block_size, downsample_factor, score_consensus_attn, upsample_output,
                   vocab_size, num_labels):
        disc_config = ElectraConfig.from_pretrained(f'google/electra-base-discriminator')
        gen_config = ElectraConfig.from_pretrained(f'google/electra-base-generator')

        gen_config.max_block_size = max_block_size
        gen_config.downsample_factor = downsample_factor
        gen_config.score_consensus_attn = score_consensus_attn
        gen_config.upsample_output = upsample_output

        disc_config.max_block_size = max_block_size
        disc_config.downsample_factor = downsample_factor
        disc_config.score_consensus_attn = score_consensus_attn
        disc_config.upsample_output = upsample_output

        disc_config.num_labels = num_labels
        disc_config.vocab_size = vocab_size

        gen_config.vocab_size = vocab_size
        return gen_config, disc_config

    @staticmethod
    def freeze_model(model: ElectraForSequenceClassification, layers):
        if layers == -1:
            return
        all_layer = model.electra.encoder.layer
        for i, layer in enumerate(all_layer):
            if i < layers:
                for params in layer.parameters():
                    params.requires_grad = False

    def make_model(self, ckp, local_rank, batch_size=8, **kwargs) -> Trainer:
        max_block_size = kwargs["max_block_size"]
        downsample_factor = kwargs["downsample_factor"]
        score_consensus_attn = kwargs["score_consensus_attn"]
        upsample_output = kwargs["upsample_output"]
        max_length = kwargs["max_length"]
        num_labels = 3
        if kwargs["binary"]:
            num_labels = 2

        tokenizer, data_collator = self.tokenizer(max_length=max_length, binary=kwargs["binary"])

        vocab_size = len(tokenizer)

        _, disc_config = self.get_config(max_block_size, downsample_factor, score_consensus_attn,
                                         upsample_output, vocab_size, num_labels)

        electra_model = ElectraForSequenceClassification(disc_config)
        electra_model.load_state_dict(torch.load(ckp, map_location=torch.device("cpu")), strict=False)

        self.freeze_model(electra_model, kwargs["freeze"])

        config = TrainingArguments(kwargs["output"],
                                   # fp16=True,
                                   fp16=False,  # no_cuda=True,
                                   overwrite_output_dir=True,
                                   dataloader_drop_last=True,
                                   per_device_train_batch_size=batch_size,
                                   seed=0,
                                   evaluation_strategy="epoch",
                                   save_strategy="epoch",
                                   max_grad_norm=1,
                                   dataloader_num_workers=0,
                                   logging_steps=100,
                                   logging_strategy="steps",
                                   log_level="info",
                                   adam_beta1=0.9,
                                   adam_beta2=0.999,
                                   adam_epsilon=kwargs["adam_eps"],
                                   weight_decay=0.01,
                                   learning_rate=kwargs["lr"],
                                   num_train_epochs=kwargs["num_epochs"],
                                   label_names=["labels"],
                                   eval_accumulation_steps=32,
                                   # ignore_data_skip=True
                                   # debug="underflow_overflow",
                                   # local_rank=local_rank,
                                   # sharded_ddp="zero_dp_2",
                                   )

        trainer = Trainer(
            electra_model,
            config,
            train_dataset=self._tokenized_dataset["train"],
            eval_dataset=self._tokenized_dataset["validation"],
            data_collator=data_collator,
            # tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )
        return trainer

    def validation(self, trainer: Trainer) -> pd.DataFrame:
        results = trainer.predict(self._tokenized_dataset["validation"])
        label_ids = results.label_ids
        predictions = np.argmax(results.predictions, axis=1)
        df = pd.DataFrame(data={
            "Predictions": predictions,
            "Labels": label_ids
        })
        return df

    def evaluate(self, trainer: Trainer) -> pd.DataFrame:
        results = trainer.predict(self._tokenized_dataset["test"])
        label_ids = results.label_ids
        predictions = np.argmax(results.predictions, axis=1)
        df = pd.DataFrame(data={
            "Predictions": predictions,
            "Labels": label_ids
        })
        return df


def compute_metrics(eval_data):
    logits, labels = eval_data
    pred = np.argmax(logits, axis=1)
    return _metric_accuracy.compute(predictions=pred, references=labels)