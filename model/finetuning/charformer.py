import torch.nn as nn
from transformers.models.electra.modeling_electra import ElectraForPreTrainingOutput, SequenceClassifierOutput
from model.charformer.electra import ElectraForPreTraining
import torch
from transformers import TrainingArguments, \
    Trainer, ElectraConfig, DataCollatorWithPadding, BatchEncoding
from datasets import DatasetDict, Dataset, load_metric
from typing import Tuple, Any, List
from utils.dataset.opinhubank.opinhubank import read_data
from model.tokenizer.charformer_tokenizer import CharformerTokenizer
from transformers.activations import get_activation
import numpy as np
import pandas as pd

_metric_accuracy = load_metric("accuracy")


class ElectraCharformerModelForFineTuning(nn.Module):
    def __init__(self, discriminator):
        super(ElectraCharformerModelForFineTuning, self).__init__()
        self.discriminator: ElectraForPreTraining = discriminator

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        output_discriminator = self.discriminator(input_ids, attention_mask, token_type_ids, labels=None)

        return ElectraForPreTrainingOutput(
            loss=None,
            logits=output_discriminator.logits,
            hidden_states=output_discriminator.hidden_states,
            attentions=output_discriminator.attentions,
        )


class ElectraClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dense.weight.data.normal_(mean=0.0, std=0.02)
        self.dense.bias.data.zero_()
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)
        self.out_proj.weight.data.normal_(mean=0.0, std=0.02)
        self.out_proj.bias.data.zero_()

    def forward(self, features, **kwargs):
        if self.config.upsample_output:
            x = features[:, 0:self.config.downsample_factor, :]
            x = torch.mean(x, dim=1)
        else:
            x = features[:, 0, :]

        x = self.dropout(x)
        x = self.dense(x)
        x = get_activation("gelu")(x)  # although BERT uses tanh here, it seems Electra authors used gelu here
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class ElectraCharformerForClassification(nn.Module):
    def __init__(self, config, discriminator):
        super(ElectraCharformerForClassification, self).__init__()
        self.config = config
        self.discriminator: ElectraCharformerModelForFineTuning = discriminator
        self.classification_head = ElectraClassificationHead(config)
        self.loss_fct = torch.nn.CrossEntropyLoss()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, **kwargs):
        output: ElectraForPreTrainingOutput = self.discriminator(input_ids, attention_mask, token_type_ids)
        logits = self.classification_head(output.hidden_states)

        loss = None
        if labels is not None:
            loss = self.loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=output.hidden_states,
            attentions=output.attentions,
        )


class DataCollator:
    def __init__(self, tokenizer, max_length=256, mlm_probability=0.15, special_token_indices=[]):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.special_token_indices = special_token_indices
        self.padding: DataCollatorWithPadding = DataCollatorWithPadding(self.tokenizer, padding='max_length',
                                                                        max_length=max_length)

    def __call__(self, examples: List[dict]):
        batch_outputs = {k: [] for k in examples[0]}
        for batch in examples:
            for key, item in batch.items():
                batch_outputs[key].append(item)
        padded: BatchEncoding = BatchEncoding(batch_outputs, tensor_type="pt")
        return padded


class ElectraCharformer:
    _dataset: DatasetDict
    _tokenized_dataset: DatasetDict

    def __init__(self, tokenizer="", data_path="", seed=0):
        self.tokenizer_name = tokenizer
        self.data_path = data_path
        self.seed = seed

    def read_data(self, max_length=1024, binary=False):
        train, test, dev = read_data(self.data_path, binary=binary)

        self._dataset = DatasetDict({
            'train': Dataset.from_dict(train),
            'validation': Dataset.from_dict(dev),
            'test': Dataset.from_dict(test)
        })

    def tokenizer(self, max_length=1024, ds_factor=4, binary=False) -> Tuple[Any, DataCollator]:
        self.read_data(max_length, binary=binary)
        tokenizer = CharformerTokenizer()

        def tokenize_function(sample):
            return tokenizer.tokenize(sample['sentence'],
                                      truncation=True, max_length=max_length, ds_factor=ds_factor)

        self._tokenized_dataset = self._dataset.map(tokenize_function, batched=True)
        data_collator = DataCollator(tokenizer=tokenizer, max_length=max_length)

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
    def freeze_model(model: ElectraCharformerForClassification, layers):
        if layers == -1:
            return
        # ElectraCharformerModelForFineTuning
        all_layer = model.discriminator.discriminator.electra.encoder.layer
        for i, layer in enumerate(all_layer):
            if i < layers:
                for params in layer.parameters():
                    params.requires_grad = False
        last_layer = model.discriminator.discriminator.electra
        if layers == 12:
            for params in last_layer.encoder_last_layer.parameters():
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

        tokenizer, data_collator = self.tokenizer(max_length=max_length, ds_factor=downsample_factor, binary=kwargs["binary"])

        vocab_size = len(tokenizer)

        _, disc_config = self.get_config(max_block_size, downsample_factor, score_consensus_attn,
                                         upsample_output, vocab_size, num_labels)

        electra_discriminator = ElectraForPreTraining(disc_config)
        discriminator = ElectraCharformerModelForFineTuning(electra_discriminator)
        discriminator.load_state_dict(torch.load(ckp, map_location=torch.device("cpu")), strict=False)

        electra_model = ElectraCharformerForClassification(disc_config, discriminator)

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
                                   weight_decay=kwargs["weight_decay"],
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
        predictions = np.argmax(results.predictions[0], axis=1)
        df = pd.DataFrame(data={
            "Predictions": predictions,
            "Labels": label_ids
        })
        return df

    def evaluate(self, trainer: Trainer) -> pd.DataFrame:
        results = trainer.predict(self._tokenized_dataset["test"])
        label_ids = results.label_ids
        predictions = np.argmax(results.predictions[0], axis=1)
        df = pd.DataFrame(data={
            "Predictions": predictions,
            "Labels": label_ids
        })
        return df


def compute_metrics(eval_data):
    logits, labels = eval_data
    pred = np.argmax(logits[0], axis=1)
    return _metric_accuracy.compute(predictions=pred, references=labels)