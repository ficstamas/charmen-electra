import torch
from transformers import AutoTokenizer, TrainingArguments,  ElectraForPreTraining,\
    Trainer, ElectraForMaskedLM, ElectraConfig, DataCollatorWithPadding, BatchEncoding, get_linear_schedule_with_warmup
from datasets import load_metric, IterableDatasetDict, DatasetDict, Dataset
from typing import Tuple, Any
from utils.dataset.webcorpus.hu import Webcorpus, WebcorpusInMemory
from utils.dataset.utils import construct_paths, train_test_split
import torch.nn as nn
from typing import List
from transformers.models.electra.modeling_electra import ElectraForPreTrainingOutput
from torch.optim import Adam
import numpy as np
from scipy.special import expit


__all__ = ["Electra", "ElectraModel"]


_metric_accuracy = load_metric("accuracy")
_metric_f1 = load_metric("f1")
_metric_precision = load_metric("precision")
_metric_recall = load_metric("recall")


class DataCollator:
    def __init__(self, tokenizer, max_length=256, mlm_probability=0.15, special_token_indices=[]):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.special_token_indices = special_token_indices
        self.padding: DataCollatorWithPadding = DataCollatorWithPadding(self.tokenizer, padding='max_length',
                                                                        max_length=max_length)

    def __call__(self, examples: List[dict]):
        padded: BatchEncoding = self.padding(examples)
        padded.data['labels'] = padded.data['input_ids'].clone()
        padded.data['labels'][torch.where(padded.data['attention_mask'] == 0)] = -100

        inputs, labels, mlm_mask = self.mask(padded.data['input_ids'],
                                             mlm_probability=self.mlm_probability,
                                             special_token_indices=self.special_token_indices,
                                             mask_token_index=self.tokenizer.mask_token_id,
                                             vocab_size=len(self.tokenizer))
        padded.data['input_ids'] = inputs
        padded.data['labels'] = labels
        padded.data['mlm_mask'] = mlm_mask
        return padded

    @staticmethod
    def mask(inputs, mlm_probability=0.15, special_token_indices=[],
             replace_prob=0.1, original_prob=0.1, ignore_index=-100, vocab_size=32001, mask_token_index=2):

        labels = inputs.clone()

        probability_matrix = torch.full(labels.shape, mlm_probability)
        special_tokens_mask = torch.full(inputs.shape, False, dtype=torch.bool)
        for sp_id in special_token_indices:
            special_tokens_mask = special_tokens_mask | (inputs == sp_id)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        mlm_mask = torch.bernoulli(probability_matrix).bool()

        labels[~mlm_mask] = ignore_index  # We only compute loss on mlm applied tokens

        # mask  (mlm_probability * (1-replace_prob-orginal_prob))
        mask_prob = 1 - replace_prob - original_prob
        mask_token_mask = torch.bernoulli(torch.full(labels.shape, mask_prob)).bool() & mlm_mask
        inputs[mask_token_mask] = mask_token_index

        # replace with a random token (mlm_probability * replace_prob)
        if int(replace_prob) != 0:
            rep_prob = replace_prob / (replace_prob + original_prob)
            replace_token_mask = torch.bernoulli(
                torch.full(labels.shape, rep_prob)).bool() & mlm_mask & ~mask_token_mask
            random_words = torch.randint(vocab_size, labels.shape, dtype=torch.long)
            inputs[replace_token_mask] = random_words[replace_token_mask]

        return inputs, labels, mlm_mask


class ElectraModel(nn.Module):
    def __init__(self, generator, discriminator, loss_weights=(1, 50)):
        super(ElectraModel, self).__init__()
        self.generator: ElectraForMaskedLM = generator
        self.discriminator: ElectraForPreTraining = discriminator
        self.generator_loss_fct = nn.CrossEntropyLoss()
        self.gumbel_dist = torch.distributions.gumbel.Gumbel(0., 1.)
        self.loss_weights = loss_weights

    def forward(self, input_ids, attention_mask, token_type_ids, labels, mlm_mask):
        output = self.generator(input_ids, attention_mask, token_type_ids, labels=labels)
        generator_logits = output.logits[mlm_mask, :]

        with torch.no_grad():
            generator_tokens = self.sample(generator_logits)
            discriminator_input = input_ids.clone()
            discriminator_input[mlm_mask] = generator_tokens

            is_replaced = mlm_mask.clone()
            is_replaced[mlm_mask] = (generator_tokens != labels[mlm_mask])

        output_discriminator = self.discriminator(discriminator_input, attention_mask, token_type_ids, labels=is_replaced)
        generator_loss = self.generator_loss_fct(generator_logits[is_replaced[mlm_mask], :], labels[is_replaced])

        return ElectraForPreTrainingOutput(
            loss=output_discriminator.loss*self.loss_weights[1] + generator_loss*self.loss_weights[0],
            logits=output_discriminator.logits,
            hidden_states=output_discriminator.hidden_states,
            attentions=output_discriminator.attentions,
        )

    def sample(self, logits, sampling="fp32_gumbel"):
        """Reimplement gumbel softmax cuz there is a bug in torch.nn.functional.gumbel_softmax when fp16
        (https://github.com/pytorch/pytorch/issues/41663).
        Gumbel softmax is equal to what official ELECTRA code do, standard gumbel dist.
        = -ln(-ln(standard uniform dist.))
        """
        if sampling == 'fp32_gumbel':
            gumbel = self.gumbel_dist.sample(logits.shape).to(logits.device)
            return (logits.float() + gumbel).argmax(dim=-1)
        elif sampling == 'fp16_gumbel':  # 5.06 ms
            gumbel = self.gumbel_dist.sample(logits.shape).to(logits.device)
            return (logits + gumbel).argmax(dim=-1)
        elif sampling == 'multinomial':  # 2.X ms
            return torch.multinomial(F.softmax(logits, dim=-1), 1).squeeze()


class Electra:
    _dataset: DatasetDict
    _tokenized_dataset: DatasetDict

    def __init__(self, tokenizer="SZTAKI-HLT/hubert-base-cc", data_path="", seed=0, data_split=0.8):
        self.tokenizer_name = tokenizer
        self.data_path = data_path
        self.seed = seed
        self.data_split = data_split

    def read_data(self):
        train_data, test_data = train_test_split(construct_paths(self.data_path, "wiki*"),
                                                 self.seed,
                                                 p=self.data_split)
        train = WebcorpusInMemory(train_data)
        # test = WebcorpusInMemory(test_data)

        self._dataset = DatasetDict({
            'train': Dataset.from_dict(train.__dict__()),
            # 'validation': Dataset.from_dict(test.__dict__())
        })

    def tokenizer(self) -> Tuple[Any, DataCollator]:
        max_length = 256
        self.read_data()
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)

        def tokenize_function(sample):
            return tokenizer(sample['sentence1'], sample['sentence2'], truncation=True, max_length=max_length)

        self._tokenized_dataset = self._dataset.map(tokenize_function, batched=True)
        data_collator = DataCollator(tokenizer=tokenizer, max_length=max_length, mlm_probability=0.15,
                                     special_token_indices=tokenizer.all_special_ids)

        return tokenizer, data_collator

    def make_model(self, output_path, local_rank, batch_size=8) -> Trainer:
        tokenizer, data_collator = self.tokenizer()

        disc_config = ElectraConfig.from_pretrained(f'google/electra-base-discriminator')
        gen_config = ElectraConfig.from_pretrained(f'google/electra-base-generator')
        gen_config.vocab_size = len(tokenizer)

        electra_generator = ElectraForMaskedLM(gen_config)
        electra_discriminator = ElectraForPreTraining(disc_config)
        electra_discriminator.electra.embeddings = electra_generator.electra.embeddings
        electra_generator.generator_lm_head.weight = electra_generator.electra.embeddings.word_embeddings.weight

        electra_model = ElectraModel(electra_generator, electra_discriminator)

        # optimizer = Adam(electra_model.parameters(), lr=2e-4, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.01)
        # scheduler = get_linear_schedule_with_warmup(optimizer, 15000, 500000)
        config = TrainingArguments(output_path,
                                   # fp16=True,
                                   fp16=False,  # no_cuda=True,
                                   overwrite_output_dir=True,
                                   max_steps=500_000,
                                   warmup_steps=15_000,
                                   save_steps=25_000,
                                   eval_steps=25_000,
                                   dataloader_drop_last=True,
                                   per_device_train_batch_size=batch_size,
                                   seed=0,
                                   evaluation_strategy="steps",
                                   max_grad_norm=1,
                                   label_names=["attention_mask", "mlm_mask"],
                                   dataloader_num_workers=0,
                                   logging_steps=500,
                                   logging_strategy="steps",
                                   log_level="info",
                                   adam_beta1=0.9,
                                   adam_beta2=0.999,
                                   adam_epsilon=1e-6,
                                   weight_decay=0.01,
                                   learning_rate=8e-4,
                                   lr_scheduler_type='linear'
                                   # ignore_data_skip=True
                                   # debug="underflow_overflow",
                                   # local_rank=local_rank,
                                   # sharded_ddp="zero_dp_2",
                                   )

        trainer = Trainer(
            electra_model,
            config,
            train_dataset=self._tokenized_dataset["train"],
            # eval_dataset=self._tokenized_dataset["validation"],
            data_collator=data_collator,
            tokenizer=tokenizer,
            # compute_metrics=Electra.compute_metrics,
            # optimizers=(optimizer, scheduler)
        )
        return trainer

    @staticmethod
    def compute_metrics(eval_data):
        logits, labels = eval_data
        preds = logits
        preds = np.round(expit(preds))
        # preds[preds >= 0] = 1
        # preds[preds < 0] = 0
        mask = labels[0] == 1
        preds = preds[mask]
        labels = labels[1][mask].astype(preds.dtype)

        bpreds = preds == 1
        blabels = labels == 1
        # _metric_f1.compute(predictions=preds, references=labels, average='binary', labels=[0, 1])
        out = _metric_accuracy.compute(predictions=preds, references=labels)
        out.update(_metric_f1.compute(predictions=bpreds, references=blabels, average='binary'))
        out.update(_metric_precision.compute(predictions=bpreds, references=blabels, average='binary'))
        out.update(_metric_recall.compute(predictions=bpreds, references=blabels, average='binary'))
        return out
