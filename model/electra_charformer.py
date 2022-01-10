import torch
from transformers import TrainingArguments,  \
    Trainer, ElectraConfig, DataCollatorWithPadding, BatchEncoding
from datasets import DatasetDict, Dataset
from typing import Tuple, Any
from utils.dataset.webcorpus.hu import WebcorpusInMemoryCharformer
from utils.dataset.utils import construct_paths, train_test_split
from typing import List
from .electra import ElectraModel, Electra
from .tokenizer.charformer_tokenizer import CharformerTokenizer
from transformers.models.electra.modeling_electra import ElectraForPreTrainingOutput
from .charformer.electra import ElectraForPreTraining, ElectraForMaskedLM


__all__ = ["ElectraCharformer"]


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


class ElectraCharformerModel(ElectraModel):
    def __init__(self, generator, discriminator, loss_weights=(1, 50)):
        super(ElectraCharformerModel, self).__init__(generator, discriminator, loss_weights)
        self.discriminator_loss_fct = torch.nn.BCEWithLogitsLoss()

    def forward(self, input_ids, attention_mask, labels, mlm_mask, token_type_ids=None):
        output = self.generator(input_ids, attention_mask, token_type_ids, labels=labels)
        generator_logits = output.logits[mlm_mask, :]

        with torch.no_grad():
            generator_tokens = self.sample(generator_logits)
            discriminator_input = input_ids.clone()
            discriminator_input[mlm_mask] = generator_tokens

            is_replaced = mlm_mask.clone()
            is_replaced[mlm_mask] = (generator_tokens != labels[mlm_mask])

        output_discriminator = self.discriminator(discriminator_input, attention_mask, token_type_ids,
                                                  labels=is_replaced)

        generator_loss = self.generator_loss_fct(generator_logits[is_replaced[mlm_mask], :], labels[is_replaced])

        active_loss = attention_mask.view(-1, output_discriminator.hidden_states.shape[1]) == 1
        active_logits = output_discriminator.logits.view(-1, output_discriminator.hidden_states.shape[1])[active_loss]
        active_labels = is_replaced[active_loss]
        discriminator_loss = self.discriminator_loss_fct(active_logits, active_labels.float())

        return ElectraForPreTrainingOutput(
            loss=discriminator_loss*self.loss_weights[1] + generator_loss*self.loss_weights[0],
            logits=output_discriminator.logits,
            hidden_states=output_discriminator.hidden_states,
            attentions=output_discriminator.attentions,
        )


class ElectraCharformer:
    _dataset: DatasetDict
    _tokenized_dataset: DatasetDict

    def __init__(self, tokenizer="", data_path="", seed=0):
        self.tokenizer_name = tokenizer
        self.data_path = data_path
        self.seed = seed

    def read_data(self, max_length=1024):
        train_data, test_data = train_test_split(construct_paths(self.data_path, "wiki*"),
                                                 self.seed,
                                                 p=0.8)
        train = WebcorpusInMemoryCharformer(train_data, max_length=max_length)
        test = WebcorpusInMemoryCharformer(test_data)

        self._dataset = DatasetDict({
            'train': Dataset.from_dict(train.__dict__()),
            'validation': Dataset.from_dict(test.__dict__())
        })

    def tokenizer(self, max_length=1024, ds_factor=4) -> Tuple[Any, DataCollator]:
        self.read_data(max_length)
        tokenizer = CharformerTokenizer()

        def tokenize_function(sample):
            return tokenizer.tokenize(sample['sentence1'], sample['sentence2'],
                                      truncation=True, max_length=max_length, ds_factor=ds_factor)

        self._tokenized_dataset = self._dataset.map(tokenize_function, batched=True)
        data_collator = DataCollator(tokenizer=tokenizer, max_length=max_length, mlm_probability=0.15,
                                     special_token_indices=tokenizer.spacial_ids)

        return tokenizer, data_collator

    def get_config(self, max_block_size, downsample_factor, score_consensus_attn, upsample_output,
                   vocab_size):
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

        gen_config.vocab_size = vocab_size
        return gen_config, disc_config

    def make_model(self, output_path, local_rank, batch_size=8) -> Trainer:
        max_block_size = 4
        downsample_factor = 4
        score_consensus_attn = True
        upsample_output = True
        max_length = 1024

        tokenizer, data_collator = self.tokenizer(max_length=max_length, ds_factor=downsample_factor)

        vocab_size = len(tokenizer)

        gen_config, disc_config = self.get_config(max_block_size, downsample_factor, score_consensus_attn,
                                                  upsample_output, vocab_size)

        electra_generator = ElectraForMaskedLM(gen_config)
        electra_discriminator = ElectraForPreTraining(disc_config)

        electra_discriminator.electra.embeddings = electra_generator.electra.embeddings
        electra_generator.generator_lm_head.weight = electra_generator.electra.embeddings.word_embeddings.weight

        electra_model = ElectraCharformerModel(electra_generator, electra_discriminator)

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
                                   learning_rate=8e-5,
                                   lr_scheduler_type='linear',
                                   eval_accumulation_steps=16,
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
            # compute_metrics=Electra.compute_metrics,
        )
        return trainer
