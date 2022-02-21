from transformers.models.electra.modeling_electra import ElectraConfig
from typing import Optional, Tuple
from model.finetuning.charformer import ElectraCharformerModelForFineTuning
from model.charformer.electra import ElectraForPreTraining
import torch
from model.tokenizer.charformer_tokenizer import CharformerTokenizer


def load_charmen_electra(checkpoint: str, max_block_size: int, downsample_factor: int, score_consensus_attn: bool,
                         upsample_output: bool, discriminator_config: Optional[ElectraConfig] = None
                         ) -> Tuple[ElectraCharformerModelForFineTuning, CharformerTokenizer]:
    """
    Loads discriminator module

    :param checkpoint: Path to saved weights
    :param max_block_size: Charformer: Max Block size
    :param downsample_factor: Charformer: Downsampling factor
    :param score_consensus_attn: Charformer: Use consensus between blocks
    :param upsample_output: Upsample output
    :param discriminator_config: An existing ElectraConfig object or None.
    In case of None, it loads the default 'google/electra-base-discriminator' from Huggingface
    :return: Returns the prepared module and the tokenizer
    """
    if discriminator_config is None:
        discriminator_config = ElectraConfig.from_pretrained('google/electra-base-discriminator')

    tokenizer = CharformerTokenizer()

    discriminator_config.max_block_size = max_block_size
    discriminator_config.downsample_factor = downsample_factor
    discriminator_config.score_consensus_attn = score_consensus_attn
    discriminator_config.upsample_output = upsample_output
    discriminator_config.vocab_size = len(tokenizer)

    electra_discriminator = ElectraForPreTraining(discriminator_config)
    discriminator = ElectraCharformerModelForFineTuning(electra_discriminator)
    discriminator.load_state_dict(torch.load(checkpoint, map_location=torch.device("cpu")), strict=False)

    return discriminator, tokenizer
