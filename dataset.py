import argparse
from enum import Enum
import logging 
from typing import Optional, Callable, List, Dict, Union

from pixel import get_attention_mask
from PIL import Image
from pixel import PangoCairoTextRenderer,PyGameTextRenderer 
from transformers import AutoTokenizer
import torch

class Split(Enum):
    TRAIN = 'train'
    VAL = 'validation'
    TEST = 'test'

logger = logging.getLogger(__name__)

def generate_features(
    data_args: argparse.Namespace,
    dataset: List[Dict [str, str]],
    transforms: Optional[Callable],
    renderer: Union[PangoCairoTextRenderer, PyGameTextRenderer],
    tokenizer: AutoTokenizer,
):  
    for (index, example) in enumerate(dataset):
        if index % 10_000 == 0:
            logger.info("Writing example %d of %d", index, len(dataset))

        document, summary = example['document'], example['summary']
        
        encoding = renderer(document)
        image = encoding.pixel_values
        num_patches = encoding.num_text_patches
        pixel_values = transforms(Image.fromarray(image))
        attention_mask = get_attention_mask(num_patches, seq_length=data_args.max_seq_length)

        label_ids = tokenizer(summary, padding=True)

        assert len(attention_mask) == data_args.max_seq_length

        if index < 3:
            logger.info("*** Example ***")
            logger.info(f"sentence: {document}")
            logger.info(f"attention_mask: {attention_mask}")
            logger.info(f"summary: {summary}")
            
        yield {"pixel_values": pixel_values, "attention_mask": attention_mask, "label_ids": label_ids}
        

def convert_examples_to_image_features(
    data_args: argparse.Namespace,
    dataset: List[Dict [str, str]],
    transforms: Optional[Callable],
    renderer: Union[PangoCairoTextRenderer, PyGameTextRenderer],
    tokenizer: AutoTokenizer,
) -> List[Dict[str, Union[int, torch.Tensor, List[int]]]]:
    """Loads a data file into a list of `Dict` containing image features"""
    features = [] 
    for (index, example) in enumerate(dataset):
        if index % 10_000 == 0:
            logger.info("Writing example %d of %d", index, len(dataset))

        document, summary = example['document'], example['summary']
        
        encoding = renderer(document)
        image = encoding.pixel_values
        num_patches = encoding.num_text_patches
        pixel_values = transforms(Image.fromarray(image))
        attention_mask = get_attention_mask(num_patches, seq_length=data_args.max_seq_length)
        
        text_ids = tokenizer.encode(summary)
        input_ids = _pad_input_ids(text_ids, tokenizer)
       
        assert len(attention_mask) == data_args.max_seq_length

        if index < 5:
            logger.info("*** Example ***")
            logger.info(f"sentence: {document}")
            logger.info(f"attention_mask: {attention_mask}")
            logger.info(f"summary: {summary}")

        features.append({"pixel_values": pixel_values, "attention_mask": attention_mask, "label_ids": input_ids})
         
    return features

def _pad_input_ids(input_ids, tokenizer, max_length=100):
    input_ids += [tokenizer.pad_token_id] * (max_length - len(input_ids))
    return input_ids