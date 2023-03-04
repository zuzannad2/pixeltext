import logging
import os
import sys
from typing import Dict, List, Tuple, Union, Optional
from dataclasses import dataclass, field

import datasets
import evaluate
import numpy as np
from datasets import load_dataset, load_metric, Dataset

from random import randint
import argparse
import transformers
import torch
import datasets
import numpy as np
from torch import nn
from PIL import Image
import wandb

import transformers
from transformers import (
    AutoTokenizer,
    AutoConfig, 
    AutoModel,
    Seq2SeqTrainer,
    default_data_collator,
    Seq2SeqTrainingArguments,
    EvalPrediction,
    HfArgumentParser,
    DataCollatorForSeq2Seq
)
from pixel import (
    PangoCairoTextRenderer,
    PyGameTextRenderer,
    get_transforms,
    PoolingMode
)
from pixel.utils.misc import get_attention_mask
from pixel_to_text_config import PixelToTextEncoderDecoderConfig
from pixel_to_text_model import PixelToTextEncoderDecoderModel
from dataset import Split, generate_features

logger = logging.getLogger(__name__)



@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    encoder_name: str = field(
        metadata={"help": "Path to pretrained encoder or model identifier from huggingface.co/models"}
    )
    decoder_name: str = field(
        metadata={"help": "Path to pretrained decoder or model identifier from huggingface.co/models"}
    )
    processor_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained processor name or path if not the same as encoder_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as decoder_name"}
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    rendering_backend: Optional[str] = field(
        default="pangocairo", metadata={
            "help": "Rendering backend to use. Options are 'pygame' or 'pangocairo'. For most applications it is "
                    "recommended to use the default 'pangocairo'."}
    )
    fallback_fonts_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Directory containing fallback font files used by the text renderer for PIXEL. "
                          "PyGame does not support fallback fonts so this argument is ignored when using the "
                          "PyGame backend."},
    ) 
    render_rgb: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to render images in RGB. RGB rendering can be useful when working with emoji "
            "but it makes rendering a bit slower, so it is recommended to turn on RGB rendering only "
            "when there is need for it. PyGame does not support fallback fonts so this argument is ignored "
            "when using the PyGame backend."
        }
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    pooling_mode: str = field(
        default="mean",
        metadata={
            "help": f"Pooling mode to use in classification head (options are {[e.value for e in PoolingMode]}."
        },
    )
    pooler_add_layer_norm: bool = field(
        default=True,
        metadata={
            "help": "Whether to add layer normalization to the classification head pooler. Note that this flag is"
            "ignored and no layer norm is added when using CLS pooling mode."
        },
    )
    dropout_prob: float = field(
        default=0.2, metadata={"help": "Dropout probability for attention blocks and classification head"}
    )
    train_decoder: bool = field(
        default=False,
        metadata={"help": "Whether to freeze updating the weights of the decoder."}
    )

    def __post_init__(self):
        self.pooling_mode = PoolingMode.from_string(self.pooling_mode)

        if not self.rendering_backend.lower() in ["pygame", "pangocairo"]:
            raise ValueError("Invalid rendering backend. Supported backends are 'pygame' and 'pangocairo'.")
        else:
            self.rendering_backend = self.rendering_backend.lower()

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    max_seq_length: Optional[int] = field(
        default=529,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=180,
        metadata={
            "help": (
                "The maximum total sequence length for target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total sequence length for validation target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
                "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
                "during ``evaluate`` and ``predict``."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    log_predictions: bool = field(
        default=True,
        metadata={
            "help": "Whether to log predicted summaries."
        },
    )

    def __post_init__(self):
        if self.dataset_name is None:
            raise ValueError("Need a dataset name.")
        

def log_predictions(args, p, tokenizer, prefix):
    # Initialize wandb if not already done
    if not args.do_train:
        wandb.init(reinit=False)

    data = []
    out_file = os.path.join(args.output_dir, f"{prefix}_predictions.csv")
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("summary\tpred\n")
        preds = np.argmax(p.predictions[0], axis=2)
        label_ids = p.label_ids
        for pred, id in zip(preds, label_ids):
            p, r = tokenizer.decode(pred), tokenizer.decode(id)
            data.append([p, r])
            f.write(f"'Pred: {p}\t, Sum: {r}\n")
            f.write("\n")

    logger.info(f"Saved predictions and labels to {out_file}")
    logger.info(f"Logging as table to wandb")

    preds_table = wandb.Table(columns=["summary", "pred"], data=data)
    wandb.log({f"{prefix}_outputs": preds_table})

def get_data(
    data_args: argparse.Namespace,
    renderer: Union[PyGameTextRenderer, PangoCairoTextRenderer],
    tokenizer: AutoTokenizer,
    split: Split,
):  
    dataset = load_dataset(path = data_args.dataset_name, split = split)
    transforms = get_transforms(
        do_resize=True, 
        size=(renderer.pixels_per_patch, renderer.pixels_per_patch * renderer.max_seq_length))

    if split == 'train':
        if data_args.max_train_samples is not None:
            dataset = dataset.shuffle().select(range(data_args.max_train_samples))
    if split == 'validation':
        if data_args.max_eval_samples is not None:
            dataset = dataset.shuffle().select(range(data_args.max_eval_samples))
    if split == 'test':
        if data_args.max_predict_samples is not None:
            dataset = dataset.shuffle().select(range(data_args.max_predict_samples))
    
    dataset = Dataset.from_generator(lambda: generate_features(data_args, dataset, transforms, renderer, tokenizer), num_proc=data_args.preprocessing_num_workers)
    #dataset = dataset.map(tokenize_labels, batched=True, num_proc=data_args.preprocessing_num_workers, desc=f"Running tokenizer on {split} dataset.")
    return dataset


def get_renderer_and_tokenizer(model_args: argparse.Namespace):
    tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.decoder_name,
            use_fast=True,
            add_prefix_space=True if model_args.decoder_name == "gpt2" else False,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
        )
    renderer_cls = PyGameTextRenderer if model_args.rendering_backend == "pygame" else PangoCairoTextRenderer
    renderer = renderer_cls.from_pretrained(
            model_args.processor_name if model_args.processor_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            fallback_fonts_dir=model_args.fallback_fonts_dir,
            rgb=model_args.render_rgb,
        )
    return renderer, tokenizer

def get_model_and_config(model_args: argparse.Namespace):    
    model = PixelToTextEncoderDecoderModel.from_encoder_decoder_pretrained(
            model_args.encoder_name,
            model_args.decoder_name,
        )
        
    for param in model.encoder.parameters():
        param.requires_grad = False

    if "opt" in model_args.decoder_name:
        if not model_args.train_decoder:
            for name, param in model.decoder.named_parameters():
                if 'encoder_attn' not in name:
                    param.requires_grad = False

    else:
        if not model_args.train_decoder:
            for name, param in model.decoder.named_parameters():
                if 'crossattention' not in name:
                    param.requires_grad = False
   
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_trainable_params = sum([np.prod(p.size()) for p in model_parameters])
    
    logger.info('Training a model with {} trainable parameters.'.format(num_trainable_params))
    logger.info(f"Using dropout with probability {model_args.dropout_prob}")
    
    return model, model.config

def main():

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))  
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = logging.INFO
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity_info()
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Load pretrained model and tokenizer
    model, config = get_model_and_config(model_args)
    renderer, tokenizer = get_renderer_and_tokenizer(model_args)
    
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
        model.decoder.resize_token_embeddings(len(tokenizer))

    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.decoder_start_token_id = tokenizer.eos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id 
    model.config.pad_token_id = tokenizer.pad_token_id 
    model.encoder.config.do_eval = True
     
    # Get the datasets
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        if training_args.do_train:
            train_dataset = get_data(data_args, renderer, tokenizer, 'train')
            logger.info(f'Successfully loaded the training data with {len(train_dataset)} examples.')
        if training_args.do_eval:
            val_dataset = get_data(data_args, renderer, tokenizer, 'validation')
            logger.info(f'Successfully loaded the validation data with {len(val_dataset)} examples.')
        if training_args.do_predict:
            test_dataset = get_data(data_args, renderer, tokenizer, 'test')
            logger.info(f'Successfully loaded the testing data with {len(test_dataset)} examples.')
    
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    def process_predictions(p: EvalPrediction):
        predictions, labels = [], []
        preds = np.argmax(p.predictions[0], axis=2)
        label_ids = p.label_ids
        for (pred, label_id) in zip(preds, label_ids):
            predictions.append(tokenizer.decode(pred))
            labels.append(tokenizer.decode(label_id))
        return predictions, labels

    def compute_metrics(p: EvalPrediction):
        bertscore, rouge = evaluate.load('bertscore'), evaluate.load('rouge')
        predictions, labels = process_predictions(p)
        print(predictions, labels)
        rouge_res = rouge.compute(predictions=predictions, references=labels)
        bert = bertscore.compute(predictions=predictions, references=labels, lang='eng')
        bert_res = {'precision': np.mean(bert['precision']), 'recall': np.mean(bert['recall']), 'f1': np.mean(bert['f1'])}
        
        return {
            "bertscore_precision": bert_res['precision'],
            "bertscore_recall": bert_res['recall'],
            "bertscore_f1": bert_res['f1'],
            "rouge1": rouge_res['rouge1'],
            "rouge2": rouge_res['rouge2'],
            "rougeL": rouge_res['rougeL'],
            "rougeLsum": rouge_res['rougeLsum']
        }

    # Initialise our trainer 
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=default_data_collator, 
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=val_dataset if training_args.do_eval else None,
        tokenizer=renderer,
        compute_metrics=compute_metrics,
    )

    last_checkpoint = None

    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        outputs = trainer.predict(test_dataset=val_dataset, metric_key_prefix="eval")
        metrics = outputs.metrics
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(val_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(val_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

        if training_args.log_predictions:
            log_predictions(args=training_args, p=outputs, tokenizer = tokenizer, prefix="eval")

    if training_args.do_predict:
        logger.info("*** Predict ***")

        predict_results = trainer.predict(test_dataset, metric_key_prefix="test")
        metrics = predict_results.metrics
        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(test_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(test_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)
        
        if training_args.log_predictions:
            log_predictions(args=training_args, p=outputs, tokenizer = tokenizer, prefix="test")
    
if __name__ == '__main__':
    main()




