import os
import os.path as osp
import time
import json
import logging
import pdb
import torch
import transformers
import sklearn
import numpy as np
import random

from random import sample
# from datasets import Dataset
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
from torch.utils.data import Dataset
from transformers import DataCollatorForLanguageModeling

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
)

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="")
    tokenizer_path: Optional[str] = field(default="")
    use_lora: bool = field(default=False, metadata={"help": "whether to use LoRA"})
    lora_r: int = field(default=8, metadata={"help": "hidden dimension for LoRA"})
    lora_alpha: int = field(default=32, metadata={"help": "alpha for LoRA"})
    lora_dropout: float = field(default=0.05, metadata={"help": "dropout rate for LoRA"})
    # lora_target_modules: str = field(default="query,value", metadata={"help": "where to perform LoRA"})
    lora_target_modules: str = field(default="Wqkv", metadata={"help": "where to perform LoRA"})


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    kmer: int = field(default=-1, metadata={"help": "k-mer for input sequence. -1 means not using k-mer."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    run_name: str = field(default="run")
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=512, metadata={"help": "Maximum sequence length."})
    gradient_accumulation_steps: int = field(default=1)
    per_device_train_batch_size: int = field(default=1)
    per_device_eval_batch_size: int = field(default=1)
    num_train_epochs: int = field(default=1)
    fp16: bool = field(default=False)
    bp16: bool = field(default=False)
    logging_steps: int = field(default=100)
    save_steps: int = field(default=100)
    eval_steps: int = field(default=100)
    evaluation_strategy: str = field(default="steps"),
    warmup_steps: int = field(default=50)
    weight_decay: float = field(default=0.01)
    learning_rate: float = field(default=1e-4)
    save_total_limit: int = field(default=3)
    load_best_model_at_end: bool = field(default=True)
    output_dir: str = field(default="output")
    find_unused_parameters: bool = field(default=False)
    checkpointing: bool = field(default=False)
    dataloader_pin_memory: bool = field(default=False)
    eval_and_save_results: bool = field(default=True)
    save_model: bool = field(default=False)
    seed: int = field(default=42)

def set_seed(args):
    # set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

def load_sequences(genes_path):
    sequences = []
    filenamelist = os.listdir(genes_path)
    
    for i in range(len(filenamelist)):
        gene_path = osp.join(genes_path, filenamelist[i])
        f = open(gene_path)
        file = f.readlines()
        f.close()
    
        for line in file:
            line = line.replace('\n', '').split(',')
            seq = line[1] + line[2]
            sequences.append(seq)
    return sequences

def load_tokens(path):
    tokens = []
    filenamelist = os.listdir(path)
    
    tokens = []
    for i in range(len(filenamelist)):
        token = np.load(osp.join(path, filenamelist[i]))
        tokens.append(torch.tensor(token))
    tokens = np.concatenate(tokens, axis=0)
    return tokens

class UnsupervisedDataset(Dataset):
    """Dataset for unsupervised fine-tuning."""

    def __init__(self, 
                 data: str, 
                 tokenizer: transformers.PreTrainedTokenizer, 
                 kmer: int = -1):

        super(UnsupervisedDataset, self).__init__()

        self.data = data
        self.tokenizer = tokenizer
        # self.input_ids = self.batched_tokenization()
        
        # if kmer != -1:
        #     # only write file on the first process
        #     if torch.distributed.get_rank() not in [0, -1]:
        #         torch.distributed.barrier()

        #     logging.warning(f"Using {kmer}-mer as input...")
        #     texts = load_or_generate_kmer(data_path, texts, kmer)

        #     if torch.distributed.get_rank() == 0:
        #         torch.distributed.barrier()
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        x = torch.tensor(self.data[i])
        return dict(input_ids=x)

@dataclass
class DataCollatorForUnsupervisedDataset(DataCollatorForLanguageModeling):
    """Collate examples for unsupervised fine-tuning."""
    def __init__(self, tokenizer, mlm=True, mlm_probability=0.15):
        super().__init__(tokenizer, mlm, mlm_probability)
        # self.tokenizer = tokenizer

    def __call__(self, examples):
        pdb.set_trace()
        # input_ids = [instance['input_ids'] for instance in instances]
        # input_ids = torch.nn.utils.rnn.pad_sequence(
        #     input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        # )
        # tokenizer(examples, return_tensors="pt", padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True)
        output = self.tokenizer(examples, return_tensors="pt", padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True)
        input_ids = output['input_ids']
        pdb.set_trace()
        batch = super().__call__(input_ids)
        pdb.set_trace()
        
        # return dict(
        #     input_ids=input_ids,
        #     attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        # )
        return batch

"""
Manually calculate the accuracy, f1, matthews_correlation, precision, recall with sklearn.
"""
def calculate_metric_with_sklearn(logits: np.ndarray, labels: np.ndarray):
    # if logits.ndim == 3:
    #     # Reshape logits to 2D if needed
    #     logits = logits.reshape(-1, logits.shape[-1])
    predictions = np.argmax(logits, axis=-1)
    valid_mask = labels != -100  # Exclude padding tokens (assuming -100 is the padding token ID)
    valid_predictions = predictions[valid_mask]
    valid_labels = labels[valid_mask]
    return {
        "accuracy": sklearn.metrics.accuracy_score(valid_labels, valid_predictions),
        # "f1": sklearn.metrics.f1_score(
        #     valid_labels, valid_predictions, average="macro", zero_division=0
        # ),
        # "matthews_correlation": sklearn.metrics.matthews_corrcoef(
        #     valid_labels, valid_predictions
        # ),
        # "precision": sklearn.metrics.precision_score(
        #     valid_labels, valid_predictions, average="macro", zero_division=0
        # ),
        # "recall": sklearn.metrics.recall_score(
        #     valid_labels, valid_predictions, average="macro", zero_division=0
        # ),
    }

"""
Compute metrics used for huggingface trainer.
""" 
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):  # Unpack logits if it's a tuple
        logits = logits[0]
    return calculate_metric_with_sklearn(logits, labels)


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    set_seed(training_args)
    
    # load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.tokenizer_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )
    
    if "InstaDeepAI" in model_args.model_name_or_path:
        tokenizer.eos_token = tokenizer.pad_token
    
    start = time.time()
    # sequences = load_sequences(data_args.data_path)
    tokens = load_tokens(data_args.data_path)
    print("Data loading time: ", time.time()-start, 's.')
    
    # Cut to 50 tokens
    tokens = tokens[:, :tokenizer.model_max_length]
    
    # pdb.set_trace()
    
    # num_sample = len(tokens) // 8
    # num_val = num_sample // 100
    num_val = 5000
    
    # tokens_selected = tokens[:num_sample]
    tokens_selected = tokens  # select all data
    np.random.shuffle(tokens_selected)

    # index = torch.LongTensor(random.sample(range(len(tokens)), num_sample))
    # tokens_selected = torch.index_select(tokens, 0, index)
    
    # define datasets and data collator
    train_dataset = UnsupervisedDataset(tokenizer=tokenizer, 
                                        data=tokens_selected[:-num_val],
                                        kmer=data_args.kmer)
    val_dataset = UnsupervisedDataset(tokenizer=tokenizer, 
                                      data=tokens_selected[-num_val:],
                                      kmer=data_args.kmer)
    # data_collator = DataCollatorForUnsupervisedDataset(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)  # 0.3/0.15
    
    # load model
    model = transformers.AutoModelForMaskedLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16, 
        )
    # pdb.set_trace()
    
    # configure LoRA
    if model_args.use_lora:
        lora_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            target_modules=list(model_args.lora_target_modules.split(",")),
            lora_dropout=model_args.lora_dropout,
            bias="none",
            task_type="SEQ_CLS",
            inference_mode=False,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    # print(model)
    
    # define trainer
    trainer = transformers.Trainer(model=model,
                                   tokenizer=tokenizer,
                                   args=training_args,
                                   compute_metrics=compute_metrics,
                                   train_dataset=train_dataset,
                                   eval_dataset=val_dataset,
                                   data_collator=data_collator)
    
    trainer.train()
    
    if training_args.save_model:
        trainer.save_state()
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

    # get the evaluation results from trainer
    if training_args.eval_and_save_results:
        results_path = os.path.join(training_args.output_dir, "results", training_args.run_name)
        # results = trainer.evaluate(eval_dataset=test_dataset)
        results = trainer.evaluate(eval_dataset=val_dataset)
        os.makedirs(results_path, exist_ok=True)
        with open(os.path.join(results_path, "eval_results.json"), "w") as f:
            json.dump(results, f)

def getTokens():
    # parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    # model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        # '/root/pretrained/DNABERT-2-117M',
        '/root/projects/DNABERT_Promotor/0pretrain/tokenizer/20240603_093134_tokenizer4096_multiprocess/',
        cache_dir=None,
        model_max_length=100,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )

    data_path = '/hy-tmp/prokaryotes/genes'
    start = time.time()
    filenamelist = os.listdir(data_path)
    token_path = '/hy-tmp/prokaryotes/tokens_new/'
    if not os.path.exists(token_path):
        os.mkdir(token_path)
    
    for i in range(len(filenamelist)):
        gene_path = osp.join(data_path, filenamelist[i])
        f = open(gene_path)
        file = f.readlines()
        f.close()
    
        sequences = []
        for line in file:
            line = line.replace('\n', '').split(',')
            seq = line[1] + line[2]
            sequences.append(seq)
        output = tokenizer(
            sequences, 
            return_tensors="np", 
            padding="max_length", 
            max_length=tokenizer.model_max_length, 
            truncation=True
        )
        data = output['input_ids']
        np.save(os.path.join(token_path, filenamelist[i].split('.')[0]+'.npy'), data)
        print(str(i), 'saved.')
    print("Token saving time: ", time.time()-start, 's.')


if __name__ == "__main__":
    # getTokens()
    train()
