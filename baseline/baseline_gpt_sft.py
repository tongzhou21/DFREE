# This code is based on the revised code from fastchat based on tatsu-lab/stanford_alpaca.
import random
from dataclasses import dataclass, field
import json

import pathlib
from typing import Dict, Optional, List

import numpy as np
import torch
from torch.utils.data import Dataset
from deepspeed import zero
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import Trainer, BitsAndBytesConfig, deepspeed
from transformers.trainer_pt_utils import LabelSmoother
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from accelerate.utils import DistributedType
import shutil
import os
import tqdm
import copy
from transformers.utils import logging
import time

logging.set_verbosity_error()

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

TEMPLATE = "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content']}}{% if loop.last %}{{ '<|im_end|>'}}{% else %}{{ '<|im_end|>\n' }}{% endif %}{% endfor %}"

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen-7B")


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    lazy_preprocess: bool = False


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=8192,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    use_lora: bool = False


@dataclass
class LoraArguments:
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "v_proj",
        ]
    )
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False


def maybe_zero_3(param):
    if hasattr(param, "ds_id"):
        assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}
    return to_return


def safe_save_model_for_hf_trainer(
    trainer: transformers.Trainer, output_dir: str, bias="none"
):
    """Collects the state dict and dump to disk."""
    if deepspeed.is_deepspeed_zero3_enabled():
        state_dict = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
    else:
        if trainer.args.use_lora:
            state_dict = get_peft_state_maybe_zero_3(
                trainer.model.named_parameters(), bias
            )
        else:
            state_dict = trainer.model.state_dict()
    if trainer.args.should_save and trainer.args.local_rank == 0:
        trainer._save(output_dir, state_dict=state_dict)


def preprocess(
    messages,
    tokenizer: transformers.PreTrainedTokenizer,
    max_len: int,
) -> Dict:
    """Preprocesses the data for supervised fine-tuning."""

    texts = []
    for i, msg in enumerate(messages):
        texts.append(
            tokenizer.apply_chat_template(
                msg,
                chat_template=TEMPLATE,
                tokenize=True,
                add_generation_prompt=False,
                padding=True,
                max_length=max_len,
                truncation=True,
            )
        )
    input_ids = torch.tensor(texts, dtype=torch.int)
    target_ids = input_ids.clone()
    target_ids[target_ids == tokenizer.pad_token_id] = IGNORE_TOKEN_ID
    attention_mask = input_ids.ne(tokenizer.pad_token_id)

    return dict(
        input_ids=input_ids, target_ids=target_ids, attention_mask=attention_mask
    )


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self, raw_data, tokenizer: transformers.PreTrainedTokenizer, max_len: int
    ):
        super(SupervisedDataset, self).__init__()

        rank0_print("Formatting inputs...")
        messages = [example["messages"] for example in raw_data]
        data_dict = preprocess(messages, tokenizer, max_len)

        self.input_ids = data_dict["input_ids"]
        self.target_ids = data_dict["target_ids"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.target_ids[i],
            attention_mask=self.attention_mask[i],
        )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self, raw_data, tokenizer: transformers.PreTrainedTokenizer, max_len: int
    ):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        ret = preprocess([self.raw_data[i]["messages"]], self.tokenizer, self.max_len)
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["target_ids"][0],
            attention_mask=ret["attention_mask"][0],
        )
        self.cached_data_dict[i] = ret

        return ret


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args,
    max_len,
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = (
        LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset
    )
    rank0_print("Loading data...")

    train_data = []
    with open(data_args.data_path, "r") as f:
        for line in f:
            train_data.append(json.loads(line))
    train_dataset = dataset_cls(train_data, tokenizer=tokenizer, max_len=max_len)

    if data_args.eval_data_path:
        eval_data = []
        with open(data_args.eval_data_path, "r") as f:
            for line in f:
                eval_data.append(json.loads(line))
        eval_dataset = dataset_cls(eval_data, tokenizer=tokenizer, max_len=max_len)
    else:
        eval_dataset = None

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        lora_args,
    ) = parser.parse_args_into_dataclasses()

    if (
        getattr(training_args, "deepspeed", None)
        and int(os.environ.get("WORLD_SIZE", 1)) == 1
    ):
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED

    if getattr(training_args, "deepspeed", None) is None:
        print('getattr(training_args, "deepspeed", None) is None:')
        device_map = 'auto'
    else:
        device_map = None



    local_rank = training_args.local_rank

    # device_map = None
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if lora_args.q_lora:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if ddp else "auto"
        if len(training_args.fsdp) > 0 or deepspeed.is_deepspeed_zero3_enabled():
            logging.warning("FSDP or ZeRO3 is incompatible with QLoRA.")

    model_load_kwargs = {
        "low_cpu_mem_usage": not deepspeed.is_deepspeed_zero3_enabled(),
    }

    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    # Load model and tokenizer
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    config.use_cache = False

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        device_map=device_map,
        # attn_implementation="sdpa", # TODO: https://github.com/QwenLM/Qwen1.5/issues/240
        attn_implementation="flash_attention_2",  # TODO: https://github.com/QwenLM/Qwen1.5/issues/240
        torch_dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        )
        if training_args.use_lora and lora_args.q_lora
        else None,
        **model_load_kwargs,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    if training_args.use_lora:
        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=lora_args.lora_target_modules,
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if lora_args.q_lora:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=training_args.gradient_checkpointing
            )

        model = get_peft_model(model, lora_config)

        model.print_trainable_parameters()

        if training_args.gradient_checkpointing:
            model.enable_input_require_grads()

    data_module = make_supervised_data_module(
        tokenizer=tokenizer, data_args=data_args, max_len=training_args.model_max_length
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module
    )

    # `not training_args.use_lora` is a temporary workaround for the issue that there are problems with
    # loading the checkpoint when using LoRA with DeepSpeed.
    # Check this issue https://github.com/huggingface/peft/issues/746 for more information.
    if (
        list(pathlib.Path(training_args.output_dir).glob("checkpoint-*"))
        and not training_args.use_lora
    ):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    safe_save_model_for_hf_trainer(
        trainer=trainer, output_dir=training_args.output_dir, bias=lora_args.lora_bias
    )

def merge_model(lora_path):
    from peft import AutoPeftModelForCausalLM

    model_name = lora_path.split('/')[-1]
    if model_name == '':
        model_name = lora_path.split('/')[-2]
    model_path = 'model_save/sft/' + model_name + '/'
    print('lora_path', lora_path)
    print('model_path', model_path)

    tokenizer = AutoTokenizer.from_pretrained(lora_path)
    tokenizer.save_pretrained(model_path)

    model = AutoPeftModelForCausalLM.from_pretrained(
        lora_path,
        device_map='auto',
        trust_remote_code=True,
        torch_dtype="auto",
        offload_folder=model_path
    ).eval()

    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(model_path, max_shard_size="4096MB", safe_serialization=True)

def predict(model_path, template_path, test_path, max_prompt_length=8192 * 2 - 1024):
    from utils import metric_api
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from utils import format_ans

    current_file_path = os.path.abspath(__file__)
    print('current_file_path', current_file_path)
    print('target', os.path.join(model_path, os.path.basename(current_file_path)))
    shutil.copyfile(current_file_path, model_path + 'code_predict.py')

    with open(test_path, 'r') as f_read:
        list_dict_data_raw = [json.loads(line) for line in f_read]
    random.shuffle(list_dict_data_raw)
    print('len(list_dict_data_raw)', len(list_dict_data_raw))

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    with open(template_path, 'r') as f_read:
        prompt_template = f_read.read()

    list_dict_data_write = []
    doc_id2events_tgt, doc_id2events_pred = {}, {}
    dict_event2count = {}

    list_token_count = []
    for idx_data, dict_data in tqdm.tqdm(enumerate(list_dict_data_raw), ncols=100, total=len(list_dict_data_raw)):
        prompt = prompt_template.replace('{{TEXT}}', dict_data['content'])
        list_token_count.append(len(tokenizer.tokenize(prompt)))
    print(np.percentile(np.array(list_token_count), [0, 25, 50, 75, 100]))
    print('token_count template', len(tokenizer.tokenize(prompt_template))) # 1858

    list_dict_data_raw = [list_dict_data_raw[idx] for _, idx in
                          sorted(zip(list_token_count, [_ for _ in range(len(list_token_count))]), reverse=True)]

    token_count_template = len(tokenizer.tokenize(prompt_template))
    print('token_count_template, max_prompt_length', token_count_template, max_prompt_length)

    for idx_data, dict_data in tqdm.tqdm(enumerate(list_dict_data_raw), ncols=100, total=len(list_dict_data_raw)):
        prompt = prompt_template.replace('{{TEXT}}', dict_data['content'])
        token_count = len(tokenizer.tokenize(prompt))
        del_token_count = token_count - max_prompt_length
        if del_token_count > 0:
            text_new = tokenizer.decode(tokenizer(dict_data['content']).input_ids[:-del_token_count])
            prompt = prompt_template.replace('{{TEXT}}', text_new)

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to("cuda")

        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=1024
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        list_dict_res = format_ans(response)
        if list_dict_res is None :
            list_dict_res = []
        else:
            for idx, dict_res in enumerate(list_dict_res):
                list_key_del = []
                for key, value in dict_res.items():
                    if value is None or value == '':
                        list_key_del.append(key)

                if list_key_del != []:
                    list_dict_res[idx] = {key: value for key, value in list_dict_res[idx].items() if key not in list_key_del}

        doc_id2events_pred[idx_data] = copy.deepcopy(list_dict_res)
        doc_id2events_tgt[idx_data] = copy.deepcopy(dict_data['events'])

        dict_data['pred_events'] = copy.deepcopy(list_dict_res)
        list_dict_data_write.append(dict_data)


    with open(model_path + 'pred_res.jsonl', 'w') as f_write:
        for dict_data in list_dict_data_write:
            f_write.write(json.dumps(dict_data, ensure_ascii=False) + '\n')

    print('len(doc_id2events_tgt.keys()), len(doc_id2events_pred.keys())',
          len(doc_id2events_tgt.keys()), len(doc_id2events_pred.keys()))

    print('dict_event2count', dict_event2count)
    str_res = metric_api(doc_id2events_tgt, doc_id2events_pred)
    str_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    str_out = f'time: {str_time}\nmodel_path: {model_path}\n' \
              f'template_path: {template_path}\ntest_path: {test_path}\n' + \
              f'max_prompt_length: {max_prompt_length}\n\n' + \
              str_res
    with open(model_path + f'eval_res.{str_time}.txt', 'w') as f_write:
        f_write.write(str_out)

if __name__ == "__main__":
    train()

    # predict(
    #     model_path='model_save/sft/Qwen1.5-7B-Chat.rand18k.prompt-v1-wodemo.tunc8192.5epoch/',
    #     template_path = 'prompt/prompt.template.v1.wodemos.txt',
    #     test_path='data/data_split/test.jsonl',
    # )



'''
CUDA_VISIBLE_DEVICES=0 bash finetune.sh

'''

