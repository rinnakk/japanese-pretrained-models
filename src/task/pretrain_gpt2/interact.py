# coding=utf-8
# Copyright 2021 rinna Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
import time
import argparse

import numpy as np
import torch
import torch.cuda.amp as amp
from transformers import GPT2LMHeadModel, GPT2Config
from transformers import T5Tokenizer


def str2bool(v):
    return v.lower() in ('true', '1', "True")


def interact(config):

    # multi-gpu init
    if torch.cuda.is_available() and config.use_gpu:
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")
    print("Using", DEVICE)

    # build tokenizer
    tokenizer = T5Tokenizer(
        vocab_file="../data/tokenizer/google_sp.model",
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
        extra_ids=0,
        additional_special_tokens=(),
        do_lower_case=True
    )

    # build model
    model_config = GPT2Config.from_json_file(config.model_config_filepath)
    model = GPT2LMHeadModel(model_config)
    model = model.to(DEVICE)
    model.eval()

    # load model from checkpoint
    if config.checkpoint_path:
        print("----- Checkpoint loaded -----")
        print("checkpoint path: {}".format(config.checkpoint_path))
        checkpoint = torch.load(config.checkpoint_path, map_location=model.device)
        print("loading model state dict...")
        model.load_state_dict(checkpoint["model"])
        model.tie_weights()  # NOTE: don't forget to tie weights after loading weights

    while True:
        try:
            # receive prompt
            prompt_text = input("Prompt: ")
            prompt_text = prompt_text.strip()
            if len(prompt_text) == 0:
                continue

            start_time = time.time()

            # convert query to model inputs
            prompt_tokens = tokenizer.tokenize(prompt_text)
            prompt_token_ids = tokenizer.convert_tokens_to_ids(prompt_tokens)
            prompt_tensor = torch.LongTensor(prompt_token_ids).to(model.device)
            prompt_tensor = prompt_tensor.view(1, -1)

            # model forward
            with amp.autocast():
                output_sequences = model.generate(
                    input_ids=prompt_tensor,
                    max_length=config.max_gen_seq_len + len(prompt_token_ids),
                    top_p=config.top_p,
                    top_k=config.top_k,
                    temperature=config.temp,
                    do_sample=config.do_sample,
                    early_stopping=True,
                    bos_token_id=tokenizer.bos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    num_return_sequences=1
                )
            
            # convert model outputs to readable sentence
            generated_sequence = output_sequences.tolist()[0][len(prompt_token_ids):]
            generated_tokens = tokenizer.convert_ids_to_tokens(generated_sequence)
            generated_text = tokenizer.convert_tokens_to_string(generated_tokens)
            
            end_time = time.time()

            # display generated response
            print(f"Generated text: {generated_text}")
            print(f"\t(response generation took {(end_time - start_time):.3f} sec)")

        # catch EOF(CTRL+d) and KeyboardInterupt(CTRL+c) for exiting
        except BaseException as e:
            if isinstance(e, (EOFError, KeyboardInterrupt)):
                exit(0)
            else:
                print(e)
                exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_config_filepath", type=str, default="model/gpt2-ja-small-config.json")
    parser.add_argument("--seed", type=int, default=42, help="random initialization seed")
    parser.add_argument("--max_gen_seq_len", type=int, default=200)
    parser.add_argument("--temp", type=float, default=1.0, help="temperature for decoding")
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--do_sample", type=str2bool, default=True)
    parser.add_argument("--use_gpu", type=str2bool, default=True, help="use gpu or not")
    parser.add_argument("--checkpoint_path", help="path to saved checkpoint file")
    
    config = parser.parse_args()

    # set random seeds
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)

    interact(config)
