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

import argparse
import random

import torch.cuda.amp as amp
from transformers import AutoModelForCausalLM, TFAutoModelForCausalLM
from transformers import T5Tokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="../data/model/huggingface_model")
    args = parser.parse_args()

    random.seed(42)

    tokenizer = T5Tokenizer.from_pretrained(args.model_dir, extra_ids=0)
    print(len(tokenizer))
    
    pt_model = AutoModelForCausalLM.from_pretrained(args.model_dir)
    tf_model = TFAutoModelForCausalLM.from_pretrained(args.model_dir)

    prompt = "誰も到達していないArtificial Intelligenceの高みへ、ともに"
    with amp.autocast():
        pt_tensor = tokenizer(prompt, return_tensors="pt")["input_ids"]
        output_sequences = pt_model.generate(
            input_ids=pt_tensor,
            max_length=50+pt_tensor.size(1),
            top_p=0.95,
            top_k=50,
            do_sample=True,
            early_stopping=True,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            num_return_sequences=1
        )
        generated = output_sequences.tolist()[0]
        generated = tokenizer.decode(generated)
        print("pytorch:   ", generated)

        tf_tensor = tokenizer(prompt, return_tensors="tf")["input_ids"]
        output_sequences = tf_model.generate(
            input_ids=tf_tensor,
            max_length=50+pt_tensor.size(1),
            top_p=0.95,
            top_k=50,
            do_sample=True,
            early_stopping=True,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            num_return_sequences=1
        )
        generated = output_sequences.numpy().tolist()[0]
        generated = tokenizer.decode(generated)
        print("tensorflow:", generated)

