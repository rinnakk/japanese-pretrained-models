import argparse
import os

import torch
from transformers import GPT2LMHeadModel, PretrainedConfig, TFGPT2LMHeadModel
from transformers import T5Tokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", help="path to saved checkpoint file", type=str, required=True)
    parser.add_argument("--save_dir", help="path to saved checkpoint file", type=str, default="../data/model/huggingface_model")
    parser.add_argument("--model_config_filepath", type=str, default="model/gpt2-ja-medium-config.json")
    args = parser.parse_args()
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")

    model_config = PretrainedConfig.from_json_file(args.model_config_filepath)
    model = GPT2LMHeadModel(model_config)
    model.load_state_dict(checkpoint["model"])
    model.save_pretrained(args.save_dir)
    
    tf_model = TFGPT2LMHeadModel.from_pretrained(f"{args.save_dir}", from_pt=True)
    tf_model.save_pretrained(args.save_dir)

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
    tokenizer.save_pretrained(args.save_dir)
    tokenizer.save_vocabulary(args.save_dir)

    

    