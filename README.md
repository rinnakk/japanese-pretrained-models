
# japanese-gpt2

![rinna-icon](./rinna.png)

This repository provides the code for training Japanese GPT-2 models. This code has been used for producing [japanese-gpt2-medium released on  HuggingFace model hub](https://huggingface.co/rinna/japanese-gpt2-medium) by [rinna](https://corp.rinna.co.jp/).

---

**Please open an issue (in English/日本語) if you encounter any problem using the code or using our models via Huggingface.**

---

## Train a Japanese GPT-2 from scratch on your own machine

1. Download training corpus [Japanese CC-100](http://data.statmt.org/cc-100/ja.txt.xz) and extract the `ja.txt` file.

2. Move the `ja.txt` file or modify `src/corpus/jp_cc100/config.py` to match the filepath of `ja.txt` with `self.raw_data_dir` in the config file.

3. Split `ja.txt` to smaller files by running:
~~~~
cd src/
python -m corpus.jp_cc100.split_to_small_files
~~~~

4. Train a medium-sized GPT-2 on 4 GPUs by running:
~~~~
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m task.pretrain.train --n_gpus 4 --save_model True --enable_log True
~~~~

## Interact with the trained model

Assume you have run the training script and saved your medium-sized GPT-2 to `data/model/gpt2-medium-xxx.checkpoint`. Run the following command to use it to complete text on one GPU by nucleus sampling with `p=0.95` and `k=40`:

~~~~
CUDA_VISIBLE_DEVICES=0 python -m task.pretrain.interact --checkpoint_path ../data/model/gpt2-medium-xxx.checkpoint --gen_type top --top_p 0.95 --top_k 40
~~~~

## Prepare files for uploading to Huggingface

1. Make your Huggingface account; Create a model repo; Clone it to your local machine.

2. Create model and config files from a checkpoint by running:
~~~~
python -m task.pretrain.checkpoint2huggingface --checkpoint_path ../data/model/gpt2-medium-xxx.checkpoint --save_dir {huggingface's model repo directory}
~~~~

3. Validate the created files by running:
~~~~
python -m task.pretrain.check_huggingface --model_dir {huggingface's model repo directory}
~~~~

4. Add files, commit, and push to your Huggingface repo.

## Customize your training script

Check available arguments by running:
~~~~
python -m task.pretrain.train --help
~~~~

## License

[The MIT license](https://opensource.org/licenses/MIT)