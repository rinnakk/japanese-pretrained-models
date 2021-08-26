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

import math
import random
import time
import argparse
import os

import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
import torch.multiprocessing as mp
import torch.cuda.amp as amp
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import RandomSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import GPT2LMHeadModel, GPT2Config
from transformers import T5Tokenizer

from task.pretrain_gpt2.data_source import DataSource, collate_fn
from task.helpers import StatisticsReporter
from optimization.lr_scheduler import get_linear_schedule_with_warmup

TASK = "pretrain_gpt2"


def str2bool(v):
    return v.lower() in ('true', '1', "True")


def mp_print(text, rank):
    if rank == 0:
        print(text)


def load_docs_from_filepath(filepath, tokenizer):
    docs = []
    with open(filepath, encoding="utf-8") as f:
        doc = []
        for line in f:
            line = line.strip()
            if line == "":
                if len(doc) > 0:
                    docs.append(doc)
                doc = []
            else:
                sent = line
                tokens = tokenizer.tokenize(sent)
                token_ids = tokenizer.convert_tokens_to_ids(tokens)
                if len(token_ids) > 0:
                    doc.append(token_ids)
    return docs


def forward_step(model, tokenizer, batch_data):
    max_seq_len = max([len(seq) for seq in batch_data])

    # padding input sequences
    batch_data = [seq + [tokenizer.pad_token_id]*(max_seq_len-len(seq)) for seq in batch_data]

    # convert to tensors
    batch_tensor = torch.LongTensor(batch_data).to(model.device)

    # get inputs and outputs
    input_ids = batch_tensor[:, :-1].contiguous()
    output_ids = batch_tensor[:, 1:].contiguous()
    
    # forward
    gpt2_outputs = model(input_ids=input_ids, return_dict=True)
    loss = F.cross_entropy(
        gpt2_outputs["logits"].view(-1, len(tokenizer)),
        output_ids.view(-1),
        ignore_index=tokenizer.pad_token_id,
        reduction="mean"
    )
    with torch.no_grad():
        ppl = loss.exp()

    return loss, ppl


def train(local_rank, config):
    global_rank = config.node_rank * config.n_gpus + local_rank
    print(f"local rank: {[local_rank]}, global_rank: {[global_rank]}")

    # set random seeds
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)

    # multi-gpu init
    if torch.cuda.is_available():
        if config.world_size > 1:
            dist.init_process_group(                                   
                backend='nccl',                                    
                init_method='env://',                                   
                world_size=config.world_size,                              
                rank=global_rank                                 
            )
            torch.cuda.set_device(local_rank)
            DEVICE = torch.device("cuda", local_rank)
        else:
            DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")

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

    # build data source and reporters
    trn_reporter = StatisticsReporter()
    dev_reporter = StatisticsReporter()

    # get data filepaths
    corpus2train_filepaths = {}
    corpus2dev_filepaths = {}
    for corpus in config.corpora:
        corpus2train_filepaths[corpus] = []
        corpus2dev_filepaths[corpus] = []
        if corpus == "jp_cc100":
            from corpus.jp_cc100.config import Config
            corpus_config = Config()

            dev_file_idx = 42
            
            corpus_filepaths = sorted(list(filter(
                lambda x: x.endswith(".txt"),
                os.listdir(corpus_config.doc_data_dir)
            )))
            for file_idx, filepath in enumerate(corpus_filepaths):
                if file_idx == dev_file_idx:
                    corpus2dev_filepaths[corpus].append(f"{corpus_config.doc_data_dir}/{corpus_filepaths[file_idx]}")
                else:
                    corpus2train_filepaths[corpus].append(f"{corpus_config.doc_data_dir}/{corpus_filepaths[file_idx]}")

        elif corpus == "jp_wiki":
            from corpus.jp_wiki.config import Config
            corpus_config = Config()

            dev_file_idx = None  # we want to learn all Wikipedia docs
            
            corpus_filepaths = sorted(list(filter(
                lambda x: x.endswith(".txt"),
                os.listdir(corpus_config.doc_data_dir)
            )))
            for file_idx, filepath in enumerate(corpus_filepaths):
                if file_idx == dev_file_idx:
                    corpus2dev_filepaths[corpus].append(f"{corpus_config.doc_data_dir}/{corpus_filepaths[file_idx]}")
                else:
                    corpus2train_filepaths[corpus].append(f"{corpus_config.doc_data_dir}/{corpus_filepaths[file_idx]}")

    # get filepaths for training data
    train_filepaths = []
    if config.balanced_corpora is None:
        for filepaths in corpus2train_filepaths.values():
            train_filepaths += filepaths
        random.shuffle(train_filepaths)
    elif config.balanced_corpora == "undersample":
        min_n_files = min([len(filepaths) for filepaths in corpus2train_filepaths.values()])
        for filepaths in corpus2train_filepaths.values():
            train_filepaths += filepaths[:min_n_files]
        random.shuffle(train_filepaths)
    elif config.balanced_corpora == "oversample":
        max_n_files = max([len(filepaths) for filepaths in corpus2train_filepaths.values()])
        for filepaths in corpus2train_filepaths.values():
            over_sample_times = math.ceil(max_n_files / len(filepaths))
            oversampled_filepaths = []
            for _ in range(over_sample_times):
                oversampled_filepaths += filepaths
            train_filepaths += oversampled_filepaths[:max_n_files]
        random.shuffle(train_filepaths)
    elif config.balanced_corpora == "custom_ratio":
        corpus2ratio = {"jp_cc100": 1, "jp_wiki": 5}
        custom_ratio_filepaths = []
        for corpus, filepaths in corpus2train_filepaths.items():
            ratio = corpus2ratio[corpus]
            custom_ratio_filepaths += filepaths * ratio
        train_filepaths = custom_ratio_filepaths
        random.shuffle(train_filepaths)
    else:
        raise Exception(f"Unknown corpora balancing strategy: {config.balanced_corpora}")
        
    if config.small_data:
        train_filepaths = train_filepaths[:2]

    # get filepaths for dev data
    dev_filepaths = []
    for filepaths in corpus2dev_filepaths.values():
        dev_filepaths += filepaths

    mp_print(f"Number of training files: {len(train_filepaths)}", global_rank)
    mp_print(f"Number of dev files: {len(dev_filepaths)}", global_rank)

    # load dev data
    if global_rank == 0:
        dev_docs = []
        for dev_filepath in dev_filepaths:
            dev_docs += load_docs_from_filepath(dev_filepath, tokenizer)
        
        random.shuffle(dev_docs)
        dev_docs = dev_docs[:10000]

        mp_print("----- Loading dev data -----", global_rank)
        dev_data_source = DataSource(config, tokenizer, dev_docs, "dev", randomize=False)
        mp_print(str(dev_data_source.statistics), global_rank)
        dev_dataloader = torch.utils.data.DataLoader(
            dev_data_source,
            batch_size=config.eval_batch_size,
            num_workers=0,
            collate_fn=collate_fn,
            pin_memory=False
        )

    # build model
    model_config = GPT2Config.from_json_file(config.model_config_filepath)
    model = GPT2LMHeadModel(model_config)
    model = model.to(DEVICE)

    # load model from checkpoint
    if config.checkpoint_path:
        mp_print("----- Checkpoint loaded -----", global_rank)
        mp_print("checkpoint path: {}".format(config.checkpoint_path), global_rank)
        checkpoint = torch.load(config.checkpoint_path, map_location=model.device)
        mp_print("loading model state dict...", global_rank)
        model.load_state_dict(checkpoint["model"])
        model.tie_weights()  # NOTE: don't forget to tie weights after loading weights

    # use mixed precision
    if config.use_amp:
        scaler = amp.GradScaler()

    # use multi gpus
    if config.world_size > 1:
        model = DDP(
            model, 
            device_ids=[local_rank]
        )

    # build optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'ln']   # no decay for bias and LayerNorm (ln)
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay': config.l2_penalty},
        {
            'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 
            'weight_decay': 0.0
        }
    ]
    optimizer = optim.AdamW(
        optimizer_grouped_parameters,
        lr=config.init_lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=config.l2_penalty
    )

    # build lr scheduler
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.n_warmup_steps,
        num_training_steps=config.n_training_steps,
    )
    
    # init environment or load from checkpoint
    if config.checkpoint_path:
        if config.resume_training:
            mp_print("loading optimizer state dict...", global_rank)
            optimizer.load_state_dict(checkpoint["optimizer"])
            mp_print("recovering lr scheduler...", global_rank)
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            mp_print("recovering others...", global_rank)
            n_step = checkpoint["n_step"]
            start_n_epoch = checkpoint["n_epoch"]
            start_train_file_idx = checkpoint["start_train_file_idx"]
            best_ppl = checkpoint.get("best_ppl", float("inf"))
        else:
            n_step = 0
            start_n_epoch = 0
            start_train_file_idx = 0
            best_ppl = float("inf")
        OUTPUT_FILEID = checkpoint["output_fileid"]
        del checkpoint
        torch.cuda.empty_cache()
    else:
        n_step = 0
        start_n_epoch = 0
        start_train_file_idx = 0
        best_ppl = float("inf")

        # names
        OUTPUT_FILEID = "gpt2-ja-{}.seed_{}.{}".format(
            config.model_size,
            config.seed,
            time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
        )
    if config.filename_note:
        OUTPUT_FILEID += f".{config.filename_note}"
    
    # define logger
    def mlog(s):
        if global_rank == 0:
            if config.enable_log:
                if not os.path.exists(f"../log/{TASK}"):
                    os.makedirs(f"../log/{TASK}")
                with open(f"../log/{TASK}/{OUTPUT_FILEID}.log", "a+", encoding="utf-8") as log_f:
                    log_f.write(s+"\n")
            mp_print(s, global_rank)
    if config.enable_log:
        if global_rank == 0:
            tb_writer = SummaryWriter(
                log_dir=f"../log/{TASK}/{OUTPUT_FILEID}",
                max_queue=5
            )

    # log hyper parameters
    start_time = time.time()
    mlog("----- Hyper-parameters -----")
    for k, v in sorted(dict(config.__dict__).items()):
        mlog("{}: {}".format(k, v))

    for epoch_idx in range(start_n_epoch, config.n_epochs):
        for train_file_idx in range(start_train_file_idx, len(train_filepaths), config.n_train_files_per_group):
            
            group_train_filepaths = train_filepaths[train_file_idx:train_file_idx+config.n_train_files_per_group]
            
            with mp.Pool(processes=config.n_train_files_per_group) as pool:
                group_train_docs = pool.starmap(
                    load_docs_from_filepath, 
                    [(train_filepath, tokenizer) for train_filepath in group_train_filepaths]
                )
                train_docs = [doc for docs in group_train_docs for doc in docs]

            train_data_source = DataSource(config, tokenizer, train_docs, "train", randomize=True)
            mp_print(str(train_data_source.statistics), global_rank)
            # single gpu or cpu
            if config.world_size == 1 or not torch.cuda.is_available():
                train_data_sampler = RandomSampler(
                    train_data_source,
                    replacement=False
                )
                train_dataloader = torch.utils.data.DataLoader(
                    train_data_source,
                    batch_size=config.batch_size,
                    sampler=train_data_sampler,
                    num_workers=0,
                    collate_fn=collate_fn,
                    pin_memory=True
                )
            # multi gpus
            else:
                train_data_sampler = DistributedSampler(
                    train_data_source,
                    num_replicas=config.world_size,
                    rank=global_rank
                )
                train_dataloader = torch.utils.data.DataLoader(
                    train_data_source,
                    batch_size=config.batch_size,
                    sampler=train_data_sampler,
                    num_workers=0,
                    collate_fn=collate_fn,
                    pin_memory=False
                )

            if isinstance(train_data_sampler, DistributedSampler):
                train_data_sampler.set_epoch(epoch_idx)

            for batch_data in train_dataloader:
                n_step += 1

                # stop if reaches the maximum tranining step
                if n_step >= config.n_training_steps:
                    break

                # forward
                model.train()
                with amp.autocast():
                    loss, ppl = forward_step(model, tokenizer, batch_data)

                # update statisitcs
                trn_reporter.update_data({"ppl": ppl.item(), "loss": loss.item()})

                # backward
                loss /= config.n_accum_steps
                if config.use_amp:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                del loss

                if n_step % config.n_accum_steps == 0:
                    # clip gradient
                    if config.max_grad_norm > 0.0:
                        if config.use_amp:
                            scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

                    # update model parameters
                    if config.use_amp:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()

                    # zero gradients
                    optimizer.zero_grad()

                # check loss
                if n_step > 0 and n_step % config.check_loss_after_n_step == 0:
                    lr = list(lr_scheduler.optimizer.param_groups)[0]["lr"]
                    log_s = f"{time.time()-start_time:.2f}s Epoch {epoch_idx}, step {n_step}, lr {lr:.5g} - "
                    log_s += trn_reporter.to_string()
                    mlog(log_s)

                    if config.enable_log and global_rank == 0:
                        for k, v in trn_reporter.items():
                            tb_writer.add_scalar(f"{k}/train", np.mean(v), n_step)

                    trn_reporter.clear()

                # evaluation on dev dataset
                if global_rank == 0 and n_step > 0 and n_step % config.validate_after_n_step == 0:
                    
                    # forward
                    with torch.no_grad():
                        model.eval()
                        
                        # use only 1 gpu for evaluation in multi-gpu situation
                        if config.world_size > 1:
                            eval_model = model.module
                        else:
                            eval_model = model

                        for eval_batch_idx, eval_batch_data in enumerate(dev_dataloader):
                            with amp.autocast():
                                loss, ppl = forward_step(eval_model, tokenizer, eval_batch_data)
                            dev_reporter.update_data({"ppl": ppl.item(), "loss": loss.item()})

                            if eval_batch_idx == len(dev_dataloader) - 1:
                                break

                    log_s = f"\n<Dev> - {time.time()-start_time:.3f}s - "
                    log_s += dev_reporter.to_string()
                    mlog(log_s)

                    # Save model if it has better monitor measurement
                    if config.save_model:
                        if not os.path.exists(f"../data/model/{TASK}"):
                            os.makedirs(f"../data/model/{TASK}")

                        model_to_save = model.module if hasattr(model, 'module') else model

                        # save current model
                        checkpoint = {
                            "model": model_to_save.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "lr_scheduler": lr_scheduler.state_dict(),
                            "n_epoch": epoch_idx,
                            "n_step": n_step,
                            "start_train_file_idx": train_file_idx,
                            "output_fileid": OUTPUT_FILEID,
                            "best_ppl": best_ppl
                        }
                        torch.save(
                            checkpoint,
                            f"../data/model/{TASK}/{OUTPUT_FILEID}.checkpoint"
                        )
                        mlog(f"checkpoint saved to data/model/{TASK}/{OUTPUT_FILEID}.checkpoint")

                        # save best model
                        cur_ppl = dev_reporter.get_value("ppl")
                        if cur_ppl < best_ppl:
                            best_ppl = cur_ppl

                            torch.save(
                                checkpoint,
                                f"../data/model/{TASK}/{OUTPUT_FILEID}.best.checkpoint"
                            )
                            mlog(f"best checkpoint saved to data/model/{TASK}/{OUTPUT_FILEID}.best.checkpoint")

                    if config.enable_log:
                        for k, v in dev_reporter.items():
                            tb_writer.add_scalar(f"{k}/dev", np.mean(v), n_step)

                    dev_reporter.clear()

                # decay learning rate
                lr_scheduler.step()

        # reset starting training file index for every epoch (if might be set to a larger value if resuming from a checkpoint)
        start_train_file_idx = 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # modeling
    parser.add_argument("--model_size", type=str, default="small", help="for naming")
    parser.add_argument("--model_config_filepath", type=str, default="model/gpt2-ja-small-config.json", help="path to model config file")
    
    # training
    parser.add_argument("--seed", type=int, default=42, help="random initialization seed")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size for training. 20 for xsmall; 8 for small; 3 for medium.")
    parser.add_argument("--eval_batch_size", type=int, default=16, help="batch size for evaluation. Double of training batch size.")
    parser.add_argument("--n_train_files_per_group", type=int, default=10, help="number of files to load for every loading")
    parser.add_argument("--n_training_steps", type=int, default=4e6, help="number of maximum training steps. 1.6e6 for xsmall; 4e6 for small; 1.3e7 for medium.")
    parser.add_argument("--n_epochs", type=int, default=10, help="number of maximum training epochs")
    parser.add_argument("--n_warmup_steps", type=int, default=2e3, help="number of warmup steps")
    parser.add_argument("--balanced_corpora", type=str, help="use the same number of files for each training corpus when there are multiple corpora. In [None, 'undersample', 'oversample', 'custom_ratio'].")
    parser.add_argument("--small_data", type=str2bool, default=False, help="use a small portion of data for bugging")
    parser.add_argument("--max_seq_len", type=int, default=1024, help="maximum input sequence length")
    parser.add_argument("--n_accum_steps", type=int, default=8, help="number of gradient accumulation steps. 3 for xsmall; 8 for small; 20 for medium.")

    # multi-gpu
    parser.add_argument("--n_nodes", type=int, default=1, help="number of nodes; See pytorch DDP tutorial for details")
    parser.add_argument("--n_gpus", type=int, default=1, help="number of GPUs; See pytorch DDP tutorial for details")
    parser.add_argument("--node_rank", type=int, default=0, help="rank of starting node; See pytorch DDP tutorial for details")
    parser.add_argument("--master_port", type=str, default="12321", help="port of starting node; See pytorch DDP tutorial for details")

    # mixed precision
    parser.add_argument("--use_amp", type=str2bool, default=True, help="use mixed precision for training")

    # optimizer
    parser.add_argument("--l2_penalty", type=float, default=0.01, help="l2 penalty")
    parser.add_argument("--init_lr", type=float, default=6e-4, help="peak learning rate; 7e-4 for xsmall; 6e-4 for small; 3e-4 for medium.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="gradient clipping threshold")

    # management
    parser.add_argument("--corpora", type=str, nargs="+", default=["jp_cc100", "jp_wiki"], help="training corpora")
    parser.add_argument("--checkpoint_path", help="path to saved checkpoint file")
    parser.add_argument("--resume_training", type=str2bool, default=False, help="resume training from checkpoint or not")
    parser.add_argument("--enable_log", type=str2bool, default=False, help="save training log or not")
    parser.add_argument("--save_model", type=str2bool, default=False, help="save model to checkpoint or not")
    parser.add_argument("--check_loss_after_n_step", type=int, default=1e2, help="print loss after every this number of steps")
    parser.add_argument("--validate_after_n_step", type=int, default=5e3, help="validate model after every this number of steps")
    parser.add_argument("--filename_note", type=str, help="suffix of saved files' names")

    config = parser.parse_args()

    # multi-gpu config
    config.world_size = config.n_gpus * config.n_nodes
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = config.master_port

    # run multi-processes
    if config.world_size > 1:
        mp.spawn(train, nprocs=config.n_gpus, args=(config,))
    else:
        train(0, config)
