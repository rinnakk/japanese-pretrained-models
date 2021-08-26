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

import os
from tqdm import tqdm

from corpus.jp_wiki.config import Config

config = Config()
N_LINES_PER_FILE = 1e6

if __name__ == "__main__":
    if not os.path.exists(f"{config.doc_data_dir}"):
        os.makedirs(f"{config.doc_data_dir}")

    output_file_id = 0
    output_file = open(f"{config.doc_data_dir}/{output_file_id}.txt", "w+", encoding="utf-8")
    cur_n_lines = 0

    with open(f"{config.extracted_data_path}", "r", encoding="utf-8") as input_file:
        with tqdm() as pbar:
            for line in input_file:
                line = line.strip()
                
                output_file.write(line + "\n")
                cur_n_lines += 1

                if line == "":
                    output_file.write
                    if cur_n_lines >= N_LINES_PER_FILE:
                        output_file.flush()
                        output_file.close()
                        output_file_id += 1
                        cur_n_lines = 0
                        output_file = open(f"{config.doc_data_dir}/{output_file_id}.txt", "w+", encoding="utf-8")
                        pbar.set_description(f"created {output_file_id} files")
                    

