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

class Config(object):
    def __init__(self):
        self.corpus_name = "jp_wiki"

        # Management
        self.download_link = "https://dumps.wikimedia.org/other/cirrussearch/20210329/jawiki-20210329-cirrussearch-content.json.gz"
        self.raw_data_dir = "../data/jp_wiki/raw_data"
        self.raw_data_path = f"{self.raw_data_dir}/wiki.json.gz"
        self.extracted_data_path = f"{self.raw_data_dir}/wiki.extracted.txt"
        self.doc_data_dir = "../data/jp_wiki/doc_data"
