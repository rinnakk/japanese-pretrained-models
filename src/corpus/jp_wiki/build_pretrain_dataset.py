# coding=utf-8
# Copyright 2021 Masatoshi Suzuki (@singletongue)
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

import json
import argparse
import os
import re
import unicodedata
import gzip
from urllib.request import urlretrieve


from tqdm import tqdm
import fugashi

from corpus.jp_wiki.config import Config

config = Config()


class MeCabSentenceSplitter(object):
    def __init__(self, mecab_dict_path=None):
        if mecab_dict_path is not None:
            self.mecab = fugashi.Tagger('-d {}'.format(mecab_dict_path))
        else:
            self.mecab = fugashi.Tagger()

    def __call__(self, text):
        sentences = []
        start = 0
        end = 0
        for line in self.mecab.parse(text).split("\n"):
            if line == "EOS":
                if len(text[start:]) > 0:
                    sentences.append(text[start:])
                break

            token, token_info = line.split("\t", maxsplit=1)
            end = text.index(token, end) + len(token)
            if "記号" in token_info and "句点" in token_info:
                sentences.append(text[start:end])
                start = end

        return sentences


def download_data():
    if not os.path.exists(config.raw_data_path):
        print(f'Downloading {config.download_link} to {config.raw_data_path}')
        urlretrieve(config.download_link, config.raw_data_path)
        print(f'Successfully downloaded {config.raw_data_path}')


def preprocess_text(text, title=None):
    text = unicodedata.normalize("NFKC", text)

    # remove invisible characters
    text = "".join(c for c in text if c.isprintable())

    # remove templates
    text = re.sub(r"\[\d+?\]", "", text)
    text = re.sub(r"\[要.+?\]", "", text)
    text = re.sub(r"\{\{+[^{}]+?\}\}+", "", text)

    # remove navigation
    if title is not None:
        text = re.sub(r"^.+? \> " + re.escape(title), "", text)

    # remove footnotes
    text = re.sub(r" \^ .+", "", text)
    # remove annotations
    text = re.sub(r"\[(要出典|リンク切れ|.+?\?)\]", "", text)

    text = re.sub(r"\s+", " ", text).strip()
    return text


def filter_text(text):
    # filter out text containing equations
    if "\displaystyle" in text:
        return False

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mecab_dict_path", type=str)
    parser.add_argument('--min_length', type=int, default=1, help='only extract sentences with no less than N characters')
    parser.add_argument('--max_length', type=int, default=1024, help='only extract sentences with no more than N characters')

    args = parser.parse_args()

    if not os.path.exists(config.raw_data_dir):
        os.makedirs(config.raw_data_dir)
    
    download_data()

    sent_splitter = MeCabSentenceSplitter(args.mecab_dict_path)

    with gzip.open(config.raw_data_path, "rt") as input_file, \
         open(config.extracted_data_path, "w") as output_file:
        for line in tqdm(input_file):
            json_item = json.loads(line)
            text = json_item.get("text")
            if text is None:
                continue

            title = json_item.get("title")
            text = preprocess_text(text, title=title)

            is_processed = False
            for sentence in sent_splitter(text):
                sentence = sentence.strip()
                if len(sentence) < args.min_length:
                    continue
                if len(sentence) > args.max_length:
                    continue
                if not filter_text(sentence):
                    continue

                assert "\n" not in text
                assert sentence != ""
                print(sentence, file=output_file)
                is_processed = True

            if is_processed:
                print("", file=output_file)