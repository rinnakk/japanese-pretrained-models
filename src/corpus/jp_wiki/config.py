class Config(object):
    def __init__(self):
        self.corpus_name = "jp_wiki"

        # Management
        self.download_link = "https://dumps.wikimedia.org/other/cirrussearch/20210329/jawiki-20210329-cirrussearch-content.json.gz"
        self.raw_data_dir = "../data/jp_wiki/raw_data"
        self.raw_data_path = f"{self.raw_data_dir}/wiki.json.gz"
        self.extracted_data_path = f"{self.raw_data_dir}/wiki.extracted.txt"
        self.doc_data_dir = "../data/jp_wiki/doc_data"
