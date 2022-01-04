# Sentence split and add [CLS] [SEP] seperators for BERTSumExt
from nltk.tokenize import sent_tokenize
from datasets import load_dataset
from tqdm import tqdm

dataset = load_dataset("cnn_dailymail", "3.0.0")

for s in ["train", "validation", "test"]:
    data = dataset[s]
    with open(f"results/sent-tokenized-{s}.txt", "w+", encoding="utf-8") as f:
        for item in tqdm(data):
            f.write(" [CLS] [SEP] ".join(sent_tokenize(item["article"])).replace("\n", " ") + "\n")
