# Calculate ROUGE metrics on the BERTSumExt output on test data

from rouge_score import rouge_scorer
from nltk.tokenize import sent_tokenize


scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeLsum"], use_stemmer=True)

from datasets import load_dataset
from tqdm import tqdm

dataset = load_dataset("cnn_dailymail", "3.0.0")
golds = list(map(lambda x: x["highlights"], dataset["test"]))

sent_3_res = []
sent_4_res = []
sent_5_res = []

with open("results/5-sent/bertsumext-out-test.txt", "r", encoding="utf-8") as f:
    for pred, gold in tqdm(zip(f, golds), total=len(golds)):
        gold_sent = sent_tokenize(gold)
        gold_formatted = "\n".join(gold_sent)
        pred_sent = sent_tokenize(pred)
        sent_3_res.append(scorer.score(gold_formatted, "\n".join(pred_sent[:3])))
        sent_4_res.append(scorer.score(gold_formatted, "\n".join(pred_sent[:4])))
        sent_5_res.append(scorer.score(gold_formatted, "\n".join(pred_sent)))

def calc_avg(res):
    for metric in ["1", "2", "Lsum"]:
        p = sum(r[f"rouge{metric}"].precision for r in res) / len(res)
        r = sum(r[f"rouge{metric}"].recall for r in res) / len(res)
        f = sum(r[f"rouge{metric}"].fmeasure for r in res) / len(res)
        print(metric)
        print(p, r, f)
        print()

print(3)
calc_avg(sent_3_res)
print()

print(4)
calc_avg(sent_4_res)
print()

print(5)
calc_avg(sent_5_res)