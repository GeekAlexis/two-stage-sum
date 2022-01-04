# Generate the Stage-1 ground truth using the ROUGE metric

import math
from concurrent.futures import ProcessPoolExecutor
from datasets import load_dataset
from tqdm import tqdm
from rouge_score import rouge_scorer
from nltk.tokenize import sent_tokenize

def inner(scorer, article, highlight, threshold):
        article = article.replace("\n", " ")
        sentences = sent_tokenize(article)
        score_sentences = sorted([
            (-scorer.score(highlight, sentence)["rougeLsum"].recall, sentence) for sentence in sentences
        ])

        highlight = highlight.replace("\n", " ")
        highlight_sentences = sent_tokenize(highlight)
        highlight = "\n".join(highlight_sentences)

        res_sentences = []
        prev_score = -math.inf
        for _, sentence in score_sentences:
            res_sentences.append(sentence)
            curr_score = scorer.score(highlight, "\n".join(res_sentences))["rougeLsum"].recall

            if curr_score - prev_score <= threshold:
                res_sentences = res_sentences[:-1] # remove the latest sentence
                break

            prev_score = curr_score

        return " ".join(res_sentences) + "\n"

def generate_label(scorer, dataset, subset_name, threshold=0):
    total_len = len(dataset[subset_name])
    articles = map(lambda x: x["article"], dataset[subset_name])
    highlights = map(lambda x: x["highlights"], dataset[subset_name])

    futures = []
    with ProcessPoolExecutor(max_workers=5) as p:
        for article, highlight in tqdm(zip(articles, highlights), total=total_len):
            future = p.submit(inner, scorer, article, highlight, threshold)
            futures.append(future)

        with open(f"results/rouge-gen-label-{subset_name}.txt", "w+", encoding="utf-8") as f:
            for future in tqdm(futures):
                res = future.result()
                f.write(res)

if __name__ == "__main__":
    scorer = rouge_scorer.RougeScorer(["rougeLsum"], use_stemmer=True)
    dataset = load_dataset("cnn_dailymail", "3.0.0")
    generate_label(scorer, dataset, "train", 0)
    generate_label(scorer, dataset, "validation", 0)
    generate_label(scorer, dataset, "test", 0)
