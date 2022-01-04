import random
import argparse
from tqdm.auto import tqdm
import nltk
from datasets import load_dataset
from common import score_rouge


try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    nltk.download("punkt", quiet=True)


def random_word_picker(article):
    '''Weak baseline - random word picker: randomly select a few of words'''
    words = article.split()
    n_word = int(len(words) / 30)
    return ' '.join([random.choice(words) for _ in range(n_word)])


def random3(article):
    '''Weak baseline - random-3: randomly select 3 sentences'''
    sents = nltk.sent_tokenize(article)
    return ' '.join(random.sample(sents, min(len(sents), 3)))


def lead3(article):
    '''Strong baseline - lead-3: pick the first 3 sentenses'''
    return ' '.join(nltk.sent_tokenize(article)[:3])


def main():
    parser = argparse.ArgumentParser(description="Evaluate baselines on a given summarization test set.")
    parser.add_argument(
        "--test-file", default=None, help="A csv or a json file containing the test data."
    )
    parser.add_argument(
        "--text-column",
        default='article',
        help="The name of the column in the datasets containing the full texts (for summarization).",
    )
    parser.add_argument(
        "--summary-column",
        default='highlights',
        help="The name of the column in the datasets containing the summaries (for summarization).",
    )
    args = parser.parse_args()

    if args.test_file:
        extension = args.test_file.split('.')[-1]
        test_dataset = load_dataset(extension, data_files={'test': args.test_file}, split='test')
    else:
        test_dataset = load_dataset("cnn_dailymail", "3.0.0", split='test')

    preds_random_word = [random_word_picker(article) for article in tqdm(test_dataset[args.text_column])]
    print('Random word picker:', score_rouge(preds_random_word, test_dataset[args.summary_column]))

    preds_random3 = [random3(article) for article in tqdm(test_dataset[args.text_column])]
    print('Random-3:', score_rouge(preds_random3, test_dataset[args.summary_column]))

    preds_lead3 = [lead3(article) for article in tqdm(test_dataset[args.text_column])]
    print('Lead-3:', score_rouge(preds_lead3, test_dataset[args.summary_column]))


if __name__ == '__main__':
    main()
