from datasets import load_metric, load_dataset
import nltk


try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    nltk.download("punkt", quiet=True)


def load_text_dataset(datafile, cnndm_dataset, split='train', text_col='article', summary_col='highlights'):
    """Load a BertSumExt or ROUGE-generated txt file into a Huggingface dataset. `cnndm_dataset` should be 
    a loaded train, validation, or test CNN\DM dataset specified by the `split` argument.
    """
    # Entries are expected to be separated by newline
    dataset = load_dataset('text', data_files={split: datafile}, split=split)

    # Rename `text` column to `article` and add the `highlights` column from CNN/DM
    dataset = dataset.rename_column('text', text_col)
    dataset = dataset.add_column(summary_col, cnndm_dataset[summary_col])

    return dataset


def postprocess_text(preds, labels):
    """Postprocess output to prepare for ROUGE evaluation."""
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLsum expects newline after each sentence
    preds = ['\n'.join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ['\n'.join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels


def score_rouge(preds, refs):
    """Compute ROUGE scores on a list of predicted summaries given reference summaries."""
    metric = load_metric('rouge')
    preds, refs = postprocess_text(preds, refs)
    result = metric.compute(predictions=preds, references=refs, use_stemmer=True)

    # Extract only mid scores from ROUGE
    # Note that rougeLsum should be used as the ROUGE-L score
    result = {k: round(v.mid.fmeasure * 100, 2) for k, v in result.items()}
    return result
    
