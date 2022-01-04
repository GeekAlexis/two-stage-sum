from pathlib import Path
import argparse
import torch
from datasets import load_dataset, load_metric
from transformers import BartTokenizerFast, DataCollatorWithPadding

from common import load_text_dataset, postprocess_text
from models import BartParaphraser


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def summarize(text, model, tokenizer):
    """Produce summary for one piece of text."""
    input_ids = tokenizer(text, truncation=True, return_tensors='pt').input_ids
    input_ids = input_ids.to(device)
    # We only need input_ids here b/c we don't have to pad for one article
    summary_ids = model.generate(input_ids,
                                 num_beams=4,
                                 no_repeat_ngram_size=3,
                                 min_length=56,
                                 max_length=142,
                                 length_penalty=2.0,
                                 early_stopping=True,
                                 decoder_start_token_id=tokenizer.eos_token_id, # BART uses EOS as the first token for decoder
                                 use_cache=True)

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


def evaluate(dataset, model, tokenizer, batch_size=8, summary_col='highlights'):
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors='pt')
    metric = load_metric('rouge')
    inp_column_names = ['input_ids', 'attention_mask']

    def batch_summarize(batch):        
        # Pad to max length in the batch
        inputs = data_collator({name: batch[name] for name in inp_column_names})
        # Move input_ids and attention_mask to GPU
        inputs = {k: v.to(device) for k, v in inputs.items()}

        summary_ids = model.generate(**inputs,
                                     num_beams=4,
                                     no_repeat_ngram_size=3,
                                     min_length=56,
                                     max_length=142,
                                     length_penalty=2.0,
                                     early_stopping=True,
                                     decoder_start_token_id=tokenizer.eos_token_id, # BART uses EOS as the first token for decoder
                                     use_cache=True)

        # Remove special tokens like <s>, </s>, and padding
        batch['pred'] = tokenizer.batch_decode(summary_ids, skip_special_tokens=True)

        preds, refs = postprocess_text(batch['pred'], batch[summary_col])
        metric.add_batch(predictions=preds, references=refs)
        return batch

    outputs = dataset.map(batch_summarize, batched=True, batch_size=batch_size,
                          remove_columns=inp_column_names)
    result = metric.compute(use_stemmer=True)
    result = {k: round(v.mid.fmeasure * 100, 2) for k, v in result.items()}
    return outputs, result


def main():
    parser = argparse.ArgumentParser(description="Evaluate BART on a given summarization test set.")
    parser.add_argument(
        "--test-file", default=None, help="A csv or a json file containing the test data."
    )
    parser.add_argument(
        "--test-ext-output", default=None, help="A txt file containing the extracted sentences on the test data for two-stage evaluation."
    )
    parser.add_argument(
        "--output-dir", default=Path(__file__).resolve().parents[2] / 'output' / 'bart-reimplement-out',
        help="Path to a output directory to save predicted results."
    )
    parser.add_argument(
        "--weight",
        default=Path(__file__).parent / 'checkpoints' / 'best.pt',
        help="Path to the trained model weight.",
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
    parser.add_argument(
        "--preprocessing-workers",
        type=int,
        default=8,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    args = parser.parse_args()

    if args.test_file:
        extension = args.test_file.split('.')[-1]
        test_dataset = load_dataset(extension, data_files={'test': args.test_file}, split='test')
    else:
        test_dataset = load_dataset("cnn_dailymail", "3.0.0", split='test')

    if args.test_ext_output:
        test_dataset = load_text_dataset(args.test_ext_output, test_dataset, split='test',
                                         text_col=args.text_column, summary_col=args.summary_column)

    checkpoint = torch.load(args.weight)
    model = BartParaphraser(checkpoint['arch'])
    model.load_state_dict(checkpoint['model'])
    model.eval()
    tokenizer = BartTokenizerFast.from_pretrained(checkpoint['arch'])

    def tokenize_dataset(batch):
        # Truncate at BART max_position_embeddings 1024
        # We want to use dynamic padding so no padding here
        return tokenizer(batch[args.text_column], truncation=True)

    processed_test = test_dataset.map(
        tokenize_dataset,
        batched=True,
        num_proc=args.preprocessing_workers,
        desc="Running tokenizer on test set"
    )

    model.to(device)
    outputs, result = evaluate(processed_test, model, tokenizer,
                               batch_size=args.eval_batch_size, summary_col=args.summary_column)
    outputs.save_to_disk(args.output_dir)
    print(result)


if __name__ == '__main__':
    main()
