import argparse
import math
import shutil
from collections import defaultdict
from pathlib import Path
import numpy as np
from tqdm.auto import tqdm

import torch
from torch.utils.data import DataLoader

from accelerate import Accelerator
from datasets import load_dataset, load_metric
from transformers import BartTokenizerFast
from transformers import DataCollatorForSeq2Seq
from transformers import AdamW, get_scheduler

from common import load_text_dataset, postprocess_text
from models import BartParaphraser, LabelSmoothingLoss


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune BART on summarization task.")
    parser.add_argument(
        "--train-file", default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation-file", default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--train-ext-output", default=None, help="A txt file containing the extracted sentences on the training data for two-stage training."
    )
    parser.add_argument(
        "--validation-ext-output", default=None, help="A txt file containing the extracted sentences on the validation data for two-stage training."
    )
    parser.add_argument(
        "--subsample-validation", type=int, default=2000, help="Number of validation samples to use to speedup evaluation."
    )
    parser.add_argument(
        "--preprocessing-workers",
        type=int,
        default=8,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--model-arch",
        default='facebook/bart-large',
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to model checkpoint to resume training.",
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
        "--train-batch-size",
        type=int,
        default=2,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay to use.")
    parser.add_argument("--label-smoothing", type=float, default=0.1, help="Label smoothing factor to use.")
    parser.add_argument(
        "--num-training-steps",
        type=int,
        default=15000,
        help="Total number of training steps to perform.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=16,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--num-warmup-steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--eval-every", type=int, default=1000, help="Interval in steps to evaluate on validation set."
    )
    parser.add_argument(
        "--use-rouge-for-eval", action='store_true', help="Use ROUGE to evaluate on validation set. Otherwise, loss will be used."
    )
    parser.add_argument(
        "--save-dir", default=Path(__file__).parent / 'checkpoints',
        help="Directory to save the model checkpoint.",
    )
    args = parser.parse_args()

    return args


def save_checkpoint(state, is_best, dir):
    save_path = Path(dir) / 'checkpoint.pt'
    torch.save(state, save_path)
    if is_best:
        shutil.copyfile(save_path, Path(dir) / 'best.pt')


def train(arch, tokenizer,
          train_dataset, val_dataset,
          checkpoint=None,
          batch_size=2,
          eval_batch_size=8,
          learning_rate=3e-05,
          weight_decay=0.01,
          gradient_accumulation_steps=16,
          label_smoothing=0.1,
          num_training_steps=18000,
          num_warmup_steps=500,
          eval_every=1000,
          use_rouge_for_eval=False,
          save_dir='./checkpoints'):
    
    # Initialize the accelerator. We will let the accelerator handle device placement for us.
    accelerator = Accelerator(fp16=True)
    print(accelerator.state)

    # Load from checkpoint if given
    if checkpoint is None:
        model = BartParaphraser(arch)
    else:
        model = BartParaphraser(checkpoint['arch'])
        model.load_state_dict(checkpoint['model'])

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model, # provide mode so that BART's prepare_decoder_input_ids_from_labels can be used
        label_pad_token_id=-100, # ignore padding in training loss
        pad_to_multiple_of=8 if accelerator.use_fp16 else None,
    )

    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator,
                                  batch_size=batch_size, num_workers=2)
    val_dataloader = DataLoader(val_dataset, collate_fn=data_collator,
                                batch_size=eval_batch_size, num_workers=2)

    # Optimizer and loss
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    criterion = LabelSmoothingLoss(epsilon=label_smoothing)

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader
    )

    # Scheduler
    lr_scheduler = get_scheduler(
        name='linear',
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    completed_steps = 0
    best_rougel = 0.
    best_val_loss = math.inf
    metrics_hist = defaultdict(list)
    # Load from checkpoint if given
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['scheduler'])
        completed_steps = checkpoint['step']
        best_rougel = checkpoint.get('best_rougel')
        best_val_loss = checkpoint.get('best_val_loss')
        metrics_hist = checkpoint['metrics']

    # Metric
    if use_rouge_for_eval:
        metric = load_metric('rouge')

    # Train!
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    num_epochs = math.ceil(num_training_steps / num_update_steps_per_epoch)
    completed_epochs = completed_steps // num_update_steps_per_epoch
    total_batch_size = batch_size * gradient_accumulation_steps

    print("***** Running training *****")
    print(f"  Num examples = {len(train_dataset)}")
    print(f"  Num Epochs = {num_epochs}")
    print(f"  Total batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    print(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    print(f"  Total optimization steps = {num_training_steps}")

    progress_bar = tqdm(range(num_training_steps))
    progress_bar.update(completed_steps)
    running_loss = 0.
    for epoch in range(completed_epochs, num_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            outputs = model(input_ids=batch['input_ids'],
                            attention_mask=batch['attention_mask'],
                            decoder_input_ids=batch['decoder_input_ids'])

            with accelerator.autocast():
                loss = criterion(outputs.logits, batch['labels'])

            loss = loss / gradient_accumulation_steps
            accelerator.backward(loss)
            running_loss += loss.item()

            if step % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

                # Calculate ROUGE on val set
                if completed_steps % eval_every == 0:
                    model.eval()
                    val_running_loss = 0.
                    for step, batch in enumerate(val_dataloader):
                        with torch.no_grad():          
                            if use_rouge_for_eval:
                                generated_tokens = accelerator.unwrap_model(model).generate(
                                    input_ids=batch['input_ids'],
                                    attention_mask=batch['attention_mask'],
                                    num_beams=2,
                                    min_length=56,
                                    max_length=142,
                                    length_penalty=2.0,
                                    early_stopping=True,
                                    use_cache=True
                                )

                                generated_tokens = generated_tokens.cpu().numpy()
                                labels = batch['labels'].cpu().numpy()

                                # Replace -100 in the labels as we can't decode them.
                                labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
                                decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                                decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
                                decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

                                metric.add_batch(predictions=decoded_preds, references=decoded_labels)
                            else:
                                outputs = model(input_ids=batch['input_ids'],
                                                attention_mask=batch['attention_mask'],
                                                decoder_input_ids=batch['decoder_input_ids'])
                                with accelerator.autocast():
                                    loss = criterion(outputs.logits, batch['labels'])
                                val_running_loss += loss.item()

                    avg_train_loss = running_loss / eval_every
                    metrics_hist['completed_steps'].append(completed_steps)
                    metrics_hist['train_loss'].append(avg_train_loss)

                    if use_rouge_for_eval:
                        result = metric.compute(use_stemmer=True)
                        result = {k: round(v.mid.fmeasure * 100, 2) for k, v in result.items()}
                        metrics_hist['val_rouge1'].append(result['rouge1'])
                        metrics_hist['val_rouge2'].append(result['rouge1'])
                        metrics_hist['val_rougel'].append(result['rougeLsum'])
                        tqdm.write(f"Epoch {epoch + 1}/{num_epochs}, Step {completed_steps}/{num_training_steps}, "
                                   f"Train Loss: {avg_train_loss:.4f}, Val ROUGE-L: {result['rougeLsum']}")
                        
                        # Checkpoint
                        is_best = result['rougeLsum'] > best_rougel
                        if is_best:
                            best_rougel = result['rougeLsum']

                        unwrapped_model = accelerator.unwrap_model(model)
                        save_checkpoint({
                            'step': completed_steps,
                            'arch': unwrapped_model.arch,
                            'model': unwrapped_model.state_dict(),
                            'optimizer'  : optimizer.state_dict(),
                            'scheduler'  : lr_scheduler.state_dict(),
                            'best_rougel': best_rougel,
                            'metrics'    : metrics_hist,
                        }, is_best, save_dir)
                        tqdm.write(f'Checkpoint saved to {save_dir}')
                    else:
                        avg_val_loss = val_running_loss / len(val_dataloader)
                        metrics_hist['val_loss'].append(avg_val_loss)
                        tqdm.write(f"Epoch {epoch + 1}/{num_epochs}, Step {completed_steps}/{num_training_steps}, "
                                   f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
                        
                        # Checkpoint
                        is_best = avg_val_loss < best_val_loss
                        if is_best:
                            best_val_loss = avg_val_loss

                        unwrapped_model = accelerator.unwrap_model(model)

                        save_checkpoint({
                            'step': completed_steps,
                            'arch': unwrapped_model.arch,
                            'model': unwrapped_model.state_dict(),
                            'optimizer'    : optimizer.state_dict(),
                            'scheduler'    : lr_scheduler.state_dict(),
                            'best_val_loss': best_val_loss,
                            'metrics'      : metrics_hist,
                        }, is_best, save_dir)
                        tqdm.write(f'Checkpoint saved to {save_dir}')

                    running_loss = 0.

            if completed_steps >= num_training_steps:
                break


def main():
    args = parse_args()

    if args.train_file:
        extension = args.train_file.split('.')[-1]
        train_dataset = load_dataset(extension, data_files={'train': args.train_file}, split='train')
    else:
        train_dataset = load_dataset("cnn_dailymail", "3.0.0", split='train')

    if args.validation_file:
        extension = args.validation_file.split('.')[-1]
        val_dataset = load_dataset(extension, data_files={'validation': args.validation_file}, split='validation')
    else:
        val_dataset = load_dataset("cnn_dailymail", "3.0.0", split='validation')

    if args.train_ext_output and args.validation_ext_output:
        train_dataset = load_text_dataset(args.train_ext_output, train_dataset, split='train',
                                          text_col=args.text_column, summary_col=args.summary_column)
        val_dataset = load_text_dataset(args.validation_ext_output, val_dataset, split='validation',
                                        text_col=args.text_column, summary_col=args.summary_column)

    checkpoint = None
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        tokenizer = BartTokenizerFast.from_pretrained(checkpoint['arch'])
    else:
        tokenizer = BartTokenizerFast.from_pretrained(args.model_arch)

    def preprocess_dataset(batch):
        # Use dynamic padding so no padding here
        model_inputs = tokenizer(batch[args.text_column], truncation=True)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(batch[args.summary_column], truncation=True)

        model_inputs['labels'] = labels['input_ids']
        return model_inputs

    processed_train = train_dataset.map(
        preprocess_dataset,
        batched=True,
        remove_columns=train_dataset.column_names, # remove old columns for model
        num_proc=args.preprocessing_workers,
        desc="Running tokenizer on train set"
    )

    # Subsample instances from the validation set
    val_subset = val_dataset.shuffle(seed=42).select(range(args.subsample_validation))
    processed_val = val_subset.map(
        preprocess_dataset,
        batched=True,
        remove_columns=val_dataset.column_names, # remove old columns for model
        num_proc=args.preprocessing_workers,
        desc="Running tokenizer on validation set"
    )

    train(args.model_arch, tokenizer, processed_train, processed_val,
          checkpoint=checkpoint,
          batch_size=args.train_batch_size,
          eval_batch_size=args.eval_batch_size,
          learning_rate=args.learning_rate,
          weight_decay=args.weight_decay,
          gradient_accumulation_steps=args.gradient_accumulation_steps,
          label_smoothing=args.label_smoothing,
          num_training_steps=args.num_training_steps,
          num_warmup_steps=args.num_warmup_steps,
          eval_every=args.eval_every,
          use_rouge_for_eval=args.use_rouge_for_eval,
          save_dir=args.save_dir)


if __name__ == '__main__':
    main()
