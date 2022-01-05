## Abstraction with BART
### Fine-tune BART directly on CNN/DM
```bash
python bart/finetune_bart.py 
```
By default, model checkpoint will be saved to `bart/checkpoints/checkpoint.pt`.
Only single GPU training is currently supported.

### Fine-tune BART Paraphraser on BertSumExt outputs
```bash
python bart/finetune_bart.py --train-ext-output output/BertSumExt/score-1.01/bertsumext-out-train.txt --validation-ext-output output/BertSumExt/Score-0.5/bertsumext-out-validation.txt
```
Note that BertSumExt outputs on the training set may need to be generated first.
By default, model checkpoint will be saved to `bart/checkpoints/checkpoint.pt`.

### Evaluate on test set and dump output
```bash
python bart/score_bart.py --weight bart/checkpoints/best.pt --output-dir output/bertsumext-bart-out
```

### Evaluate on test set output (ROUGE 1/2/L F1)
```bash
python bart/score_output.py --output-dir output/bertsumext-bart-out
```

The above commands should be run in the root directory.
Use the commandline option `-h` for more information about script usage.
