## Extraction with BertSumExt

### Directory structure

- PreSumm-dev: modified [PreSumm repo](https://github.com/nlpyang/PreSumm)
- src: Several scripts to pre/postprocess the data, or calculate the metrics

### Modification of BertSumExt

The uploaded version contains the code that we used for the final output,
i.e., using 1.01 as the threshold.

To modify the threshold, modify `PreSumm-dev/src/models/trainer_ext.py` line 275, replace `1.01` as any threshold you want.

To reproduce the results with output sentence length constraints,
replace `(idx[j] >= len(sent_scores[0]) or sent_scores[0, idx[j]] < 1.01))` with `(len(_pred) == LEN)`,
where `LEN` is the length constraint.

### Generate BertSumExt outputs

Download [bertext_cnndm_transformer](https://drive.google.com/file/d/1kKWoV0QCbeIuFt85beQgJ4v0lujaXobJ/view) and uncompress in the directory `PreSumm-dev/models`.

The BERTSumExt Model requires the input to be sentences separated with `[SEP] [CLS]`, run the following command in the bertsumext root directory to pre-process.

```bash
python -m src.preprocess
```

Then, run the following commands in the `PreSumm-dev/src/` directory to get the raw output of BertSumExt

```bash
python train.py -task ext -mode test_text -text_src ../../results/sent-tokenized-test.txt -test_from ../models/bertext_cnndm_transformer/bertext_cnndm_transformer.pt -result_path ../../results/raw-output-test.txt -visible_gpus 0 && \
python train.py -task ext -mode test_text -text_src ../../results/sent-tokenized-validation.txt -test_from ../models/bertext_cnndm_transformer/bertext_cnndm_transformer.pt -result_path ../../results/raw-output-validation.txt -visible_gpus 0 && \
python train.py -task ext -mode test_text -text_src ../../results/sent-tokenized-train.txt -test_from ../models/bertext_cnndm_transformer/bertext_cnndm_transformer.pt -result_path ../../results/raw-output-train.txt -visible_gpus 0
```

The results will be in `results/raw-output-SUBSET.txt_step-1.candidate`, each sentence is separated by `<q>`.
Run the following command in the bertsumext root directory to get the final results:

```bash
python -m src.postprocess
```

### Generate Oracle (greedy ROUGE recall)

```bash
python -m src.gen_stage_1
```

### Metric calculation

Modify the file path in `src/calc_rouge_2` and run

```bash
python -m src.calc_rouge_2
```
