# Remove the <q> from the outputs

for s in ["train", "test", "validation"]:
    try:
        in_fname = f"results/raw-output-{s}.txt_step-1.candidate"
        out_fname = f"results/bertsumext-out-{s}.txt"

        with open(in_fname, "r", encoding="utf-8") as in_f:
            with open(out_fname, "w+", encoding="utf-8") as out_f:
                for i in in_f:
                    out_f.write(i.replace("<q>", " "))
    except:
        pass
