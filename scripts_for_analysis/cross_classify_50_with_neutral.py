# A script that iterates over lines of a given input file and produces a classifier's output for each line in the file.

import transformers
import sys
import gzip
from tqdm import tqdm
import json


device = "cuda:0"
en_transformer_model_path = "s-nlp/roberta-base-formality-ranker"
cocoa_transformer_model_path = "/mnt/gpu_data3/formality-experiments/models/test_mdeberta_500"

languages = ["de", "en", "fr", "it", "es", "pl", "pt", "nl"]

result = {
    key: {
        "formal": {lang: {"formal": 0, "informal": 0, "neutral": 0} for lang in languages},
        "informal": {lang: {"formal": 0, "informal": 0, "neutral": 0} for lang in languages},
    } for key in languages
}

tokenizers = {
    "en": transformers.AutoTokenizer.from_pretrained(en_transformer_model_path),
    "cocoa": transformers.AutoTokenizer.from_pretrained(cocoa_transformer_model_path)
}
models = {
    "en": transformers.AutoModelForSequenceClassification.from_pretrained(en_transformer_model_path),
    "cocoa":  transformers.AutoModelForSequenceClassification.from_pretrained(cocoa_transformer_model_path)
}

pipes = {
    "en": transformers.TextClassificationPipeline(model=models["en"], tokenizer=tokenizers["en"], device=0),
    "cocoa": transformers.TextClassificationPipeline(model=models["cocoa"], tokenizer=tokenizers["cocoa"], device=0)
}

models['en'].eval()
models['cocoa'].eval()
models['en'].to(device)
models['cocoa'].to(device)

formality_labels = {
    "LABEL_0": "INFORMAL",
    "LABEL_1": "FORMAL",
    "LABEL_2": "NEUTRAL",
}


def classify_batch(inputs, en=True):
    pipe_id = "en" if en else "cocoa"
    return pipes[pipe_id](inputs)

def interpret_classification(x, src_lang, tgt_lang, category):
    if src_lang == "en":
        assert x['label'] in ["informal", "formal"]
        if category == "informal":
            if x["label"] == "informal" and x["score"] >= 0.67:
                result[tgt_lang]["informal"][src_lang]["informal"] += 1
            elif x["label"] == "formal" and x["score"] >= 0.67:
                result[tgt_lang]["informal"][src_lang]["formal"] += 1
            else:
                result[tgt_lang]["informal"][src_lang]["neutral"] += 1
        else:
            if x["label"] == "formal" and x["score"] >= 0.67:
                result[tgt_lang]["formal"][src_lang]["formal"] += 1
            elif x["label"] == "informal" and x["score"] >= 0.67:
                result[tgt_lang]["formal"][src_lang]["informal"] += 1
            else:
                result[tgt_lang]["formal"][src_lang]["neutral"] += 1
    else:
        assert x['label'] in ["LABEL_0", "LABEL_1", "LABEL_2"]
        if category == "informal":
            if x['label'] == "LABEL_0":
                result[tgt_lang]["informal"][src_lang]["informal"] += 1
            elif x['label'] == "LABEL_1":
                result[tgt_lang]["informal"][src_lang]["formal"] += 1
            elif x['label'] == "LABEL_2":
                result[tgt_lang]["informal"][src_lang]["neutral"] += 1
        else:
            if x['label'] == "LABEL_1":
                result[tgt_lang]["formal"][src_lang]["formal"] += 1
            elif x['label'] == "LABEL_0":
                result[tgt_lang]["formal"][src_lang]["informal"] += 1
            elif x['label'] == "LABEL_2":
                result[tgt_lang]["formal"][src_lang]["neutral"] += 1

batch_size = 10000


for category in ["formal", "informal"]:
    for tgt_lang in languages:
        print(f"Processing target language: {tgt_lang}")
        base_path = f"/mnt/gpu_data3/formality_dataset/all2x_deberta_classified/mt_fame_50k/all2{tgt_lang}"
        for src_lang in languages:
            print(f"\tProcessing source language: {src_lang}")
            print(result)
            # print(json.dumps(result, sort_keys=False, indent=4))
            if src_lang == tgt_lang:
                continue
            src_batch = []
            file_path = f"{base_path}/{src_lang}-{tgt_lang}.{category}.tsv"
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.decode("utf-8") if type(line) is not str else line
                    splitted_line = line.split("\t")
                    source_text = splitted_line[0].strip()
                    src_batch.append(source_text)
                    if len(src_batch) == batch_size:
                        print(f"\t\tClassifying new batch")
                        classified = classify_batch(src_batch, en=True if src_lang == 'en' else False)
                        for i in range(batch_size):
                            x = classified[i]
                            interpret_classification(x, src_lang, tgt_lang, category)

                        src_batch = []

                if len (src_batch) > 0:
                    classified = classify_batch(src_batch, en=True if src_lang == 'en' else False)
                    for i in range(len(classified)):
                        x = classified[i]
                        interpret_classification(x, src_lang, tgt_lang, category)

with open("cross_classification50k_with_neutral.txt", "w") as o:
    o.write(json.dumps(result, sort_keys=False, indent=4))

