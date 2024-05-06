from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import sys
import random
import os

lang_path = sys.argv[1]

def load_targets(path, formal=False):
    # Load all files for a given target language and sample 1/14 of each file
    data = []

    for file in os.listdir(path):
        if '.informal.tsv' in file and formal:
            continue
        elif '.formal.tsv' in file and not formal:
            continue

        with open(os.path.join(path, file), 'r') as f:
            for line in f:
                if random.random() <= 1.0 / 14.0:
                    line = line.strip().split("\t")[1]
                    data.append(line)

    label = 1 if formal else 0
    labels = [label] * len(data)
    return data, labels


pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=200000)),
    ('scaler', StandardScaler(with_mean = False)),
    ('clf', LogisticRegression(fit_intercept=False)),
])


all_data = []
all_labels = []

for formal in [True, False]:
    formal_lbl = 1 if formal else 0
    data, labels = load_targets(lang_path, formal)
    all_data += data
    all_labels += labels


c = list(zip(all_data, all_labels))
random.shuffle(c)
all_data, all_labels = zip(*c)

pipeline.fit(all_data, all_labels) 

def language_indicators(feature_names, feature_importances):
    for i, language in enumerate(feature_importances):
        scored_features = list(zip(feature_names, language))
        scored_features = sorted(scored_features, key=lambda x: x[1], reverse=True)
        print(len(scored_features))
        print("\n\nTOP 50 tokens related to FORMAL class")
        for feature, score in scored_features[:100]:
            print("\t'{feature}': {score}".format(feature=feature, score=score))

        print("\n\nTOP 50 tokens related to INFORMAL class")
        for feature, score in scored_features[-100:][::-1]:
            print("\t'{feature}': {score}".format(feature=feature, score=score))


language_indicators(
    pipeline.named_steps['tfidf'].get_feature_names_out(), 
    pipeline.named_steps['clf'].coef_
)