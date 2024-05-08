# FAME-MT dataset
Dataset of translations between 15 source and 8 target languages annotated with formality information.

Source languages supported: Czech, Danish, Dutch, English, German, French, Italian, Norwegian, Polish, Portuguese, Russian, Slovak, Spanish, Swedish, Ukrainian
Target languages supported: Dutch, English, German, French, Italian, Polish, Portuguese, Spanish

For each combination of the source and target language, we provide samples of 50,000 examples, for which target sentence is considered formal and 50,000 translation examples, for which target sentence is considered informal.

The dataset is a subset of corpora downloaded using MTData tool (https://github.com/thammegowda/mtdata)

# Hugginface models
Along with the datasets, we published models available online:

- Marian-based formality-aware translator from English to Polish: https://huggingface.co/laniqo/marian_formality_fine_tuned_en_pl/
- Marian-based formality-aware translator from English to German: https://huggingface.co/laniqo/marian_formality_fine_tuned_en_de/
- Multilingual formality classifier, detecting whether a given sentence is formal or not, supporting Dutch, German, French, Italian, Polish, Portuguese, Spanish: https://huggingface.co/laniqo/fame_mt_mdeberta_formality_classifer


