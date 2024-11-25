# FAME-MT dataset
Dataset of translations between 15 source and 8 target languages annotated with formality information.

Source languages supported: `Czech, Danish, Dutch, English, German, French, Italian, Norwegian, Polish, Portuguese, Russian, Slovak, Spanish, Swedish, Ukrainian`

Target languages supported: `Dutch, English, German, French, Italian, Polish, Portuguese, Spanish`

For each combination of the source and target language, we provide samples of 50,000 examples, for which target sentence is considered formal and 50,000 translation examples, for which target sentence is considered informal.

The dataset is a subset of corpora downloaded using MTData tool (https://github.com/thammegowda/mtdata)

# Hugginface models
Along with the datasets, we published models available online:

- Marian-based formality-aware translator from English to Polish: https://huggingface.co/laniqo/marian_formality_fine_tuned_en_pl/
- Marian-based formality-aware translator from English to German: https://huggingface.co/laniqo/marian_formality_fine_tuned_en_de/
- Multilingual formality classifier, detecting whether a given sentence is formal or not, supporting Dutch, German, French, Italian, Polish, Portuguese, Spanish: https://huggingface.co/laniqo/fame_mt_mdeberta_formality_classifer

# Citing this work

Please use the following BibTeX entry for references:

```
@inproceedings{DBLP:conf/eamt/WisniewskiRN24,
  author       = {Dawid Wisniewski and
                  Zofia Rostek and
                  Artur Nowakowski},
  editor       = {Carolina Scarton and
                  Charlotte Prescott and
                  Chris Bayliss and
                  Chris Oakley and
                  Joanna Wright and
                  Stuart Wrigley and
                  Xingyi Song and
                  Edward Gow{-}Smith and
                  Rachel Bawden and
                  V{\'{\i}}ctor M. S{\'{a}}nchez{-}Cartagena and
                  Patrick Cadwell and
                  Ekaterina Lapshinova{-}Koltunski and
                  Vera Cabarr{\~{a}}o and
                  Konstantinos Chatzitheodorou and
                  Mary Nurminen and
                  Diptesh Kanojia and
                  Helena Moniz},
  title        = {{FAME-MT} Dataset: Formality Awareness Made Easy for Machine Translation
                  Purposes},
  booktitle    = {Proceedings of the 25th Annual Conference of the European Association
                  for Machine Translation (Volume 1), {EAMT} 2024, Sheffield, UK, June
                  24-27, 2024},
  pages        = {164--180},
  publisher    = {European Association for Machine Translation {(EAMT)}},
  year         = {2024},
  url          = {https://aclanthology.org/2024.eamt-1.16},
  timestamp    = {Tue, 01 Oct 2024 22:18:16 +0200},
  biburl       = {https://dblp.org/rec/conf/eamt/WisniewskiRN24.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```


