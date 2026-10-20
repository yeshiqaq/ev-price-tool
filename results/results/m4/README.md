# M4: Complaint text analytics (results)

Notebook: `notebooks/03_text_analytics.ipynb` (M4). Input: 9,440 NHTSA EV owner complaints received before 1 May 2023, across 38 model lines, matched to 97.1% of listings.

## Pipeline
Clean text (staff notes removed from 6.5% of complaints) → bag-of-words and TF-IDF → n-grams → sentiment (VADER on all complaints; DistilBERT check on 300) → keyword theme tags → model-year reliability features (`models/dim_reliability.csv`, 159 rows).

## Theme shares (share of complaints; one complaint can carry several themes)
| Theme | Share |
|---|---|
| Braking and driver assist | 41.4% |
| Software and screens | 25.7% |
| Battery and charging | 15.6% |
| Mention a recall | 14.3% |
| Report a crash or fire | 10.5% |

## Theme tags vs NHTSA component categories
| Theme tag | Precision | Recall |
|---|---|---|
| Braking / driver assist | 0.822 | 0.844 |
| Battery / charging | 0.731 | 0.371 |
| Software / screens | 0.512 | 0.440 |

## Sentiment check
DistilBERT rated 99% of 300 sampled complaints negative; VADER rated 76% (mean VADER compound score −0.42). Sentiment is therefore used to compare how negative complaints are, not whether they are negative.

## Reliability table (159 model-years)
82 use their own figures, 53 fall back to whole-model-line figures (few complaints), 14 have no complaints that year and use the model line, and 10 model lines have no complaints at all.

## Files
| File | Content |
|---|---|
| `fig_top_tfidf_words.png` | Most distinctive TF-IDF words |
| `fig_top_ngrams.png` | Most common complaint phrases (n-grams) |
| `fig_sentiment_hist.png` | Distribution of VADER sentiment |
| `fig_theme_shares.png` | Share of complaints by theme |
| `fig_themes_by_model_line.png` | Theme shares by model line |
| `fig_sentiment_by_model_line.png` | Average sentiment by model line |
| `tag_accuracy_check.csv` | Theme tags vs NHTSA categories: precision and recall |
| `text_key_numbers.csv` | Headline numbers from the analysis |

Not included: `complaints_tagged.csv` (full complaint text; regenerate with the notebook) and `dim_reliability.csv` (stored in `models/`).
