# Amazon Prime Video Conversion Prediction

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CooperNing/Amazon-Prime-Video-Conversion-Prediction/blob/main/Amazon_Prime_Video_Conversion_Prediction.ipynb)

A supervised learning project that predicts how many daily conversions (`cvt_per_day`) an Amazon Prime Video title generates from its catalog metadata, so that merchandising and on-site placement decisions can be guided by the attributes that actually drive viewing. The full workflow lives in the notebook in this repository.

## Data

A Prime Video catalog extract (`TVdata.txt`) with 4,226 titles and 16 columns: on-site placement (weighted categorical and horizontal position), studio, release year, genres, runtime, MPAA rating, awards, star category, budget, box office, IMDb votes and rating, and Metacritic score. Zeros in several numeric fields are physically impossible and are therefore treated as missing (up to 76% of rows for box office) and imputed with the column mean.

## Method

Studio, MPAA rating, awards and individually split genres are one-hot encoded, rare genres are grouped into a single bucket, and release year is binned into ten quantile ranges. After mean imputation and standard scaling, the data is split 85/15 into train and test. Three regressors are compared: Lasso and Ridge, each with alpha tuned on a held-out validation split, and a Random Forest tuned by 5-fold grid search over `max_depth` and `n_estimators`.

## Key findings

The Random Forest clearly outperforms both linear baselines on the held-out test set, reaching an R-squared of 0.514 with an MSE of 1.29e8 (RMSE 11,357), compared with 0.114 for Ridge (MSE 2.35e8) and 0.099 for Lasso (MSE 2.39e8). The large gap indicates that conversion behaviour is strongly non-linear in these features, and the final section of the notebook ranks the twenty most important predictors to show which catalog and placement attributes carry that signal.

## Tech stack

Python, pandas, NumPy, scikit-learn, seaborn, Matplotlib and Google Colab.

## How to run

The dataset (`TVdata.txt`) is committed to this repository, so the notebook runs end to end without any manual file upload. Open the notebook with the Colab badge above and choose Runtime > Run all; the data is read straight from this repo:

```python
DATA_URL = "https://raw.githubusercontent.com/CooperNing/Amazon-Prime-Video-Conversion-Prediction/main/TVdata.txt"
TV = pd.read_table(DATA_URL, header=0, sep=',', lineterminator='\n')
```

To run it locally instead, clone the repo and install the dependencies:

```bash
git clone https://github.com/CooperNing/Amazon-Prime-Video-Conversion-Prediction.git
cd Amazon-Prime-Video-Conversion-Prediction
pip install pandas numpy scikit-learn seaborn matplotlib jupyter
jupyter notebook Amazon_Prime_Video_Conversion_Prediction.ipynb
```
