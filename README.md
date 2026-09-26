# Amazon Prime Video Conversion Prediction

A supervised learning project that predicts how many daily conversions (`cvt_per_day`) an Amazon Prime Video title generates from its catalog metadata, so that merchandising and on-site placement decisions can be guided by the attributes that actually drive viewing. The full workflow lives in the notebook in this repository.

## Data

A Prime Video catalog extract (`TVdata.txt`) with 4,226 titles and 16 columns: on-site placement (weighted categorical and horizontal position), studio, release year, genres, runtime, MPAA rating, awards, star category, budget, box office, IMDb votes and rating, and Metacritic score. Zeros in several numeric fields are physically impossible and are therefore treated as missing (up to 76% of rows for box office) and imputed with the column mean.

## Method

Studio, MPAA rating, awards and individually split genres are one-hot encoded, rare genres are grouped into a single bucket, and release year is binned into ten quantile ranges. After mean imputation and standard scaling, the data is split 85/15 into train and test. Three regressors are compared: Lasso and Ridge, each with alpha tuned on a held-out validation split, and a Random Forest tuned by 5-fold grid search over `max_depth` and `n_estimators`.

## Key findings

The Random Forest clearly outperforms both linear baselines on the held-out test set, reaching an R-squared of 0.514 with an MSE of 1.29e8 (RMSE 11,357), compared with 0.114 for Ridge (MSE 2.35e8) and 0.099 for Lasso (MSE 2.39e8). The large gap indicates that conversion behaviour is strongly non-linear in these features, and the final section of the notebook ranks the twenty most important predictors to show which catalog and placement attributes carry that signal.

## Tech stack

Python, pandas, NumPy, scikit-learn, seaborn, Matplotlib and Google Colab.
