# SteamPulse Development Challenges

This document records the main problems encountered while developing SteamPulse, the approaches used to address them, and the limitations that remain.

The purpose is to document the actual development process, including unsuccessful experiments and decisions that affected the final project.

## 1. Steam API Collection

### Problem

Collecting a large catalogue of Steam applications required many API requests. Initial requests returned HTTP 404 and 403 errors before a working request pattern was established. Collection also encountered HTTP 429 responses caused by rate limiting.

Some application detail requests failed, making it difficult to collect every field for every application in a single uninterrupted run.

### Approach

The collection process was developed around separate catalogue and application detail requests. Python scripts were used to collect records, track progress, record failed requests, and support further collection attempts.

### Remaining limitation

The presence of a successful collection run does not establish that every Steam application was collected or that every application's metadata is complete.

The project should be described as a collected Steam catalogue, not a guaranteed complete record of every game available on Steam.

## 2. CSV Corruption and File Handling

### Problem

The project generated large CSV files during collection and processing. Some file outputs became truncated, overwritten, malformed, or unexpectedly small. At different stages, the reported file size did not always match the file that was expected to contain the full dataset.

These issues made it important to verify file paths, file sizes, row counts, and contents rather than relying only on script progress messages.

### Approach

Python inspection scripts were used to check files and review dataset contents. DuckDB was also used to read problematic CSV data with error tolerant settings so that usable records could be recovered for further processing.

### Remaining limitation

Error tolerant CSV reading can skip or misinterpret malformed rows. Recovered data cannot automatically be assumed to be complete or identical to the original source.

Raw files have been retained locally. Large source data and intermediate outputs are excluded from the planned public repository.

## 3. Large Dataset and Repository Size

### Problem

The local project grew to several gigabytes because it contains raw datasets, cleaned datasets, intermediate processing outputs, feature tables, experiments, and model artifacts.

Several individual files are larger than 1 GB.

Uploading the entire local directory to GitHub would create an unnecessarily large repository and could exceed ordinary GitHub file size limits.

### Approach

The publication plan separates files needed by the Streamlit application from files used during data collection, cleaning, modelling, and experimentation.

The application requires a scored catalogue CSV and two JSON reports. The selected packaged model can be retained as a separate reproducibility artifact.

### Remaining limitation

The public repository will not contain every raw dataset or intermediate file needed to repeat the entire collection and modelling pipeline from the beginning. The documentation must make this distinction clear.

## 4. Cleaning and Dataset Population

### Problem

The dataset contained records that required cleaning and normalisation. The number of rows differed between the original catalogue, the cleaned collection, and the modelling population.

Recorded dataset sizes were:

| Stage | Records |
|---|---:|
| Initial Steam application catalogue | 186,504 |
| Cleaned game dataset | 185,462 |
| Final modelling and scoring population | 139,860 |
| Labelled modelling population | 22,625 |

These are counts from different processing stages and should not be treated as interchangeable.

### Approach

Separate files were retained for raw collection, cleaned records, processed modelling data, and the scored catalogue.

This made it possible to distinguish the broad collection from the smaller population used for the final modelling workflow.

### Remaining limitation

Differences in row counts reflect the requirements of the respective processing stages and the availability of usable data. They do not prove that every record excluded from the modelling population was invalid.

The final catalogue should not be described as a perfect or exhaustive record of Steam games.

## 5. Price Units and Currency Conversion

### Problem

Price values collected from Steam could be represented as integer amounts in minor currency units. Treating these values directly as whole currency units could produce implausibly high prices.

The dataset also contained different currencies, free games, and records with missing prices.

### Approach

A price correction step divided the affected integer price values by 100 before applying currency conversion. The correction was recorded in a price unit audit.

### Remaining limitation

Currency conversions used available exchange rates rather than historical rates for each game's release date.

The corrected values should not be interpreted as exact historical launch prices. Price data also remains incomplete for some games.

## 6. Missing Recommendation Outcomes

### Problem

Observed recommendation information was not available for every game. Treating missing values as zero would incorrectly classify unknown outcomes as negative examples.

This matters because the model's target depends on observed recommendation counts.

### Approach

The labelled population was separated from the larger catalogue. The positive class was defined using a threshold of 6,785 recommendations derived from the training distribution.

Games without observed outcomes were not automatically assigned negative target labels for model training.

### Remaining limitation

Only 22,625 of the 139,860 games in the final scoring population have observed target labels.

The model still scores games without observed outcomes, but their actual target outcomes cannot be evaluated with the available data.

## 7. Model Selection and Convergence

### Problem

The target is imbalanced, with high recommendation engagement representing a minority class. Accuracy alone would not adequately describe model performance.

Several model types, feature combinations, and decision thresholds were evaluated. Some LinearSVC experiments produced convergence warnings even after the iteration limit was increased.

### Approach

Candidates were compared using average precision, ROC AUC, precision, recall, and F1. Logistic Regression was selected for the release candidate based on its validation results and its suitability for packaging with the structured and text processing steps.

The final candidate uses regularization parameter `C = 0.5`.

### Final recorded validation results

| Metric | Result |
|---|---:|
| Average precision | 0.4467 |
| ROC AUC | 0.8451 |
| Precision | 35.03% |
| Recall | 60.24% |
| F1 score | 0.4430 |

The original average precision target was 0.45. The selected model did not quite reach it.

### Remaining limitation

The metrics represent the recorded validation results and do not guarantee similar performance across every catalogue population.

The selected threshold was chosen using validation data. The model's decision scores are not calibrated probabilities.

## 8. Evaluation and Metadata Timing

### Problem

The project used a temporal split to examine model performance on later release years. However, some input metadata may have been recorded or updated after a game launched.

Multiple experiments also reused the same validation data. Earlier experiments evaluated the test set repeatedly.

### Approach

The application reports validation metrics and explains the temporal partitions and evaluation limitations.

The project does not present the model as a fully validated pre release forecasting system.

### Remaining limitation

The test set is not an untouched final holdout. Its results should not be described as an independent final estimate of generalisation.

Post release metadata and unequal recommendation accumulation periods also limit how the results should be interpreted.

## 9. Different Prediction Rates Across Populations

### Problem

The full catalogue has a predicted HIT rate of 3.67%, while the labelled validation subset has a predicted HIT rate of approximately 13.06%.

At first, these figures appeared inconsistent.

### Analysis

The populations have different sizes and different availability of observed target outcomes.

| Population | Games | Predicted HIT rate |
|---|---:|---:|
| Entire scored catalogue | 139,860 | 3.67% |
| Games with observed outcomes | 22,625 | 15.50% |
| Games without observed outcomes | 117,235 | 1.38% |
| Complete 2023 and 2024 release year population | 30,335 | 2.96% |
| Labelled validation subset | 4,372 | 13.06% |

Within the labelled validation subset, 571 of 4,372 games were predicted as HIT. Across the complete 2023 and 2024 release year population, 899 of 30,335 games were predicted as HIT.

### Interpretation

The percentages use different denominators and should not be directly compared as though they describe the same population.

The results show that the labelled and unlabelled groups have different prediction rate distributions. This difference alone does not establish why the distributions differ or whether predictions for unlabelled games are correct.

The decision threshold was not changed simply to make the catalogue prediction rate resemble the validation subset rate.

## 10. Small Samples in Release Year Charts

### Problem

Some early release years contained only a handful of games. A rate of 100% based on one or two records can be easy to misinterpret when shown alongside years containing thousands of games.

### Approach

The overview chart was changed to display only release years with at least 200 catalogue games. The chart includes game counts in its hover information and explains that it displays model classifications rather than actual historical success rates.

### Remaining limitation

The chart is descriptive of model outputs. It is not a measurement of the historical success rate of games released in each year.

Games excluded from the chart remain part of the dataset and other application pages.

## 11. Streamlit Application Debugging

### Problem

During dashboard development, running the application directly with Python produced Streamlit context warnings. An earlier version also displayed some HTML as literal text, and a previous page implementation encountered a variable initialization error.

### Approach

The application is launched through Streamlit:

```bash
python -m streamlit run App/app.py
```

The HTML rendering was adjusted, and the application initialization was corrected. The dashboard was then checked locally across its main pages.

### Deployment check

The project uses a directory named `Reports`. Earlier code referenced the lowercase path `reports`. Windows can tolerate this difference, while Linux deployment environments are case sensitive.

Before deployment, verify that `App/app.py` uses the exact directory capitalization for both report paths.

## 12. Model Packaging and Catalogue Scoring

### Problem

The final application needed to display consistent predictions without retraining the model whenever a user opened the dashboard.

The feature processing pipeline also needed to be preserved alongside the classifier for reproducibility.

### Approach

The selected Logistic Regression model was packaged with its preprocessing components, encoders, vectorizers, selected threshold, and recorded metadata.

A verification step reconstructed the validation features and reproduced the saved validation metrics. The full scoring process produced a separate catalogue predictions CSV for the Streamlit application.

### Remaining limitation

The dashboard displays precomputed predictions rather than performing model inference for each visitor. Reproducing the full training process requires the relevant source data and intermediate processing artifacts, most of which are excluded from the public application repository.

## Summary

The main engineering challenges in SteamPulse involved API rate limits, large CSV files, inconsistent price units, missing target labels, class imbalance, model convergence, validation design, and application deployment compatibility.

The project addressed many of these issues through data checks, price corrections, explicit target construction, candidate comparisons, model packaging, and reproducibility verification.

Important limitations remain. The target is a proxy for recommendation engagement, much of the catalogue lacks observed labels, some metadata may be post release, and the test set was exposed during earlier experiments.

These limitations are part of the project record and should remain visible when interpreting the results.