mimic_chemo datasets

two cohorts: mimic_cohort_aplasia_45_days, mimic_cohort_NF_30_days (prediction of aplasia and neutropenic fever side effects of chemotherapy, respectively) extracted within our previous work: https://www.medrxiv.org/content/10.64898/2025.12.12.25342142v1

folders:

cohorts - contain the target (column label), subject and admission ids, and demographics (age and gender)

folds - train, val, test splits for 5-fold cross validation used in our investigation. For simple training (no hyper-parameter tuning) concatenate train+val for training, test for testing.

raw_files - all laboratory measurements taken in the 14 days before discharge

preprocessed_files:
  - aggregate - average value for top 100 most frequent labs <-- start with this configuration
  - concatenated - average value per day for top 100 most frequent labs

these files still contain some missing values but can be zero-imputed if needed (some ML models )
