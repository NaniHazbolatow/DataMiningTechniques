import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss

def sensitivity_matcher(miss, covariates):
    """
    When a question is sensitive, the missing mechanism is often in-between MNAR and MAR.
    This function tries to impute the missing values by using a random forest to predict the
    missingness, and then uses the random forest's leaf structure to find similar observations. 
    We use those similar observations to impute the missing values (i.e., we find a donor).
   The method works as follows:
      1. Create a missing indicator from the outcome variable.
      2. Fit a RandomForestClassifier (with full data, no bootstrap, 
         min_samples_leaf=25, and sqrt(p) features) to predict the missingness.
         (See Propensity Score and Proximity Matching Using Random Forest (doi: 10.1016/j.cct.2015.12.012))
      3. Compute the propensity scores as the predicted probability of missingness.
      4. Extract leaf assignments from the forest, then compute a pairwise 
         similarity (fraction of trees that assign two observations to the same leaf)
         and convert it to a distance measure (1 - similarity).
      5. For each observation with a missing outcome, match it to the observed 
         donor (with non-missing outcome) that has the smallest distance.
      6. Impute the missing outcome with the donor’s observed outcome.
      7. Compute a pseudo R² from the model’s log loss. (Not neccessarily useful, but
         it is a good sanity check to see if the missing mechanism is more MNAR than MAR.)
    
    Parameters:
      miss: pd.Series
         Outcome variable with missing values.
      covariates: pd.DataFrame
         Covariate matrix.

    Returns:
      output_dataset: pd.DataFrame
         DataFrame that includes the original covariates, the original outcome,
         the imputed outcome, and the propensity (missing) score.
      matching_table: pd.DataFrame
         Table with each missing case's index, donor index, and similarity score.
    """
    # 1. Create the missing indicator (1 if missing, 0 if observed)
    missing_indicator = miss.isnull().astype(int).values

    # 2. Fit the random forest to predict the missingness using all covariates.
    rf = RandomForestClassifier(
        n_estimators=500,
        bootstrap=False,          # use all the data in each tree
        min_samples_leaf=25,      # set minimum leaf size
        max_features="sqrt",      # number of features to try at each split
        random_state=42
    )
    rf.fit(covariates, missing_indicator)

    # 3. Compute the "propensity" score, here the probability that an observation's
    # outcome is missing.
    propensity_scores = rf.predict_proba(covariates)[:, 1]

    # 4. Get the leaf indices for each observation in every tree
    leaf_indices = rf.apply(covariates)  # shape: (n_samples, n_trees)
    n_samples, _ = leaf_indices.shape

    # Build the similarity matrix: similarity = fraction of trees where two observations share the same leaf
    similarity_matrix = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(i, n_samples):
            sim = np.mean(leaf_indices[i] == leaf_indices[j])
            similarity_matrix[i, j] = sim
            similarity_matrix[j, i] = sim  # the matrix is symmetric

    # Convert similarity into a distance metric
    distance_matrix = 1 - similarity_matrix

    # 5. Compute a pseudo R² for the missing indicator model.
    # Using log loss for the model vs. a null model that predicts the baseline missing probability.
    logloss_model = log_loss(missing_indicator, propensity_scores)
    null_prob = np.mean(missing_indicator)
    logloss_null = log_loss(missing_indicator, np.full_like(missing_indicator, null_prob))
    pseudo_R2 = 1 - (logloss_model / logloss_null)
    print("Model pseudo R²:", pseudo_R2)

    # 6. Matching: For each missing observation, find the nearest donor among those with an observed outcome.
    observed_mask = miss.notnull()
    observed_indices = np.where(observed_mask)[0]
    missing_indices = np.where(~observed_mask)[0]

    # Create a copy of miss to hold imputed values.
    imputed_values = miss.copy()

    # Create a table to record the matching: missing index, donor index, similarity (1 - distance)
    matching_rows = []
    for i in missing_indices:
        # Calculate the distance for missing observation i to all observed (non-missing) observations
        distances_to_donors = distance_matrix[i, observed_indices]
        # Identify the index in observed_indices with the minimal distance
        donor_idx = observed_indices[np.argmin(distances_to_donors)]
        sim = 1 - distance_matrix[i, donor_idx]  # similarity score
        # Impute the missing outcome with the donor's observed outcome
        imputed_values.iloc[i] = miss.iloc[donor_idx]
        matching_rows.append((i, donor_idx, sim))
    
    matching_table = pd.DataFrame(matching_rows, columns=['missing_index', 'donor_index', 'similarity'])

    return imputed_values.values, matching_table

