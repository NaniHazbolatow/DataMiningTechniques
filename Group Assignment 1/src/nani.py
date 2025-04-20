import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss  # only if you still want to import it elsewhere

def sensitivity_matcher(miss: pd.Series,
                        covariates: pd.DataFrame,
                        min_leaf: int = 25):
    """
    Impute missing values by matching on Random Forest proximity,
    and report only the AUC of the missingness model.
    """
    # 1. Missingness indicator
    missing_indicator = miss.isnull().astype(int).values

    # 2. Fit RF classifier to predict missingness
    rf = RandomForestClassifier(
        n_estimators=500,
        bootstrap=False,
        min_samples_leaf=min_leaf,
        max_features="sqrt",
        random_state=42
    )
    rf.fit(covariates, missing_indicator)

    # 3. Propensity scores = P(missing)
    propensity_scores = rf.predict_proba(covariates)[:, 1]

    # 4. Leaf assignments
    leaf_indices = rf.apply(covariates)   # shape = (n_samples, n_trees)
    n_samples = leaf_indices.shape[0]

    # Build similarity matrix
    similarity_matrix = np.zeros((n_samples, n_samples), dtype=float)
    for i in range(n_samples):
        for j in range(i, n_samples):
            sim = np.mean(leaf_indices[i] == leaf_indices[j])
            similarity_matrix[i, j] = sim
            similarity_matrix[j, i] = sim

    # Convert to distances
    distance_matrix = 1.0 - similarity_matrix

    # ** Only AUC **
    auc_score = roc_auc_score(missing_indicator, propensity_scores)
    print(f"AUC for missingness model: {auc_score:.4f}")

    # 5. Matching & imputation
    observed_mask   = miss.notnull().values
    observed_idx    = np.where(observed_mask)[0]
    missing_idx     = np.where(~observed_mask)[0]

    imputed = miss.copy()
    matches = []

    for i in missing_idx:
        # find nearest donor
        dists = distance_matrix[i, observed_idx]
        donor = observed_idx[np.argmin(dists)]
        sim   = 1.0 - distance_matrix[i, donor]

        # impute
        imputed.iloc[i] = miss.iloc[donor]
        matches.append((i, donor, sim))

    matching_table = pd.DataFrame(
        matches,
        columns=["missing_index", "donor_index", "similarity"]
    )

    return imputed.values, matching_table

def convert_to_binary(df, column):
    unique_values = df[column].dropna().unique()
    if len(unique_values) == 2:
        mapping = {unique_values[0]: 0, unique_values[1]: 1}
        return df[column].map(mapping)
    else:
        raise ValueError("Column is not binary")