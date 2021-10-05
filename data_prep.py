import pandas as pd
import numpy as np
from math import sqrt
from scipy.stats import chi2_contingency

def _get_dataset():
    # Read in the data
    return pd.read_csv('./data/data2021.student.csv')

def _get_high_corr_cols(df:pd.DataFrame, corr_threshold:float):
    """
    Retrieved from https://chrisalbon.com/code/machine_learning/feature_selection/drop_highly_correlated_features/
    """
    # Create correlation matrix
    corr_matrix = df.corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Find index of feature columns with correlation greater than correlation threshold
    return [column for column in upper.columns if any(upper[column] > corr_threshold)]

def _prep_dataset(  df:pd.DataFrame, 
                    bins=None, set_to_cat_threshold=10, bin_type='width',
                    variance_threshold=0.005, cat_monotony_threshold=0.95, num_corr_threshold=0.95, cat_association_threshold=0.95,
                    normalize=True,
                    row_missing_val_threshold=0.5,
                    col_missing_val_threshold=0.8,
                    verbose=False):

    # Add/remove verbose prints
    if verbose:
        printv = lambda msg: print("[v] " + msg)
    else:
        printv = lambda msg: None

    ## Remove the 'ID' column
    df = df.drop(['ID'], axis=1)

    ## Convert attributes with less than threshold unique values to categorical
    lt_cat_threshold_cols = [col for col in df.columns if df[col].nunique() <= set_to_cat_threshold]
    printv(f"Converting columns with >={set_to_cat_threshold} unique values to categorical: {lt_cat_threshold_cols}")
    df[lt_cat_threshold_cols] = df[lt_cat_threshold_cols].astype('category')

    # Get number of missing values in all columns
    cols_missing_values = df.drop('Class', axis=1).isnull().mean()

    ## Drop cols with missing values (>=col_missing_val_threshold) # TODO maybe drop for numeric at like 75% and for categorical at like 90%?
    cols_missing_values_over_thresh = cols_missing_values[cols_missing_values >= col_missing_val_threshold]
    printv(f"Removing columns with >={col_missing_val_threshold} missing values: {list(cols_missing_values_over_thresh.index)}")
    df = df.drop(cols_missing_values_over_thresh.index, axis=1)

    # Account for missing values in categorical columns (not including 'Class')
    cat_cols = df.select_dtypes(include=['category']).columns.drop('Class')
    for col in cat_cols:
        if cols_missing_values[col] > 0:
            ## In cat. columns with <=0.05 missing values, replace missing values with the mode
            df[col] = df[col].fillna(value=df[col].mode()[0])

    ## In numeric columns, replace all remaining missing values with the mean
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(value=df[col].mean())
    
    if normalize:
        ## Normalise (min-max) numeric columns
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_cols:
            df[col] = (df[col]-df[col].min())/(df[col].max()-df[col].min())
        printv(f"Normalising (min-max) numeric columns")

        ## Drop low-variance numeric columns
        numeric_col_variance = df[numeric_cols].var()
        variance_threshold_fail_cols = numeric_col_variance[numeric_col_variance < variance_threshold]
        printv(f"Dropping low-variance numeric columns: {list(variance_threshold_fail_cols.index)}")
        df = df.drop(list(variance_threshold_fail_cols.index), axis=1)
    
    ## Remove (100% identical) duplicate columns
    duplicate_cols = df.transpose().duplicated()
    printv(f"Removing duplicate columns '{list(df.columns[duplicate_cols])}'")
    df = df.drop(list(df.columns[duplicate_cols]), axis=1)

    # Drop second of high-correlation numeric columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    high_cor_cols = _get_high_corr_cols(df[numeric_cols], num_corr_threshold)
    printv(f"Dropping 2nd of high-corellation numeric columns: {high_cor_cols}")
    df = df.drop(high_cor_cols, axis=1)

    ## Convert numeric attributes to categorical through binning
    if bins != None:
        # TODO depth-wise binning
        assert bin_type in ['width', 'depth']

        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

        printv(f"Converting numeric attributes to categorical through {bin_type}-wise binning ({bins} bins): {list(numeric_cols)}")

        for col in numeric_cols:
            df[col] = pd.cut(df[col], bins=bins, labels=[f'bin-{ii}' for ii in range(10)])

    # Drop monotonous categorical cols
    cat_cols = df.select_dtypes(include=['category']).drop(['Class'], axis=1).columns
    high_monotony_cols = []
    for col in cat_cols:
        if df[col].value_counts(normalize=True).max() >= cat_monotony_threshold:
            high_monotony_cols.append(col)

    printv(f"Dropping highly monotonous (>={cat_monotony_threshold}) categorical cols: {high_monotony_cols}")
    df = df.drop(high_monotony_cols, axis=1)

    ## Dropping categorical columns with extremely high Cramer's V association
    cat_cols = df.select_dtypes(include=['category']).drop(['Class'], axis=1).columns
    cat_high_assoc_cols = []
    for i1, col1 in enumerate(cat_cols):
        for i2, col2 in enumerate(cat_cols[:(i1+1)]):
            contingency_table = pd.crosstab(df[col1], df[col2])
            k = contingency_table.shape[0]
            r = contingency_table.shape[1]
            try:
                chi2, p, dof, ex = chi2_contingency(contingency_table, correction=True)
                cramers_v = sqrt(chi2 / (len(df[col1]) * min(k-1, r-1))) # https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V
            except ZeroDivisionError:
                cramers_v = -1
            except ValueError:
                cramers_v = -1

            if cramers_v >= cat_association_threshold and col1 != col2:
                cat_high_assoc_cols.append(col2)
    printv(f"Dropping high Cramer's V association categorical columns: {cat_high_assoc_cols}")

    ## Split train & test set
    df_train:pd.DataFrame = df.iloc[:1000]
    df_test:pd.DataFrame = df.iloc[1000:]

    ## Remove duplicate rows in training set
    printv(f"Dropping {len(df_train.duplicated())} duplicate rows")
    df_train = df_train.drop_duplicates()
    
    # Remove rows in training set with at least threshold missing values
    na_thresh = int(len(df_train.columns) * (1-row_missing_val_threshold))
    df_train_dropna = df_train.dropna(thresh=na_thresh)
    printv(f"Dropped {len(df_train) - len(df_train_dropna)} columns that contained >{row_missing_val_threshold} missing values")
    df_train = df_train_dropna

    printv(f"Final training set size: {len(df_train)}")
    printv(f"Final test set size: {len(df_test)}")
    printv(f"Final columns ({len(df_train.columns)}): {list(df_train.columns)}")

    return df_train, df_test

def get_prepped_dataset(bins=None, cat_binning_threshold=30, bin_type='width',
                    normalize=False,
                    variance_threshold=0.005, cat_monotony_threshold=0.95, num_corr_threshold=0.95, cat_association_threshold=0.95, row_missing_val_threshold=0.5, col_missing_val_threshold=0.8, 
                    verbose=False):
    df = _get_dataset()

    return _prep_dataset(df, bins=bins, set_to_cat_threshold=cat_binning_threshold, bin_type=bin_type,
                    variance_threshold=variance_threshold, cat_monotony_threshold=cat_monotony_threshold, num_corr_threshold=num_corr_threshold, cat_association_threshold=cat_association_threshold,
                    row_missing_val_threshold=row_missing_val_threshold, col_missing_val_threshold=col_missing_val_threshold, normalize=normalize,
                    verbose=verbose)

def main():
    df_in = _get_dataset()

    df_train, df_test = _prep_dataset(df_in, bins=10)

    print("Done! Saving...")

    ## Final saving
    df_in.to_csv('df_in_new.csv', index=False)
    df_train.to_csv('df_out_new.csv', index=False)
    df_test.to_csv('df_test_out_new.csv', index=False)

if __name__ == '__main__':
    main()