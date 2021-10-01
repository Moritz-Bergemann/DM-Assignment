import pandas as pd
import numpy as np

def _get_dataset():
    # Read in the data
    return pd.read_csv('./data/data2021.student.csv')

def _get_high_corr_cols(df:pd.DataFrame):
    """
    Retrieved from https://chrisalbon.com/code/machine_learning/feature_selection/drop_highly_correlated_features/
    """
    # Create correlation matrix
    corr_matrix = df.corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    # Find index of feature columns with correlation greater than 0.95
    return [column for column in upper.columns if any(upper[column] > 0.95)]

def _prep_dataset(df:pd.DataFrame, bins=None, variance_threshold=0.005, cat_monotony_threshold=0.95):
    ## Basic operations
    # Remove the 'ID' index (since it is useless and interferes with everything else)
    df = df.drop(['ID'], axis=1)

    # Convert attributes with less than 30 unique values to categorical # TODO JUSTIFY 30
    lt_30_unique_cols = [col for col in df.columns if df[col].nunique() <= 30]
    df[lt_30_unique_cols] = df[lt_30_unique_cols].astype('category')

    # # Convert numeric attributes to categorical through binning # TODO do binning as h-param later
    # numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    # for col in numeric_cols:
    #     # pd.cut(df[col], bins=10)
    #     df[col] = pd.cut(df[col], bins=10, labels=[f'bin-{ii}' for ii in range(10)])

    # Get number of missing values in all columns
    cols_missing_values = df.isnull().mean()
    
    # TODO maybe drop for numeric at like 75% and for categorical at like 90%?

    # Drop cols with missing values (>0.8)
    cols_missing_values_80 = cols_missing_values[cols_missing_values > 0.8]
    print(f"Removing columns with more than 80% missing values: {list(cols_missing_values_80.index)}")
    df = df.drop(cols_missing_values_80.index, axis=1)

    # In numeric columns, replace all missing values with the mean
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(value=df[col].mean())

    ## Account for missing values in categorical columns (not including 'Class')
    cat_cols = df.select_dtypes(include=['category']).columns.drop('Class')
    for col in cat_cols:
        # In cat. columns with <0.05 missing values, replace missing values with the mean
        if cols_missing_values[col] <= 0.05:
            df[col] = df[col].fillna(value=df[col].mode())
        else:
            # TODO sort this
            # Make the missing value a category
            df[col].cat = df[col].cat.add_categories('missing')
            print(df[col].unique())
            print(type(df[col].cat))
            print(df[col].cat.categories)
    
    # Normalise (min-max) numeric columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        df[col] = (df[col]-df[col].min())/(df[col].max()-df[col].min())

    # Drop low-variance (<=0.05) numeric columns
    numeric_col_variance = df[numeric_cols].var()
    variance_threshold_fail_cols = numeric_col_variance[numeric_col_variance < variance_threshold]
    df = df.drop(variance_threshold_fail_cols.index)
    
    # Removing duplicate columns
    duplicate_cols = df.transpose().duplicated()
    print(f"Removing duplicate columns '{list(df.columns[duplicate_cols])}'")
    df = df.drop(list(df.columns[duplicate_cols]), axis=1)

    # Drop second of high-correlation numeric columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    high_cor_cols = _get_high_corr_cols(df[numeric_cols])
    df = df.drop(high_cor_cols, axis=1)

    # TODO drop low "uniqueness" categorical cols
    cat_cols = df.select_dtypes(include=['category']).drop(['Class'], axis=1).columns
    high_monotony_cols = []
    for col in cat_cols:
        if df[col].value_counts(normalize=True).max() >= cat_monotony_threshold:
            high_cor_cols.append(col)
    df = df.drop(high_monotony_cols)

    # TODO Drop highly "correlated" categorical

    # TODO anything else??

    ## Removing Useless Columns
    # Remove high-correlation columns

    df_train:pd.DataFrame = df.iloc[:1000]
    df_test = df.iloc[1000:]

    # Remove duplicate rows
    df_train = df_train.drop_duplicates()

    return df_train, df_test

def get_prepped_dataset():
    df = _get_dataset()

    return _prep_dataset(df)

def main():
    df_in = _get_dataset()

    df_train, df_test = _prep_dataset(df_in)

    print("Done! Saving...")

    ## Final saving
    df_in.to_csv('df_in_new.csv', index=False)
    df_train.to_csv('df_out_new.csv', index=False)

if __name__ == '__main__':
    main()