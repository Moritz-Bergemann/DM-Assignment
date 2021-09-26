import pandas as pd

def _get_dataset():
    # Read in the data
    return pd.read_csv('./data/data2021.student.csv')

def _prep_dataset(df_in):
    ## Basic operations
    # Drop rows to predict
    df = df_in.dropna(subset=['Class'])

    # Remove the 'ID' index (since it is useless and interferes with everything else)
    df = df.drop(['ID'], axis=1)

    ## Type Conversions
    # Convert attributes with less than 30 unique values to categorical
    lt_30_unique_cols = [col for col in df.columns if df[col].nunique() <= 30]
    df[lt_30_unique_cols] = df[lt_30_unique_cols].astype('category')

    # Convert numeric attributes to categorical through binning
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        # pd.cut(df[col], bins=10)
        df[col] = pd.cut(df[col], bins=10, labels=[f'bin-{ii}' for ii in range(10)])

    ## Removing duplicate rows
    print("Removing duplicate rows...")
    df = df.drop_duplicates()

    ## Removing duplicate columns
    duplicate_cols = df.transpose().duplicated()
    print(f"Removing duplicate columns '{list(df.columns[duplicate_cols])}'")
    df = df.drop(list(df.columns[duplicate_cols]), axis=1)

    ## Removing Useless Columns
    # Remove high-correlation columns
    print(f"Removing second of known 'duplicate' columns (100% correlation): [C4, C25]")
    df = df.drop(['C25'], axis=1)

    # Removing columns with >80% missing values    
    cols_missing_values = df.isnull().mean()
    cols_missing_values_80 = cols_missing_values[cols_missing_values > 0.8]
    print(f"Removing columns with more than 80% missing values: {list(cols_missing_values_80.index)}")
    df = df.drop(cols_missing_values_80.index, axis=1)

    # Removing columns with all the same values
    cols_nunique_1 = df.nunique()[df.nunique() == 1]
    df = df.drop(cols_nunique_1.index, axis=1)
    print(f"Removing columns with only 1 unique value: {list(cols_nunique_1.index)}")

    # Fill in null values with column mode
    print(f"Filling missing values with column node")
    for col in df.columns:
        df[col] = df[col].fillna(value=df[col].mode())

    return df

def get_prepped_dataset():
    df = _get_dataset()

    return _prep_dataset(df)

def main():
    df_in = _get_dataset()

    df = _prep_dataset(df_in)

    print("Done! Saving...")

    ## Final saving
    df_in.to_csv('df_in.csv', index=False)
    df.to_csv('df_out.csv', index=False)


if __name__ == '__main__':
    main()