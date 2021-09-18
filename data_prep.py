import pandas as pd

def main():
    # Read in the data
    df_in = pd.read_csv('data/data2021.student.csv')

    ## Basic operations
    # Drop rows to predict
    df = df_in.dropna(subset=['Class'])

    # Remove the 'ID' index (since it is useless and interferes with everything else)
    df = df.drop(['ID'], axis=1)

    ## Removing duplicate rows
    df = df.drop_duplicates()

    ## Removing duplicate columns
    duplicate_cols = df.transpose().duplicated()
    print(f"Removing duplicate columns '{list(df.columns[duplicate_cols])}'")
    df = df.drop(list(df.columns[duplicate_cols]), axis=1)

    ## Removing Useless Columns
    print(f"Removing known 'duplicate' rows (100% correlation) ")

    ## Final saving
    df_in.to_csv('df_in.csv', index=False)
    df.to_csv('df_out.csv', index=False)



if __name__ == '__main__':
    main()