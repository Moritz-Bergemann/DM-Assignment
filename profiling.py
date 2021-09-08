# Load libraries
import pandas as pd

# Load the dataset
df = pd.read_csv('data2021.student.csv')

# Do Basic Profiling

from pandas_profiling import ProfileReport
report = ProfileReport(df)
report.to_file(output_file='output.html')