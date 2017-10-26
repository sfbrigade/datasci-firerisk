import pandas as pd
import numpy as np


def get_column_report(df, numerical='nan'):
	column_report = []
	for column in df.columns:
		unique = df[column].unique()
		sample = np.nan
		for value in unique:
			if value is not np.nan:
				sample = value
				break
		if numerical=='str':
			nans = len(df[df[column]=='0'])
		elif numerical=='int':
			nans = len(df[df[column]==0])
		else:
			nans = df[column].isnull().sum()
		pct_nan = 100. * nans / df.shape[0]
		column_report.append([column, df[column].dtype, len(unique), sample, nans, pct_nan])
	columns=["Column Name", "Data Type", "Unique Count", "Sample Value", "NaNs/Zeros", "% NaN"]
	column_report = pd.DataFrame(column_report, columns=columns).round(2)
	column_report.sort_values(by="NaNs/Zeros", inplace=True)
	return column_report


