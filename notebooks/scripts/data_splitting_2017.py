#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import re
from tqdm import tqdm
import datetime as datetime
from sklearn.model_selection import train_test_split

filename = 'bpi_2017.csv'
print('Reading the dataset ' + filename)
df = pd.read_csv(filename, parse_dates = ['time:timestamp'])
df.info()
# The default name indicating the case ID is case:concept:name
# concept:name is the event
# time:timestamp is the corresponding timestamp

print('Obtaining datetime format from the dataset')
# Obtain date (datetime format) from datatype of time:timestamp 
df['Date'] = np.array(df['time:timestamp'].values, dtype = 'datetime64[D]').astype(datetime.datetime)
df

print('Determining the training and testing data\'s date boundaries')
# Determine training and testing data's date boundaries
date_unique = sorted(df['Date'].unique())
total_date = len(date_unique)
all_train_nr = round(total_date * 0.8)
date_before_test = date_unique[all_train_nr - 1]

print('Removing entries with case ID across date boundaries')
# Remove entries with case ID across date boundaries
small_df = df[['Date', 'case:concept:name']].drop_duplicates()
small_df_1 = small_df[small_df['Date'] <= date_before_test]
small_df_2 = small_df[small_df['Date'] > date_before_test]
small_df_inter = set(small_df_1['case:concept:name'].unique()).intersection(set(small_df_2['case:concept:name'].unique()))
case_unique_train = sorted(list(set(small_df_1['case:concept:name'].unique()) - small_df_inter))
case_unique_test = sorted(list(set(small_df_2['case:concept:name'].unique()) - small_df_inter))

print('Determining training and testing data\'s ID boundaries')
# Determine training and testing data's ID boundaries
all_case = sorted(df['case:concept:name'].unique())
total_case = len(all_case)
all_train_case = round(total_case * 0.8)
case_all_train = sorted(all_case)[: all_train_case]
case_test = sorted(all_case)[all_train_case: ]

print('Combining ID boundaries ')
# Combine ID boundaries and time boundaries
final_all_train = sorted(list(set(case_unique_train).intersection(set(case_all_train))))
final_test = sorted(list(set(case_unique_test).intersection(set(case_test))))

print('Splitting training and validation dataset')
# Split training and validation dataset
final_train, final_val = train_test_split(final_all_train, test_size = 0.2)

# Split the dataset
df_train = df[df['case:concept:name'].isin(final_train)]
df_val = df[df['case:concept:name'].isin(final_val)]
df_test = df[df['case:concept:name'].isin(final_test)]
df_train = df_train.drop(columns = ['Unnamed: 0', 'Date']).reset_index(drop = True)
df_val = df_val.drop(columns = ['Unnamed: 0', 'Date']).reset_index(drop = True)
df_test = df_test.drop(columns = ['Unnamed: 0', 'Date']).reset_index(drop = True)

print('Exporting to CSVs')
df_train.to_csv('bpi2017_train.csv')
df_val.to_csv('bpi2017_val.csv')
df_test.to_csv('bpi2017_test.csv')