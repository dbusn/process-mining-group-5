#!/usr/bin/env python
# coding: utf-8

# In[1]:

import psutil
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import datetime as datetime


# ### Data preparation

# In[2]:
process  = psutil.Process(os.getpid())

df_train = pd.read_csv("bpi2017_train.csv", parse_dates = ['time:timestamp'])
df_val = pd.read_csv("bpi2017_val.csv", parse_dates = ['time:timestamp'])
df_test = pd.read_csv("bpi2017_test.csv", parse_dates = ['time:timestamp'])

# The default name indicating the case ID is case:concept:name
# concept:name is the event
# time:timestamp is the corresponding timestamp
# Load the datasets, sort them on case and consequently timestamp, then reset the index
df_train = df_train.sort_values(by = ['case:concept:name', 'time:timestamp']).reset_index()
df_val = df_val.sort_values(by = ['case:concept:name', 'time:timestamp']).reset_index()
df_test = df_test.sort_values(by = ['case:concept:name', 'time:timestamp']).reset_index()

# Remove obsolete columns
df_train = df_train.drop(['index', 'Unnamed: 0'], axis = 1)
df_val = df_val.drop(['index', 'Unnamed: 0'], axis = 1)
df_test = df_test.drop(['index', 'Unnamed: 0'], axis = 1)


# # 1. Calculate the time difference

# In[3]:


# Cumulative sum function to be used later
def CumSum(lists):
    # Returns the cumulative sum of a list
    length = len(lists)
    cu_list = [sum(lists[0: x: 1]) for x in range(0, length + 1)]
    return cu_list[1: ]


# In[4]:


def time_difference(df):
    # Calculate time difference between each row
    df['time_diff'] = df['time:timestamp'].diff().dt.total_seconds()
    # Set the time difference of the 1st row to 0 as it's currently NaN
    df.at[0, 'time_diff'] = 0
    # Count number of steps per process
    length_per_case_List = df.groupby(['case:concept:name'])['time_diff'].count().tolist()

    # Using the cumulative sum we get all the positions that are a first step in a process
    # And then the time difference can be set to 0
    position_lst = CumSum(length_per_case_List)
    for i in tqdm(position_lst):
        df.at[i, 'time_diff'] = 0
    # For Loop mysteriously creates an empty row at the end of the df, gotta delete it
    df = df.iloc[: -1]

    # Unzip the position list to get the number of each steps of each process, make that into a list
    step_in_process = []
    for x in tqdm(length_per_case_List):
        for y in range(x):
            step_in_process.append(y + 1)
    # Assign position number to each row/process
    df['position'] = step_in_process
    return df


# In[5]:


# Apply the above changes to all dataframes
# The warnings are obsolete, it's because it uses .at which is considerably faster than .loc
df_train = time_difference(df_train)
df_val = time_difference(df_val)
df_test = time_difference(df_test)


# # 2. Baseline Time Prediction (Only on Training Dataset)

# In[6]:


# Get the list of position number
step_in_process_train = df_train['position'].tolist()
# Calculate mean time difference grouped by position based on the number of cases
mean_time_lst = df_train.groupby('position')['time_diff'].mean().tolist()

# Create the predicted time column per entry using the mean time difference
pred_time_lst_train = [mean_time_lst[j - 1] for j in step_in_process_train]
df_train['baseline_predicted_time'] = pred_time_lst_train
df_train


# # 3. Baseline Case Prediction (only on the training dataset)

# In[7]:


position_df = df_train.groupby('position').agg(pd.Series.mode)['concept:name'].to_frame()
df_train = pd.merge(df_train, position_df, on='position')
df_train


# # 4. Apply Above Calculated Mean Time to Validation and Test datasets

# In[8]:


def apply_time_prediction(df):
    # Get the list of position number
    step_in_process = df['position'].tolist()

    # Create the predicted time column per entry using the mean time difference
    # If some position numbers are not shown in the training dataset, its predicted time will be 0
    pred_time_lst = []
    for j in step_in_process:
        if j <= len(mean_time_lst):
            pred_time_lst.append(mean_time_lst[j - 1])
        else:
            pred_time_lst.append(0)
    df['baseline_predicted_time'] = pred_time_lst
    return df


# In[9]:


# Apply the above changes to all dataframes
df_val = apply_time_prediction(df_val)
df_test = apply_time_prediction(df_test)


# # 5. Apply Baseline Case prediction to Validation and Test datasets

# In[10]:


def apply_case_prediction(df: pd.DataFrame) -> pd.DataFrame:
    # Merge the dataframe with position with the dataframe prediction is applied to
    df = pd.merge(df, position_df, on='position')
    
    # Sort values by timestamp, like in the original dataset
    df.sort_values(by=['time:timestamp'], inplace=True)
    
    # Rename the column labels due to applying merge
    df.rename(columns = {"concept:name_y":"baseline_action_pred", "concept:name_x":"concept:name"}, inplace=True)
    
    return df


# In[11]:


df_val = apply_case_prediction(df_val)
df_test = apply_case_prediction(df_test)

print("Total memory memory used: " + str((process.memory_info().rss) >> 20) + " MB")

print("CPU Time:" + str(process.cpu_times().user + process.cpu_times().system))
