# # Import modules
import copy
import datetime as datetime
import logging
import math
import multiprocessing as mp
import os
import time
import re
import warnings
from functools import partial
from operator import itemgetter

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import jellyfish as jf
import scipy
import networkx as nx
import numpy as np
import pandas as pd
import tensorflow as tf

# Supress tensorflow's logging info
tf.get_logger().setLevel(logging.ERROR)
tf.autograph.set_verbosity(1)

from dateutil.parser import parse
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential, load_model
from scipy.stats import pearsonr
from sklearn import metrics
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    VotingRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from tensorflow import concat, matmul, multiply, sigmoid, transpose
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import (
    LSTM,
    BatchNormalization,
    Concatenate,
    Dense,
    Dropout,
    Embedding,
    Input,
    Layer,
)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.optimizers import Nadam as nadam_v2
from tqdm import tqdm

# For exporting metrics
f = open("metrics.txt", "w")
f.write("DBL Process Mining Demo tool v3\n")
# Reopen with append later
f.close()

# Supress the warnings
warnings.filterwarnings(action="ignore")

# Read the split dataset
# Available on github/data/splits
df_train = pd.read_csv("bpi2017_train_filtered.csv")
df_test = pd.read_csv("bpi2017_test_filtered.csv")
df_val = pd.read_csv("bpi2017_val_filtered.csv")

n = 2000
df_train = df_train[:n]
df_test = df_test[:n]
df_val = df_val[:n]

# Keep the original files so they don't have to be reloaded for each model
orig_train = df_train
orig_test = df_test
orig_val = df_val

result_df = pd.DataFrame()

print("----- RANDOM FOREST -----")
# Perform conversion
df_train["Date"] = np.array(
    df_train["time:timestamp"].values, dtype="datetime64"
).astype(datetime.datetime)
df_train["time:unix"] = (df_train["Date"] - pd.Timestamp("1970-01-01")) // pd.Timedelta(
    "1s"
)
df_test["Date"] = np.array(df_test["time:timestamp"].values, dtype="datetime64").astype(
    datetime.datetime
)
df_test["time:unix"] = (df_test["Date"] - pd.Timestamp("1970-01-01")) // pd.Timedelta(
    "1s"
)
df_val["Date"] = np.array(df_val["time:timestamp"].values, dtype="datetime64").astype(
    datetime.datetime
)
df_val["time:unix"] = (df_val["Date"] - pd.Timestamp("1970-01-01")) // pd.Timedelta(
    "1s"
)


# Assign position
def assign_position(df: pd.DataFrame) -> pd.DataFrame:
    # Count number of processes per trace/ID
    count_lst = df.groupby("case:concept:name").count()["lifecycle:transition"].tolist()
    position_lst_1 = [list(range(1, i + 1)) for i in count_lst]
    position_lst = []
    for i in position_lst_1:
        for j in i:
            position_lst.append(j)
    df["position"] = position_lst
    return df


df_train = assign_position(df_train)
df_val = assign_position(df_val)
df_test = assign_position(df_test)

# Define mapping for lifecycle:transition
mapping_train = {
    item: i for i, item in enumerate(df_train["lifecycle:transition"].unique())
}
mapping_test = {
    item: i for i, item in enumerate(df_test["lifecycle:transition"].unique())
}
mapping_val = {
    item: i for i, item in enumerate(df_val["lifecycle:transition"].unique())
}

# Apply mapping
df_train["transition"] = df_train["lifecycle:transition"].apply(
    lambda x: mapping_train[x]
)
df_test["transition"] = df_test["lifecycle:transition"].apply(lambda x: mapping_test[x])
df_val["transition"] = df_val["lifecycle:transition"].apply(lambda x: mapping_val[x])


# Define mapping for Action
mapping_train = {item: i for i, item in enumerate(df_train["Action"].unique())}
mapping_test = {item: i for i, item in enumerate(df_test["Action"].unique())}
mapping_val = {item: i for i, item in enumerate(df_val["Action"].unique())}

# Apply mapping
df_train["action"] = df_train["Action"].apply(lambda x: mapping_train[x])
df_test["action"] = df_test["Action"].apply(lambda x: mapping_test[x])
df_val["action"] = df_val["Action"].apply(lambda x: mapping_val[x])


# Define mapping for case:LoanGoal
mapping_train = {item: i for i, item in enumerate(df_train["case:LoanGoal"].unique())}
mapping_test = {item: i for i, item in enumerate(df_test["case:LoanGoal"].unique())}
mapping_val = {item: i for i, item in enumerate(df_val["case:LoanGoal"].unique())}

# Apply mapping
df_train["goal"] = df_train["case:LoanGoal"].apply(lambda x: mapping_train[x])
df_test["goal"] = df_test["case:LoanGoal"].apply(lambda x: mapping_test[x])
df_val["goal"] = df_val["case:LoanGoal"].apply(lambda x: mapping_val[x])


# Define mapping for case:LoanGoal
mapping_train = {
    item: i for i, item in enumerate(df_train["case:ApplicationType"].unique())
}
mapping_test = {
    item: i for i, item in enumerate(df_test["case:ApplicationType"].unique())
}
mapping_val = {
    item: i for i, item in enumerate(df_val["case:ApplicationType"].unique())
}

# Apply mapping
df_train["type"] = df_train["case:ApplicationType"].apply(lambda x: mapping_train[x])
df_test["type"] = df_test["case:ApplicationType"].apply(lambda x: mapping_test[x])
df_val["type"] = df_val["case:ApplicationType"].apply(lambda x: mapping_val[x])

df_1 = df_train.copy()
df_2 = df_val.copy()
# Define predictors
predictors = ["time:unix", "transition", "type", "action"]
# worse accuracy (compared to only using time:unix and transition): position, goal
# better accuracy (---): action, type

# Define the classifier
rfc = RandomForestClassifier(n_estimators=50)

# Fit the model
rfc.fit(df_1[predictors], df_1["concept:name"])
pred_val = rfc.predict(df_2[predictors])
df_2["predicted_action"] = pred_val
#
actions_taken = df_2["concept:name"]
actions_taken = actions_taken[1:]

actions_pred = df_2["predicted_action"]
actions_pred = actions_pred[:-1]

test = pd.concat([actions_taken, actions_pred], axis=1)
test.dropna(axis=0, inplace=True)

predictors = ["time:unix", "transition", "type", "action"]

pred_test = rfc.predict(df_test[predictors])
pred_val = rfc.predict(df_val[predictors])

result_df["time:timestamp"] = df_test["time:timestamp"]
result_df["Event"] = df_test["concept:name"]
result_df["Random_forest_pred"] = pred_test

df_test["predicted_action"] = pred_test
df_val["predicted_action"] = pred_val

actions_taken = df_val["concept:name"]
actions_taken = actions_taken[1:]

actions_pred = df_val["predicted_action"]
actions_pred = actions_pred[:-1]

test = pd.concat([actions_taken, actions_pred], axis=1)
test.dropna(axis=0, inplace=True)

random_forest_acc = str(
    round(
        metrics.accuracy_score(test["concept:name"], test["predicted_action"]) * 100, 2,
    )
)
# random_forest_mae = str(
#     round(
#         metrics.mean_absolute_error(test["concept:name"], test["predicted_action"]), 2,
#     )
# )
# random_forest_rmse = str(
#     round(
#         metrics.mean_squared_error(
#             test["concept:name"], test["predicted_action"], squared=False
#         ),
#         2,
#     )
# )

metrics_out = open("metrics.txt", "a")

metrics_out.write("Random forest accuracy: " + random_forest_acc + "%\n")
# metrics_out.write("Random forest mean absolute error: " + random_forest_mae + "\n")
# metrics_out.write("Random forest RMSE: " + random_forest_rmse + "\n")

print(
    "RANDOM FOREST ACCURACY: ", random_forest_acc, "%",
)

# print(
#     "RANDOM FOREST MEAN ABSOLUTE ERROR: ", random_forest_mae, "%",
# )

# print("RANDOM FOREST RMSE: ", random_forest_rmse, "%")

# 5s pause to examine the metrics
time.sleep(5)


print("----- LOGISTIC REGRESSION -----")
df_train = orig_train
df_test = orig_test
df_val = orig_val


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def softmax(zs):
    return np.exp(zs) / sum(np.exp(zs))


# factorization of categorical atrtibutes (of interest) of the training data
df_train_f = df_train.copy()
vals, code_Action = pd.factorize(df_train_f["Action"])
df_train_f["Action"] = vals

vals, code_Origin = pd.factorize(df_train_f["EventOrigin"])
df_train_f["EventOrigin"] = vals

vals, code_lifecycle_transition = pd.factorize(df_train_f["lifecycle:transition"])
df_train_f["lifecycle:transition"] = vals

vals, code_loan_goal = pd.factorize(df_train_f["case:LoanGoal"])
df_train_f["case:LoanGoal"] = vals

vals, code_appl_type = pd.factorize(df_train_f["case:ApplicationType"])
df_train_f["case:ApplicationType"] = vals

vals, code_concept_name = pd.factorize(df_train_f["concept:name"])
df_train_f["concept:name"] = vals

# and for test data
df_test_f = df_test.copy()
vals, code_Action = pd.factorize(df_test_f["Action"])
df_test_f["Action"] = vals

vals, code_Origin = pd.factorize(df_test_f["EventOrigin"])
df_test_f["EventOrigin"] = vals

vals, code_lifecycle_transition = pd.factorize(df_test_f["lifecycle:transition"])
df_test_f["lifecycle:transition"] = vals

vals, code_loan_goal = pd.factorize(df_test_f["case:LoanGoal"])
df_test_f["case:LoanGoal"] = vals

vals, code_appl_type = pd.factorize(df_test_f["case:ApplicationType"])
df_test_f["case:ApplicationType"] = vals

vals, code_concept_name = pd.factorize(df_test_f["concept:name"])
df_test_f["concept:name"] = vals

# to see the "code" - so which number corresponds to which class - print the code_... for the attribute of interest
# select the features that will be used for prediction
features = [
    "Action",
    "EventOrigin",
    "lifecycle:transition",
    "case:LoanGoal",
    "case:ApplicationType",
    "case:RequestedAmount",
]


# split the predictors and the outcome variable
X_train = df_train_f[features]
y_train = df_train_f["concept:name"]
X_test = df_test_f[features]
y_test = df_test_f["concept:name"]


# initialize and use the scaler
standardizer = StandardScaler()
X_train_st = standardizer.fit_transform(X_train)
X_test_st = standardizer.transform(X_test)

# fit the model
lr = LogisticRegression(multi_class="multinomial")
lr.fit(X_train_st, y_train)

# Obtain decision function for first training unit
# lr.decision_function(X_train_st[0])


# obtain the prediction for the 1st unit:
prediction = np.argmax(
    softmax(lr.decision_function(X_train_st[0].reshape(1, -1))[0]).round(1)
)
code_concept_name[prediction]

prediction = np.argmax(
    softmax(lr.decision_function(X_train_st[15].reshape(1, -1))[0]).round(1)
)

lr.predict_proba(X_train_st[:2]).round(3)

# evaluate the model (??)
lr.score(X_test_st, y_test)

# initialize column for predictions
df_train["predicted_action"] = 0
df_train["true_action"] = df_train_f["concept:name"]

# add predictions to the column in df_train
for x in tqdm(range(0, len(X_train))):
    df_train["predicted_action"].loc[x] = np.argmax(
        lr.decision_function(X_train_st[x].reshape(1, -1))[0]
    )

df_test["predicted_action"] = 0
df_test["true_action"] = df_test_f["concept:name"]

for x in tqdm(range(0, len(X_test))):
    df_test["predicted_action"].loc[x] = np.argmax(
        lr.decision_function(X_test_st[x].reshape(1, -1))[0]
    )

# make a dictionary to translate the factors into initial event names
keys = [i for i in range(0, len(code_concept_name))]
codes = code_concept_name
dict_concept_name = {keys[i]: codes[i] for i in range(len(codes))}

# translate the factors into initial event name
df_test["predicted_event_name"] = X_test["Action"].map(dict_concept_name)
result_df["log_reg_pred"] = df_test["predicted_event_name"]

logistic_regression_acc = str(
    round(
        metrics.accuracy_score(df_test["predicted_action"], df_test["true_action"])
        * 100,
        2,
    )
)
logistic_regression_mae = str(
    round(
        metrics.mean_absolute_error(
            df_test["predicted_action"], df_test["true_action"]
        ),
        2,
    )
)
logistic_regression_rmse = str(
    round(
        metrics.mean_squared_error(
            df_test["predicted_action"], df_test["true_action"], squared=False
        ),
        2,
    )
)


metrics_out.write("Logistic regression accuracy: " + logistic_regression_acc + "%\n")
metrics_out.write("Logistic regression MAE: " + logistic_regression_mae + "\n")
metrics_out.write("Logistic regression RMSE: " + logistic_regression_rmse + "\n")


print(
    "LOGISTIC REGRESSION ACCURACY:", logistic_regression_acc, "%",
)

print(
    "LOGISTIC REGRESSION MEAN ABSOLUTE ERROR:", logistic_regression_mae,
)

print("LOGISTIC REGRESION RMSE:", logistic_regression_rmse)

time.sleep(5)

print("----- SVM REGRESSION ------")
df_train = pd.read_csv("bpi2017_train_filtered.csv", parse_dates=["time:timestamp"])
df_test = pd.read_csv("bpi2017_test_filtered.csv", parse_dates=["time:timestamp"])
df_val = pd.read_csv("bpi2017_val_filtered.csv", parse_dates=["time:timestamp"])

df_train = df_train[:n]
df_test = df_test[:n]
df_val = df_val[:n]

orig_train = df_train
orig_test = df_test
orig_val = df_val

df_train = df_train.sort_values(
    by=["case:concept:name", "time:timestamp"]
).reset_index()
df_val = df_val.sort_values(by=["case:concept:name", "time:timestamp"]).reset_index()
df_test = df_test.sort_values(by=["case:concept:name", "time:timestamp"]).reset_index()

# Cumulative sum function to be used later
def CumSum(lists):
    # Returns the cumulative sum of a list
    length = len(lists)
    cu_list = [sum(lists[0:x:1]) for x in range(0, length + 1)]
    return cu_list[1:]


def time_difference(df):
    # Calculate time difference between each row
    df["time_diff"] = df["time:timestamp"].diff().dt.total_seconds()
    # Set the time difference of the 1st row to 0 as it's currently NaN
    df.at[0, "time_diff"] = 0
    # Count number of steps per process
    length_per_case_List = (
        df.groupby(["case:concept:name"])["time_diff"].count().tolist()
    )

    # Using the cumulative sum we get all the positions that are a first step in a process
    # And then the time difference can be set to 0
    position_lst = CumSum(length_per_case_List)
    for i in tqdm(position_lst):
        df.at[i, "time_diff"] = 0
    # For Loop mysteriously creates an empty row at the end of the df, gotta delete it
    df = df.iloc[:-1]

    # Unzip the position list to get the number of each steps of each process, make that into a list
    step_in_process = []
    for x in tqdm(length_per_case_List):
        for y in range(x):
            step_in_process.append(y + 1)
    # Assign position number to each row/process
    df["position"] = step_in_process
    return df


# Apply the above changes to all dataframes
# The warnings are obsolete, it's because it uses .at which is considerably faster than .loc
df_train = time_difference(df_train)
df_val = time_difference(df_val)
df_test = time_difference(df_test)

# one hot encoding
df_train_f = pd.get_dummies(df_train, columns=["concept:name"])
df_test_f = pd.get_dummies(df_test, columns=["concept:name"])

# factorization of categorical atrtibutes (of interest) of the training data
# should be replaced with some "smarter" feature transformation
vals, code_Action = pd.factorize(df_train_f["Action"])
df_train_f["Action"] = vals

vals, code_Origin = pd.factorize(df_train_f["EventOrigin"])
df_train_f["EventOrigin"] = vals

vals, code_lifecycle_transition = pd.factorize(df_train_f["lifecycle:transition"])
df_train_f["lifecycle:transition"] = vals

vals, code_loan_goal = pd.factorize(df_train_f["case:LoanGoal"])
df_train_f["case:LoanGoal"] = vals

vals, code_appl_type = pd.factorize(df_train_f["case:ApplicationType"])
df_train_f["case:ApplicationType"] = vals

# and for test data
vals, code_Action = pd.factorize(df_test_f["Action"])
df_test_f["Action"] = vals

vals, code_Origin = pd.factorize(df_test_f["EventOrigin"])
df_test_f["EventOrigin"] = vals

vals, code_lifecycle_transition = pd.factorize(df_test_f["lifecycle:transition"])
df_test_f["lifecycle:transition"] = vals

vals, code_loan_goal = pd.factorize(df_test_f["case:LoanGoal"])
df_test_f["case:LoanGoal"] = vals

vals, code_appl_type = pd.factorize(df_test_f["case:ApplicationType"])
df_test_f["case:ApplicationType"] = vals

# select the features that will be used for prediction
features = [
    "Action",
    "EventOrigin",
    "lifecycle:transition",
    "case:LoanGoal",
    "case:ApplicationType",
    "case:RequestedAmount",
    "position",
    "concept:name_A_Accepted",
    "concept:name_A_Cancelled",
    "concept:name_A_Complete",
    "concept:name_A_Concept",
    "concept:name_A_Create Application",
    "concept:name_A_Denied",
    "concept:name_A_Incomplete",
    "concept:name_A_Pending",
    "concept:name_A_Submitted",
    "concept:name_A_Validating",
    "concept:name_O_Accepted",
    "concept:name_O_Cancelled",
    "concept:name_O_Create Offer",
    "concept:name_O_Created",
    "concept:name_O_Refused",
    "concept:name_O_Returned",
    "concept:name_O_Sent (mail and online)",
    "concept:name_O_Sent (online only)",
    "concept:name_W_Call after offers",
    "concept:name_W_Call incomplete files",
    "concept:name_W_Complete application",
    "concept:name_W_Handle leads",
    "concept:name_W_Validate application",
]

# split the predictors and the outcome variable
X_train = df_train_f[features]
y_train = df_train_f["time_diff"]
X_test = df_test_f[features]
y_test = df_test_f["time_diff"]

# Initialize and fit the model (on a slice - to have a reasonable running time)
regr = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.01, degree=3, verbose=True))
regr.fit(X_train, y_train)


# ## Make predictions for a slice of test

predictions = regr.predict(X_test)
result_df["svm_pred"] = predictions
# compare the predictions with actual data and check accuracy (proportion of correct predictions)
compare_predictions = pd.DataFrame()
compare_predictions["true"] = y_test[:n]
compare_predictions["predicted"] = predictions
compare_predictions["position"] = X_test["position"][:n]
compare_predictions["diff"] = (
    compare_predictions["true"] - compare_predictions["predicted"]
)

svm_acc = str(round(abs(regr.score(X_test, y_test) * 100), 2))
svm_mae = str(round(metrics.mean_absolute_error(predictions, y_test[:n]), 2))
svm_rmse = str(
    round(metrics.mean_squared_error(predictions, y_test[:n], squared=False), 2)
)


metrics_out.write("SVM Accuracy: " + svm_acc + "%\n")
metrics_out.write("SVM MAE: " + svm_mae + "\n")
metrics_out.write("SVM RMSE: " + svm_rmse + "\n")


print("SVM ACCURACY:", svm_acc, "%")
print("SVM MAE:", svm_mae)
print("SVM RMSE:", svm_rmse)

time.sleep(5)

print("----- VOTING REGRESSION -----")
df_train = orig_train
df_test = orig_test
df_val = orig_val

df_train = df_train.sort_values(
    by=["case:concept:name", "time:timestamp"]
).reset_index()
df_val = df_val.sort_values(by=["case:concept:name", "time:timestamp"]).reset_index()
df_test = df_test.sort_values(by=["case:concept:name", "time:timestamp"]).reset_index()

# Cumulative sum function to be used later
def CumSum(lists):
    # Returns the cumulative sum of a list
    length = len(lists)
    cu_list = [sum(lists[0:x:1]) for x in range(0, length + 1)]
    return cu_list[1:]


def time_difference(df):
    # Calculate time difference between each row
    df["time_diff"] = df["time:timestamp"].diff().dt.total_seconds()
    # Set the time difference of the 1st row to 0 as it's currently NaN
    df.at[0, "time_diff"] = 0
    # Count number of steps per process
    length_per_case_List = (
        df.groupby(["case:concept:name"])["time_diff"].count().tolist()
    )

    # Using the cumulative sum we get all the positions that are a first step in a process
    # And then the time difference can be set to 0
    position_lst = CumSum(length_per_case_List)
    for i in tqdm(position_lst):
        df.at[i, "time_diff"] = 0
    # For Loop mysteriously creates an empty row at the end of the df, gotta delete it
    df = df.iloc[:-1]

    # Unzip the position list to get the number of each steps of each process, make that into a list
    step_in_process = []
    for x in tqdm(length_per_case_List):
        for y in range(x):
            step_in_process.append(y + 1)
    # Assign position number to each row/process
    df["position"] = step_in_process
    return df


df_train = time_difference(df_train)
df_val = time_difference(df_val)
df_test = time_difference(df_test)

# factorization of categorical atrtibutes (of interest) of the training data
df_train_f = df_train.copy()
vals, code_Action = pd.factorize(df_train_f["Action"])
df_train_f["Action"] = vals

vals, code_Origin = pd.factorize(df_train_f["EventOrigin"])
df_train_f["EventOrigin"] = vals

vals, code_lifecycle_transition = pd.factorize(df_train_f["lifecycle:transition"])
df_train_f["lifecycle:transition"] = vals

vals, code_loan_goal = pd.factorize(df_train_f["case:LoanGoal"])
df_train_f["case:LoanGoal"] = vals

vals, code_appl_type = pd.factorize(df_train_f["case:ApplicationType"])
df_train_f["case:ApplicationType"] = vals

# and for test data
df_test_f = df_test.copy()
vals, code_Action = pd.factorize(df_test_f["Action"])
df_test_f["Action"] = vals

vals, code_Origin = pd.factorize(df_test_f["EventOrigin"])
df_test_f["EventOrigin"] = vals

vals, code_lifecycle_transition = pd.factorize(df_test_f["lifecycle:transition"])
df_test_f["lifecycle:transition"] = vals

vals, code_loan_goal = pd.factorize(df_test_f["case:LoanGoal"])
df_test_f["case:LoanGoal"] = vals

vals, code_appl_type = pd.factorize(df_test_f["case:ApplicationType"])
df_test_f["case:ApplicationType"] = vals

vals, code_concept_name = pd.factorize(df_test_f["concept:name"])

df_train_10 = df_train_f[df_train_f["position"] <= 10][:2000]
df_test_10 = df_test_f[df_test_f["position"] <= 10][:2000]

df_train_5 = df_train_f[df_train_f["position"] <= 5][:2000]
df_test_5 = df_test_f[df_test_f["position"] <= 5][:2000]

features = [
    "lifecycle:transition",
    "case:LoanGoal",
    "case:ApplicationType",
    "case:RequestedAmount",
    "position",
]

X_train = df_train_f[features][:2000]
y_train = df_train_f["time_diff"][:2000]
X_test = df_test_f[features][:2000]
y_test = df_test_f["time_diff"][:2000]

n1, n2, n4, n5, m4, m5, c2, c4, c5, norm, fi = (
    1000,
    1000,
    1000,
    1000,
    50,
    100,
    "squared_error",
    "squared_error",
    "poisson",
    False,
    False,
)
reg1 = GradientBoostingRegressor(n_estimators=n1)
reg2 = RandomForestRegressor(n_estimators=n2, criterion=c2)
reg3 = LinearRegression(normalize=norm, fit_intercept=fi)
reg4 = RandomForestRegressor(n_estimators=n4, max_depth=m4, criterion=c4)
reg5 = RandomForestRegressor(n_estimators=n5, max_depth=m5, criterion=c5)

reg1.fit(X_train, y_train)
reg2.fit(X_train, y_train)
reg3.fit(X_train, y_train)
reg4.fit(X_train, y_train)
reg5.fit(X_train, y_train)

ereg = VotingRegressor(
    [("gb", reg1), ("rf", reg2), ("lr", reg3), ("rf2", reg4), ("rf3", reg5)],
    verbose=True,
)
ereg.fit(X_train, y_train)
voting_pred = ereg.predict(X_test)
result_df["voting_pred"] = voting_pred

vregress_acc = str(round(ereg.score(X_train, y_train), 2))
vreg_mae = str(round(metrics.mean_absolute_error(predictions, y_test[:n]), 2))
vreg_rmse = str(
    round(metrics.mean_squared_error(predictions, y_test[:n], squared=False), 2)
)


metrics_out.write("Voting regression acurracy: " + vregress_acc + "%\n")
metrics_out.write("Voting regression MAE: " + vreg_mae + "\n")
metrics_out.write("Voting regression RMSE: " + vreg_rmse + "\n")


print("VOTING REGRESSION ACURRACY: ", vregress_acc, "%")
print("VOTING REGRESSION MAE: ", vreg_mae)
print("VOTING REGRESSION RMSE: ", vreg_rmse)

time.sleep(5)

print("----- RNN -----")

df_train = pd.read_csv("bpi2017_train_filtered.csv",  parse_dates = ['time:timestamp'])
df_val = pd.read_csv("bpi2017_val_filtered.csv", parse_dates = ['time:timestamp'])
df_test = pd.read_csv("bpi2017_test_filtered.csv", parse_dates= ['time:timestamp'])

df_train = df_train[:n]
df_val = df_val[:n]
df_test = df_test[:n]

df_train = df_train.sort_values(by = ['case:concept:name', 'time:timestamp']).reset_index(drop = True)
df_val = df_val.sort_values(by = ['case:concept:name', 'time:timestamp']).reset_index(drop = True)
df_test = df_test.sort_values(by = ['case:concept:name', 'time:timestamp']).reset_index(drop = True)

# Global Feature 1

df_train["case_occurrence_nr"] = df_train.groupby(['case:concept:name'])['time:timestamp'].cumcount().tolist()
df_val["case_occurrence_nr"] = df_val.groupby(['case:concept:name'])['time:timestamp'].cumcount().tolist()
df_test["case_occurrence_nr"] = df_test.groupby(['case:concept:name'])['time:timestamp'].cumcount().tolist()

min_max_scaler_occur = MinMaxScaler()
df_train['nor_case_occurrence_nr'] = min_max_scaler_occur.fit_transform(np.array(df_train['case_occurrence_nr']).reshape(-1, 1))
df_val['nor_case_occurrence_nr'] = min_max_scaler_occur.transform(np.array(df_val['case_occurrence_nr']).reshape(-1, 1))
df_test['nor_case_occurrence_nr'] = min_max_scaler_occur.transform(np.array(df_test['case_occurrence_nr']).reshape(-1, 1))

# Global Feature 2

def case_in_hr(df):
    df['date'] = df['time:timestamp'].dt.date
    df['hour'] = df['time:timestamp'].dt.hour
    df_1 = df.groupby(['date', 'hour']).count()[['case:concept:name']].reset_index()
    df_1.rename(columns = {'case:concept:name': 'case_nr_per_hr'}, inplace = True)
    df = pd.merge(df, df_1, on = ['date', 'hour'], how = "left")
    df.drop(columns = ['date', 'hour'], inplace = True)
    return df

df_train = case_in_hr(df_train)
df_val = case_in_hr(df_val)
df_test = case_in_hr(df_test)

min_max_scaler_case = MinMaxScaler()
df_train['nor_case_nr_per_hr'] = min_max_scaler_case.fit_transform(np.array(df_train['case_nr_per_hr']).reshape(-1, 1))
df_val['nor_case_nr_per_hr'] = min_max_scaler_case.transform(np.array(df_val['case_nr_per_hr']).reshape(-1, 1))
df_test['nor_case_nr_per_hr'] = min_max_scaler_case.transform(np.array(df_test['case_nr_per_hr']).reshape(-1, 1))

#  Normalize time difference so that the time difference's value is within 0 and 1
min_max_scaler = MinMaxScaler()
df_train['nor_future_time_diff'] = min_max_scaler.fit_transform(np.array(df_train['future_time_diff']).reshape(-1, 1))
# Use the range from training data on validation and test data
df_val['nor_future_time_diff'] = min_max_scaler.transform(np.array(df_val['future_time_diff']).reshape(-1, 1))
df_test['nor_future_time_diff'] = min_max_scaler.transform(np.array(df_test['future_time_diff']).reshape(-1, 1))

df_old = pd.read_csv("bpi2017_train_filtered.csv", parse_dates = ['time:timestamp'])
df_old = df_old[:n]

def onehot_now(df):
    # Extract categorical and numerical variables
    df_cat = df[['concept:name', 'lifecycle:transition', 'EventOrigin', 'Action']]
    df_num = df[['nor_time_since_last_event', 'nor_time_since_case_starts', 'nor_time_since_midnight', 'nor_time_since_week_start', 'position', 'nor_case_occurrence_nr', 'nor_case_nr_per_hr']]
    # Convert categorical variable columns to one-hot encoding (A large matrix with dummy variables is made)
    enc = OneHotEncoder(handle_unknown = 'ignore', sparse = False)
    enc.fit(df_old[['concept:name', 'lifecycle:transition', 'EventOrigin', 'Action']])
    transformed = enc.transform(df_cat)
    # Create a dataframe using the newly created matrix
    df_ohe = pd.DataFrame(transformed, columns = enc.get_feature_names())
    # Combine dummy dataframe with numerical dataframe
    df_ohe = pd.concat([df_ohe, df_num], axis = 1)
    return df_ohe

df_train_now = onehot_now(df_train)
df_val_now = onehot_now(df_val)
df_test_now = onehot_now(df_test)

enc = OneHotEncoder(handle_unknown = 'ignore', sparse = False)
enc.fit(df_old[['next:concept:name']])
df_train_next = enc.transform(df_train[['next:concept:name']])
df_val_next = enc.transform(df_val[['next:concept:name']])
df_test_next = enc.transform(df_test[['next:concept:name']])

# Source: https://towardsdatascience.com/how-to-reshape-data-and-do-regression-for-time-series-using-lstm-133dad96cd00

def lstm_data_transform(x_data, y_data_1, y_data_2, num_steps):
    # Reshape the feature array to (621131, 27, 1) so that it fulfills the format requirement of LSTM (Number Of Examples, Time Steps, Features Per Step)
    # Slide window approach to prevent throwing data away
    # Prepare the list for the transformed data
    X, y_1, y_2 = list(), list(), list()
    # Loop of the entire data set
    for i in range(x_data.shape[0]):
        # Compute a new (sliding window) index
        end = i + num_steps
        # If index is larger than the size of the dataset, we stop
        if end >= x_data.shape[0]:
            break
        # Get a sequence of data for x
        seq_X = x_data[i: end]
        # Get only the last element of the sequency for y
        seq_y_1 = y_data_1[end]
        seq_y_2 = y_data_2[end]
        # Append the list with sequencies
        X.append(seq_X)
        y_1.append(seq_y_1)
        y_2.append(seq_y_2)
    # Make final arrays
    x_array = np.array(X)
    y_array_1 = np.array(y_1)
    y_array_2 = np.array(y_2)
    return x_array, y_array_1, y_array_2

def zero_row(df, df_now, df_next, time_step):
    # Convert all required data from dataframe to numpy arrays
    case_lst = df['case:concept:name'].unique().tolist()
    x = df_now.to_numpy()
    y_1 = df_next
    y_2 = df[['nor_future_time_diff']].to_numpy()

    new_x = []
    new_y_1 = []
    new_y_2 = []

    for i in tqdm(case_lst):
        index_lst = df[df['case:concept:name'] == i].index
        # Create rows with just 0 at the beginning so that number of samples after sliding window matches the actual sample size, and no future data is used
        x_a = x[index_lst[0]: index_lst[-1] + 1, : ]
        y_1_a = y_1[index_lst[0]: index_lst[-1] + 1, : ]
        y_2_a = y_2[index_lst[0]: index_lst[-1] + 1, : ]
        x_0 = np.zeros((time_step, x_a.shape[1]), dtype = float)
        y_0_1 = np.zeros((time_step, y_1_a.shape[1]), dtype = float)
        y_0_2 = np.zeros((time_step, y_2_a.shape[1]), dtype = float)
        x_a = np.concatenate((x_0, x_a))
        y_1_a = np.concatenate((y_0_1, y_1_a))
        y_2_a = np.concatenate((y_0_2, y_2_a))
        x_a, y_1_a, y_2_a = lstm_data_transform(x_a, y_1_a, y_2_a, time_step)
        new_x.append(x_a)
        new_y_1.append(y_1_a)
        new_y_2.append(y_2_a)
    
    actual_x = []
    actual_y_1 = []
    actual_y_2 = []
    
    for i in new_x:
        for j in i:
            actual_x.append(j)
    for i in new_y_1:
        for j in i:
            actual_y_1.append(j)
    for i in new_y_2:
        for j in i:
            actual_y_2.append(j)

    return np.asarray(actual_x), np.asarray(actual_y_1), np.asarray(actual_y_2)

time_step = 3 # Your chosen batch-size/timestep

x_train, y_train_event, y_train_time = zero_row(df_train, df_train_now, df_train_next, time_step)
x_val, y_val_event, y_val_time = zero_row(df_val, df_val_now, df_val_next, time_step)
x_test, y_test_event, y_test_time = zero_row(df_test, df_test_now, df_test_next, time_step)

epoch_nr = 20

device_name = tf.test.gpu_device_name()

def train(link):
    with tf.device(device_name):
        if link == 'Nil':
            # build the model: 
            main_input = Input(shape = (time_step, df_train_now.shape[1]), name = 'main_input')

            # train a 2-layer LSTM with one shared layer
            l1 = LSTM(100, implementation = 2, kernel_initializer = 'glorot_uniform', return_sequences = True, dropout = 0.2)(main_input) # the shared layer
            b1 = BatchNormalization()(l1)
            l2_1 = LSTM(100, implementation = 2, kernel_initializer = 'glorot_uniform', return_sequences = False, dropout = 0.2)(b1) # the layer specialized in activity prediction
            b2_1 = BatchNormalization()(l2_1)
            l2_2 = LSTM(100, implementation = 2, kernel_initializer = 'glorot_uniform', return_sequences = False, dropout = 0.2)(b1) # the layer specialized in time prediction
            b2_2 = BatchNormalization()(l2_2)
            act_output = Dense(len(df_train['next:concept:name'].unique().tolist()), activation = 'softmax', kernel_initializer = 'glorot_uniform', name = 'act_output')(b2_1)
            time_output = Dense(1, kernel_initializer = 'glorot_uniform', name = 'time_output')(b2_2)  

            model = Model(inputs = [main_input], outputs = [act_output, time_output])

            opt = Nadam(learning_rate = 0.002, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-08, schedule_decay = 0.004, clipvalue = 3)

            # The loss used in model training is mean_squared_error because it is time prediction
            # The optimizer is Nadam
            model.compile(loss = {'act_output':'categorical_crossentropy', 'time_output': 'mae'}, optimizer = opt)
            return model
        else:
            model = load_model(link)

        # Save the best model
        early_stopping = EarlyStopping(monitor = 'val_loss', patience = 42)
        checkpoint_filepath = '/content/drive/MyDrive/Process Mining RNN/model/weights.{epoch:02d}.h5'
        model_checkpoint_callback = ModelCheckpoint(filepath = checkpoint_filepath, monitor = 'val_loss', mode = 'min', save_best_only = True)
        lr_reducer = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.5, patience = 10, verbose = 0, mode = 'auto', min_delta = 0.0001, cooldown = 0, min_lr = 0)

        # Fit the model with 20 epoches and batch size 64
        # Validation data is used here for evaluation during the training process
        model.fit(x_train, {'act_output': y_train_event, 'time_output': y_train_time}, validation_data = (x_val, {'act_output': y_val_event, 'time_output': y_val_time}), epochs = epoch_nr, batch_size = 128, callbacks = [early_stopping, model_checkpoint_callback, lr_reducer])
        return model

model = train('Nil')

# Make predictions
train_predict_event, train_predict_time = model.predict(x_train)
val_predict_event, val_predict_time = model.predict(x_val)
test_predict_event, test_predict_time = model.predict(x_test)

# Obtain event predictions from the highest probability of the label found, then find its label string name
train_pred_event_lst = enc.get_feature_names()[np.argmax(train_predict_event, axis = 1)]
train_pred_event_lst = [i.replace('x0_', '') for i in train_pred_event_lst]
val_pred_event_lst = enc.get_feature_names()[np.argmax(val_predict_event, axis = 1)]
val_pred_event_lst = [i.replace('x0_', '') for i in val_pred_event_lst]
test_pred_event_lst = enc.get_feature_names()[np.argmax(test_predict_event, axis = 1)]
test_pred_event_lst = [i.replace('x0_', '') for i in test_pred_event_lst]

df_train['RNN_next_event'] = train_pred_event_lst
df_val['RNN_next_event'] = val_pred_event_lst
df_test['RNN_next_event'] = test_pred_event_lst

# Invert time predictions from min-max scaling to their actual value
train_predict_time = min_max_scaler.inverse_transform(train_predict_time)
val_predict_time = min_max_scaler.inverse_transform(val_predict_time)
test_predict_time = min_max_scaler.inverse_transform(test_predict_time)

train_pred_time_lst = train_predict_time[: , 0].tolist()
val_pred_time_lst = val_predict_time[: , 0].tolist()
test_pred_time_lst = test_predict_time[: , 0].tolist()

df_train['RNN_time_diff'] = train_pred_time_lst
df_val['RNN_time_diff'] = val_pred_time_lst
df_test['RNN_time_diff'] = test_pred_time_lst

train_true = df_train['future_time_diff'].tolist()
val_true = df_val['future_time_diff'].tolist()
test_true = df_test['future_time_diff'].tolist()

train_rnn = df_train['RNN_time_diff'].tolist()
val_rnn = df_val['RNN_time_diff'].tolist()
test_rnn = df_test['RNN_time_diff'].tolist()

# rnn_acc =  str(round(metrics.accuracy_score(test_true, test_rnn), 2))
rnn_mae = str(round(metrics.mean_absolute_error(test_true, test_rnn), 2))
rnn_rmse = str(round(metrics.mean_squared_error(test_true, test_rnn, squared=False), 2))

# metrics_out.write("RNN Accuracy: " + rnn_acc + " %\n")
metrics_out.write("RNN MAE: " + rnn_mae + "\n")
metrics_out.write("RNN RMSE: "  + rnn_rmse + "\n")


print("RNN METRICS")
# print("RNN ACCURACY: " + rnn_acc + " %")
print("RNN MAE: " + rnn_mae)
print("RNN RMSE: " + rnn_rmse)

result_df["RNN Predictions"] = df_test['future_time_diff'].tolist()

time.sleep(5)
print("----- MM-PRED -----")


class LogFile:
    def __init__(
        self,
        filename,
        delim,
        header,
        rows,
        time_attr,
        trace_attr,
        activity_attr=None,
        values=None,
        integer_input=False,
        convert=True,
        k=1,
        dtype=None,
    ):
        self.filename = filename
        self.time = time_attr
        self.trace = trace_attr
        self.activity = activity_attr
        if values is not None:
            self.values = values
        else:
            self.values = {}
        self.numericalAttributes = set()
        self.categoricalAttributes = set()
        self.ignoreHistoryAttributes = set()
        if self.trace is None:
            self.k = 0
        else:
            self.k = k

        type = "str"
        if integer_input:
            type = "int"
        if filename is not None:
            n = 2000
            if dtype is not None:
                self.data = pd.read_csv(
                    self.filename,
                    header=header,
                    nrows=rows,
                    delimiter=delim,
                    encoding="latin-1",
                    dtype=dtype,
                )
                self.data = self.data[:n]

            else:
                self.data = pd.read_csv(
                    self.filename,
                    header=header,
                    nrows=rows,
                    delimiter=delim,
                    encoding="latin-1",
                )
                self.data = self.data[:n]

            # Determine types for all columns - numerical or categorical
            for col_type in self.data.dtypes.iteritems():
                if col_type[1] == "float64":
                    self.numericalAttributes.add(col_type[0])
                else:
                    self.categoricalAttributes.add(col_type[0])

            if convert:
                self.convert2int()

            self.contextdata = None

    def get_data(self):
        if self.contextdata is None:
            return self.data
        return self.contextdata

    def get_cases(self):
        return self.get_data().groupby([self.trace])

    def filter_case_length(self, min_length):
        cases = self.data.groupby([self.trace])
        filtered_cases = []
        for case in cases:
            if len(case[1]) > min_length:
                filtered_cases.append(case[1])
        self.data = pd.concat(filtered_cases, ignore_index=True)

    def convert2int(self):
        self.convert2ints("converted_ints.csv")

    def convert2ints(self, file_out):
        """
        Convert csv file with string values to csv file with integer values.
        (File/string operations more efficient than pandas operations)
        :param file_out: filename for newly created file
        :return: number of lines converted
        """
        self.data = self.data.apply(lambda x: self.convert_column2ints(x))
        self.data.to_csv(file_out, index=False)

    def convert_column2ints(self, x):
        def test(a, b):
            # Return all elements from a that are not in b, make use of the fact that both a and b are unique and sorted
            a_ix = 0
            b_ix = 0
            new_uniques = []
            while a_ix < len(a) and b_ix < len(b):
                if a[a_ix] < b[b_ix]:
                    new_uniques.append(a[a_ix])
                    a_ix += 1
                elif a[a_ix] > b[b_ix]:
                    b_ix += 1
                else:
                    a_ix += 1
                    b_ix += 1
            if a_ix < len(a):
                new_uniques.extend(a[a_ix:])
            return new_uniques

        if self.time is not None and x.name == self.time:
            return x

        print("PREPROCESSING: Converting", x.name)
        if x.name not in self.values:
            x = x.astype("str")
            self.values[x.name], y = np.unique(x, return_inverse=True)
            return y + 1
        else:
            x = x.astype("str")
            self.values[x.name] = np.append(
                self.values[x.name], test(np.unique(x), self.values[x.name])
            )

            print("PREPROCESSING: Substituting values with ints")
            xsorted = np.argsort(self.values[x.name])
            ypos = np.searchsorted(self.values[x.name][xsorted], x)
            indices = xsorted[ypos]

        return indices + 1

    def convert_string2int(self, column, value):
        if column not in self.values:
            return value
        vals = self.values[column]
        found = np.where(vals == value)
        if len(found[0]) == 0:
            return None
        else:
            return found[0][0] + 1

    def convert_int2string(self, column, int_val):
        if column not in self.values:
            return int_val
        return self.values[column][int_val - 1]

    def attributes(self):
        return self.data.columns

    def keep_attributes(self, keep_attrs):
        if self.time and self.time not in keep_attrs and self.time in self.data:
            keep_attrs.append(self.time)
        if self.trace and self.trace not in keep_attrs:
            keep_attrs.append(self.trace)
        self.data = self.data[keep_attrs]

    def remove_attributes(self, remove_attrs):
        """
        Remove attributes with the given prefixes from the data
        :param remove_attrs: a list of prefixes of attributes that should be removed from the data
        :return: None
        """
        remove = []
        for attr in self.data:
            for prefix in remove_attrs:
                if attr.startswith(prefix):
                    remove.append(attr)
                    break

        self.data = self.data.drop(remove, axis=1)

    def filter(self, filter_condition):
        self.data = self.data[eval(filter_condition)]

    def filter_copy(self, filter_condition):
        log_copy = copy.deepcopy(self)
        log_copy.data = self.data[eval(filter_condition)]
        return log_copy

    def get_column(self, attribute):
        return self.data[attribute]

    def get_labels(self, label):
        labels = {}
        if self.trace is None:
            for row in self.data.itertuples():
                labels[row.Index] = getattr(row, label)
        else:
            traces = self.data.groupby([self.trace])
            for trace in traces:
                labels[trace[0]] = getattr(trace[1].iloc[0], label)
        return labels

    def create_trace_attribute(self):
        print("Create trace attribute")
        with mp.Pool(mp.cpu_count()) as p:
            result = p.map(
                self.create_trace_attribute_case, self.data.groupby([self.trace])
            )
        self.data = pd.concat(result)
        self.categoricalAttributes.add("trace")

    def create_trace_attribute_case(self, case_tuple):
        trace = []
        case_data = pd.DataFrame()
        for row in case_tuple[1].iterrows():
            row_content = row[1]
            trace.append(row_content[self.activity])
            row_content["trace"] = str(trace)
            case_data = case_data.append(row_content)
        return case_data

    def create_k_context(self):
        """
        Create the k-context from the current LogFile
        :return: None
        """
        print("Create k-context:", self.k)

        if self.k == 0:
            self.contextdata = self.data

        if self.contextdata is None:
            with mp.Pool(mp.cpu_count()) as p:
                result = p.map(
                    self.create_k_context_trace, self.data.groupby([self.trace])
                )

            self.contextdata = pd.concat(result, ignore_index=True)

    def create_k_context_trace(self, trace):
        contextdata = pd.DataFrame()

        trace_data = trace[1]
        shift_data = trace_data.shift().fillna(0)
        shift_data.at[shift_data.first_valid_index(), self.trace] = trace[0]
        joined_trace = shift_data.join(trace_data, lsuffix="_Prev0")
        for i in range(1, self.k):
            shift_data = shift_data.shift().fillna(0)
            shift_data.at[shift_data.first_valid_index(), self.trace] = trace[0]
            joined_trace = shift_data.join(joined_trace, lsuffix="_Prev%i" % i)
        contextdata = contextdata.append(joined_trace, ignore_index=True)
        contextdata = contextdata.astype("int", errors="ignore")
        return contextdata

    def add_duration_to_k_context(self):
        """
        Add durations to the k-context, only calculates if k-context has been calculated
        :return:
        """
        if self.contextdata is None:
            return

        for i in range(self.k):
            self.contextdata["duration_%i" % (i)] = self.contextdata.apply(
                self.calc_duration, axis=1, args=(i,)
            )
            self.numericalAttributes.add("duration_%i" % (i))

    def calc_duration(self, row, k):
        if row[self.time + "_Prev%i" % (k)] != 0:
            startTime = parse(
                self.convert_int2string(
                    self.time, int(row[self.time + "_Prev%i" % (k)])
                )
            )
            endTime = parse(self.convert_int2string(self.time, int(row[self.time])))
            return (endTime - startTime).total_seconds()
        else:
            return 0

    def discretize(self, row, bins=25):
        if isinstance(bins, int):
            labels = [str(i) for i in range(1, bins + 1)]
        else:
            labels = [str(i) for i in range(1, len(bins))]
        if self.isNumericAttribute(row):
            self.numericalAttributes.remove(row)
            self.categoricalAttributes.add(row)
            self.contextdata[row], binned = pd.cut(
                self.contextdata[row], bins, retbins=True, labels=labels
            )

        return binned

    def isNumericAttribute(self, attribute):
        if attribute in self.numericalAttributes:
            return True
        else:
            for k in range(self.k):
                if attribute.replace("_Prev%i" % (k), "") in self.numericalAttributes:
                    return True
        return False

    def isCategoricalAttribute(self, attribute):
        if attribute in self.categoricalAttributes:
            return True
        else:
            for k in range(self.k):
                if attribute.replace("_Prev%i" % (k), "") in self.categoricalAttributes:
                    return True
        return False

    def add_end_events(self):
        cases = self.get_cases()
        print("Run end event map")
        with mp.Pool(mp.cpu_count()) as p:
            result = p.map(self.add_end_event_case, cases)

        print("Combine results")
        new_data = []
        for r in result:
            new_data.extend(r)

        self.data = pd.DataFrame.from_records(new_data)

    def add_end_event_case(self, case_obj):
        case_name, case = case_obj
        new_data = []
        for i in range(0, len(case)):
            new_data.append(case.iloc[i].to_dict())

        record = {}
        for col in self.data:
            if col == self.trace:
                record[col] = case_name
            elif col == self.time:
                record[col] = new_data[-1][self.time]
            else:
                record[col] = "end"
        new_data.append(record)
        return new_data

    def splitTrainTest(self, train_percentage, split_case=True, method="train-test"):
        import random

        train_percentage = train_percentage / 100.0

        if split_case:
            if method == "random":
                train_inds = random.sample(
                    range(self.contextdata.shape[0]),
                    k=round(self.contextdata.shape[0] * train_percentage),
                )
                test_inds = list(
                    set(range(self.contextdata.shape[0])).difference(set(train_inds))
                )
            elif method == "train-test":
                train_inds = np.arange(0, self.contextdata.shape[0] * train_percentage)
                test_inds = list(
                    set(range(self.contextdata.shape[0])).difference(set(train_inds))
                )
            else:
                test_inds = np.arange(
                    0, self.contextdata.shape[0] * (1 - train_percentage)
                )
                train_inds = list(
                    set(range(self.contextdata.shape[0])).difference(set(test_inds))
                )
        else:
            train_inds = []
            test_inds = []
            cases = self.contextdata[self.trace].unique()
            if method == "random":
                train_cases = random.sample(
                    list(cases), k=round(len(cases) * train_percentage)
                )
                test_cases = list(set(cases).difference(set(train_cases)))
            elif method == "train-test":
                train_cases = cases[: round(len(cases) * train_percentage)]
                test_cases = cases[round(len(cases) * train_percentage) :]
            else:
                train_cases = cases[round(len(cases) * (1 - train_percentage)) :]
                test_cases = cases[: round(len(cases) * (1 - train_percentage))]

            for train_case in train_cases:
                train_inds.extend(
                    list(
                        self.contextdata[
                            self.contextdata[self.trace] == train_case
                        ].index
                    )
                )
            for test_case in test_cases:
                test_inds.extend(
                    list(
                        self.contextdata[
                            self.contextdata[self.trace] == test_case
                        ].index
                    )
                )

        train = self.contextdata.loc[train_inds]
        test = self.contextdata.loc[test_inds]

        print("Train:", len(train_inds))
        print("Test:", len(test_inds))

        train_logfile = LogFile(
            None,
            None,
            None,
            None,
            self.time,
            self.trace,
            self.activity,
            self.values,
            False,
            False,
        )
        train_logfile.filename = self.filename
        train_logfile.values = self.values
        train_logfile.contextdata = train
        train_logfile.categoricalAttributes = self.categoricalAttributes
        train_logfile.numericalAttributes = self.numericalAttributes
        train_logfile.data = self.data.loc[train_inds]
        train_logfile.k = self.k

        test_logfile = LogFile(
            None,
            None,
            None,
            None,
            self.time,
            self.trace,
            self.activity,
            self.values,
            False,
            False,
        )
        test_logfile.filename = self.filename
        test_logfile.values = self.values
        test_logfile.contextdata = test
        test_logfile.categoricalAttributes = self.categoricalAttributes
        test_logfile.numericalAttributes = self.numericalAttributes
        test_logfile.data = self.data.loc[test_inds]
        test_logfile.k = self.k

        return train_logfile, test_logfile

    def split_days(self, date_format, num_days=1):

        self.contextdata["days"] = self.contextdata[self.time].map(
            lambda l: str(datetime.strptime(l, date_format).isocalendar()[:3])
        )
        days = {}
        for group_name, group in self.contextdata.groupby("days"):
            new_logfile = LogFile(
                None,
                None,
                None,
                None,
                self.time,
                self.trace,
                self.activity,
                self.values,
                False,
                False,
            )
            new_logfile.filename = self.filename
            new_logfile.values = self.values
            new_logfile.categoricalAttributes = self.categoricalAttributes
            new_logfile.numericalAttributes = self.numericalAttributes
            new_logfile.k = self.k
            new_logfile.contextdata = group.drop("days", axis=1)
            new_logfile.data = new_logfile.contextdata[self.attributes()]

            days[group_name] = {}
            days[group_name]["data"] = new_logfile
        return days

    def split_weeks(self, date_format, num_days=1):

        self.contextdata["year_week"] = self.contextdata[self.time].map(
            lambda l: str(datetime.strptime(l, date_format).isocalendar()[:2])
        )
        weeks = {}
        for group_name, group in self.contextdata.groupby("year_week"):
            new_logfile = LogFile(
                None,
                None,
                None,
                None,
                self.time,
                self.trace,
                self.activity,
                self.values,
                False,
                False,
            )
            new_logfile.filename = self.filename
            new_logfile.values = self.values
            new_logfile.categoricalAttributes = self.categoricalAttributes
            new_logfile.numericalAttributes = self.numericalAttributes
            new_logfile.k = self.k
            new_logfile.contextdata = group.drop("year_week", axis=1)
            new_logfile.data = new_logfile.contextdata[self.attributes()]

            year, week = eval(group_name)
            group_name = "%i/" % year
            if week < 10:
                group_name += "0"
            group_name += str(week)

            weeks[group_name] = {}
            weeks[group_name]["data"] = new_logfile
        return weeks

    def split_months(self, date_format, num_days=1):

        self.contextdata["month"] = self.contextdata[self.time].map(
            lambda l: str(datetime.strptime(l, date_format).strftime("%Y/%m"))
        )

        months = {}
        for group_name, group in self.contextdata.groupby("month"):
            new_logfile = LogFile(
                None,
                None,
                None,
                None,
                self.time,
                self.trace,
                self.activity,
                self.values,
                False,
                False,
            )
            new_logfile.filename = self.filename
            new_logfile.values = self.values
            new_logfile.categoricalAttributes = self.categoricalAttributes
            new_logfile.numericalAttributes = self.numericalAttributes
            new_logfile.k = self.k
            new_logfile.contextdata = group.drop("month", axis=1)
            new_logfile.data = new_logfile.contextdata[self.attributes()]

            months[group_name] = {}
            months[group_name]["data"] = new_logfile
        return months

    def split_date(self, date_format, year_week, from_week=None):

        self.contextdata["year_week"] = self.contextdata[self.time].map(
            lambda l: str(datetime.strptime(l, date_format).isocalendar()[:2])
        )

        if from_week:
            train = self.contextdata[
                (self.contextdata["year_week"] >= from_week)
                & (self.contextdata["year_week"] < year_week)
            ]
        else:
            train = self.contextdata[self.contextdata["year_week"] < year_week]
        test = self.contextdata[self.contextdata["year_week"] == year_week]

        train_logfile = LogFile(
            None,
            None,
            None,
            None,
            self.time,
            self.trace,
            self.activity,
            self.values,
            False,
            False,
        )
        train_logfile.filename = self.filename
        train_logfile.values = self.values
        train_logfile.contextdata = train
        train_logfile.categoricalAttributes = self.categoricalAttributes
        train_logfile.numericalAttributes = self.numericalAttributes
        train_logfile.data = train[self.attributes()]
        train_logfile.k = self.k

        test_logfile = LogFile(
            None,
            None,
            None,
            None,
            self.time,
            self.trace,
            self.activity,
            self.values,
            False,
            False,
        )
        test_logfile.filename = self.filename
        test_logfile.values = self.values
        test_logfile.contextdata = test
        test_logfile.categoricalAttributes = self.categoricalAttributes
        test_logfile.numericalAttributes = self.numericalAttributes
        test_logfile.data = test[self.attributes()]
        test_logfile.k = self.k

        return train_logfile, test_logfile

    def create_folds(self, k):
        result = []
        folds = np.array_split(np.arange(0, self.contextdata.shape[0]), k)
        for f in folds:
            fold_context = self.contextdata.loc[f]

            logfile = LogFile(
                None,
                None,
                None,
                None,
                self.time,
                self.trace,
                self.activity,
                self.values,
                False,
                False,
            )
            logfile.filename = self.filename
            logfile.values = self.values
            logfile.contextdata = fold_context
            logfile.categoricalAttributes = self.categoricalAttributes
            logfile.numericalAttributes = self.numericalAttributes
            logfile.data = self.data.loc[f]
            logfile.k = self.k
            result.append(logfile)
        return result

    def extend_data(self, log):
        train_logfile = LogFile(
            None,
            None,
            None,
            None,
            self.time,
            self.trace,
            self.activity,
            self.values,
            False,
            False,
        )
        train_logfile.filename = self.filename
        train_logfile.values = self.values
        train_logfile.contextdata = self.contextdata.append(log.contextdata)
        train_logfile.categoricalAttributes = self.categoricalAttributes
        train_logfile.numericalAttributes = self.numericalAttributes
        train_logfile.data = self.data.append(log.data)
        train_logfile.k = self.k
        return train_logfile

    def get_traces(self):
        return [list(case[1][self.activity]) for case in self.get_cases()]

    def get_follows_relations(self, window=None):
        return self.get_traces_follows_relations(self.get_traces(), window)

    def get_traces_follows_relations(self, traces, window):
        follow_counts = {}
        counts = {}
        for trace in traces:
            for i in range(len(trace)):
                act = trace[i]
                if act not in follow_counts:
                    follow_counts[act] = {}
                    counts[act] = 0
                counts[act] += 1

                stop_value = len(trace)
                if window:
                    stop_value = min(len(trace), i + window)

                for fol_act in set(trace[i + 1 : stop_value + 1]):
                    if fol_act not in follow_counts[act]:
                        follow_counts[act][fol_act] = 0
                    follow_counts[act][fol_act] += 1

        follows = {}
        for a in range(1, len(self.values[self.activity]) + 1):
            always = 0
            sometimes = 0
            if a in follow_counts:
                for b in follow_counts[a]:
                    if a != b:
                        if follow_counts[a][b] == counts[a]:
                            always += 1
                        else:
                            sometimes += 1
            never = len(self.values[self.activity]) - always - sometimes
            follows[a] = (always, sometimes, never)

        return follows, follow_counts

    def get_relation_entropy(self):
        follows, _ = self.get_follows_relations()
        full_entropy = []
        for act in range(1, len(self.values[self.activity]) + 1):
            RC = follows[act]
            p_a = RC[0] / len(self.values[self.activity])
            p_s = RC[1] / len(self.values[self.activity])
            p_n = RC[2] / len(self.values[self.activity])
            entropy = 0
            if p_a != 0:
                entropy -= p_a * math.log(p_a)
            if p_s != 0:
                entropy -= p_s * math.log(p_s)
            if p_n != 0:
                entropy -= p_n * math.log(p_n)
            full_entropy.append(entropy)
        return full_entropy

    def get_j_measure_trace(self, trace, window):
        _, follows = self.get_traces_follows_relations([trace], window)
        j_measure = []
        value_counts = {}
        for e in trace:
            if e not in value_counts:
                value_counts[e] = 0
            value_counts[e] += 1
        for act_1 in range(1, len(self.values[self.activity]) + 1):
            for act_2 in range(1, len(self.values[self.activity]) + 1):
                num_events = len(trace)
                if act_1 in follows and act_2 in follows[act_1]:
                    p_aFb = follows[act_1][act_2] / value_counts.get(act_1, 0)
                else:
                    p_aFb = 0

                if act_1 not in value_counts:
                    p_a = 0
                else:
                    p_a = value_counts.get(act_1, 0) / num_events

                if act_2 not in value_counts:
                    p_b = 0
                else:
                    p_b = value_counts.get(act_2, 0) / num_events

                j_value = 0
                if p_aFb != 0 and p_b != 0:
                    j_value += p_aFb * math.log(p_aFb / p_b, 2)

                if p_aFb != 1 and p_b != 1:
                    j_value += (1 - p_aFb) * math.log((1 - p_aFb) / (1 - p_b), 2)

                j_measure.append(p_a * j_value)

        return j_measure

    def get_j_measure(self, window=5):
        traces = self.get_traces()
        return [self.get_j_measure_trace(trace, window) for trace in traces]


def combine(logfiles):
    if len(logfiles) == 0:
        return None

    log = copy.deepcopy(logfiles[0])
    for i in range(1, len(logfiles)):
        log = log.extend_data(logfiles[i])
    return log


REPR_DIM = 100


class Modulator(Layer):
    def __init__(self, attr_idx, num_attrs, time, **kwargs):
        self.attr_idx = attr_idx
        self.num_attrs = num_attrs  # Number of extra attributes used in the modulator (other than the event)
        self.time_step = time

        super(Modulator, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name="Modulator_W",
            shape=(self.num_attrs + 1, (self.num_attrs + 2) * REPR_DIM),
            initializer="uniform",
            trainable=True,
        )
        self.b = self.add_weight(
            name="Modulator_b",
            shape=(self.num_attrs + 1, 1),
            initializer="zeros",
            trainable=True,
        )

        # super(Modulator, self).build(input_shape)
        self.built = True

    def call(self, x):
        # split input to different representation vectors
        representations = []
        for i in range(self.num_attrs + 1):
            representations.append(x[:, ((i + 1) * self.time_step) - 1, :])

        # Calculate z-vector
        tmp = []
        for elem_product in range(self.num_attrs + 1):
            if elem_product != self.attr_idx:
                tmp.append(
                    multiply(
                        representations[self.attr_idx],
                        representations[elem_product],
                        name="Modulator_repr_mult_" + str(elem_product),
                    )
                )
        for attr_idx in range(self.num_attrs + 1):
            tmp.append(representations[attr_idx])
        z = concat(tmp, axis=1, name="Modulator_concatz")
        # Calculate b-vectors

        b = tf.sigmoid(
            matmul(self.W, transpose(z), name="Modulator_matmulb") + self.b,
            name="Modulator_sigmoid",
        )

        # Use b-vectors to output
        tmp = transpose(
            multiply(
                b[0, :],
                transpose(
                    x[
                        :,
                        (self.attr_idx * self.time_step) : (
                            (self.attr_idx + 1) * self.time_step
                        ),
                        :,
                    ]
                ),
            ),
            name="Modulator_mult_0",
        )
        for i in range(1, self.num_attrs + 1):
            tmp = tmp + transpose(
                multiply(
                    b[i, :],
                    transpose(
                        x[:, (i * self.time_step) : ((i + 1) * self.time_step), :]
                    ),
                ),
                name="Modulator_mult_" + str(i),
            )

        return tmp

    def compute_output_shape(self, input_shape):
        return (None, self.time_step, REPR_DIM)

    def get_config(self):
        config = {
            "attr_idx": self.attr_idx,
            "num_attrs": self.num_attrs,
            "time": self.time_step,
        }
        base_config = super(Modulator, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def create_model_cudnn(vec, vocab_act_size, vocab_role_size, output_folder):
    # Create embeddings + Concat
    act_input = Input(shape=(vec["prefixes"]["x_ac_inp"].shape[1],), name="act_input")
    role_input = Input(shape=(vec["prefixes"]["x_rl_inp"].shape[1],), name="role_input")

    act_embedding = Embedding(
        vocab_act_size, 100, input_length=vec["prefixes"]["x_ac_inp"].shape[1],
    )(act_input)
    act_dropout = Dropout(0.2)(act_embedding)
    act_e_lstm_1 = LSTM(32, return_sequences=True)(act_dropout)
    act_e_lstm_2 = LSTM(100, return_sequences=True)(act_e_lstm_1)

    role_embedding = Embedding(
        vocab_role_size, 100, input_length=vec["prefixes"]["x_rl_inp"].shape[1],
    )(role_input)
    role_dropout = Dropout(0.2)(role_embedding)
    role_e_lstm_1 = LSTM(32, return_sequences=True)(role_dropout)
    role_e_lstm_2 = LSTM(100, return_sequences=True)(role_e_lstm_1)

    concat1 = Concatenate(axis=1)([act_e_lstm_2, role_e_lstm_2])
    normal = BatchNormalization()(concat1)

    act_modulator = Modulator(attr_idx=0, num_attrs=1)(normal)
    role_modulator = Modulator(attr_idx=1, num_attrs=1)(normal)

    # Use LSTM to decode events
    act_d_lstm_1 = LSTM(100, return_sequences=True)(act_modulator)
    act_d_lstm_2 = LSTM(32, return_sequences=False)(act_d_lstm_1)

    role_d_lstm_1 = LSTM(100, return_sequences=True)(role_modulator)
    role_d_lstm_2 = LSTM(32, return_sequences=False)(role_d_lstm_1)

    act_output = Dense(vocab_act_size, name="act_output", activation="softmax")(
        act_d_lstm_2
    )
    role_output = Dense(vocab_role_size, name="role_output", activation="softmax")(
        role_d_lstm_2
    )

    model = Model(inputs=[act_input, role_input], outputs=[act_output, role_output])

    opt = Nadam(
        lr=0.002,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-08,
        schedule_decay=0.004,
        clipvalue=3,
    )
    model.compile(
        loss={
            "act_output": "categorical_crossentropy",
            "role_output": "categorical_crossentropy",
        },
        optimizer=opt,
    )

    model.summary()

    output_file_path = os.path.join(
        output_folder, "model_rd_{epoch:03d}-{val_loss:.2f}.h5"
    )

    # Saving
    model_checkpoint = ModelCheckpoint(
        output_file_path,
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode="auto",
    )

    early_stopping = EarlyStopping(monitor="val_loss", patience=42)

    model.fit(
        {
            "act_input": vec["prefixes"]["x_ac_inp"],
            "role_input": vec["prefixes"]["x_rl_inp"],
        },
        {
            "act_output": vec["next_evt"]["y_ac_inp"],
            "role_output": vec["next_evt"]["y_rl_inp"],
        },
        validation_split=0.2,
        verbose=2,
        batch_size=5,
        callbacks=[early_stopping, model_checkpoint],
        epochs=200,
    )


def create_model(log, output_folder, epochs, early_stop):
    vec = vectorization(log)
    vocab_act_size = len(log.values["event"]) + 1
    vocab_role_size = len(log.values["role"]) + 1

    # Create embeddings + Concat
    act_input = Input(shape=(vec["prefixes"]["x_ac_inp"].shape[1],), name="act_input")
    role_input = Input(shape=(vec["prefixes"]["x_rl_inp"].shape[1],), name="role_input")

    act_embedding = Embedding(
        vocab_act_size, 100, input_length=vec["prefixes"]["x_ac_inp"].shape[1],
    )(act_input)
    act_dropout = Dropout(0.2)(act_embedding)
    act_e_lstm_1 = LSTM(32, return_sequences=True)(act_dropout)
    act_e_lstm_2 = LSTM(100, return_sequences=True)(act_e_lstm_1)

    role_embedding = Embedding(
        vocab_role_size, 100, input_length=vec["prefixes"]["x_rl_inp"].shape[1],
    )(role_input)
    role_dropout = Dropout(0.2)(role_embedding)
    role_e_lstm_1 = LSTM(32, return_sequences=True)(role_dropout)
    role_e_lstm_2 = LSTM(100, return_sequences=True)(role_e_lstm_1)

    concat1 = Concatenate(axis=1)([act_e_lstm_2, role_e_lstm_2])
    normal = BatchNormalization()(concat1)

    act_modulator = Modulator(attr_idx=0, num_attrs=1, time=log.k)(normal)
    role_modulator = Modulator(attr_idx=1, num_attrs=1, time=log.k)(normal)

    # Use LSTM to decode events
    act_d_lstm_1 = LSTM(100, return_sequences=True)(act_modulator)
    act_d_lstm_2 = LSTM(32, return_sequences=False)(act_d_lstm_1)

    role_d_lstm_1 = LSTM(100, return_sequences=True)(role_modulator)
    role_d_lstm_2 = LSTM(32, return_sequences=False)(role_d_lstm_1)

    act_output = Dense(vocab_act_size, name="act_output", activation="softmax")(
        act_d_lstm_2
    )
    role_output = Dense(vocab_role_size, name="role_output", activation="softmax")(
        role_d_lstm_2
    )

    model = Model(inputs=[act_input, role_input], outputs=[act_output, role_output])

    opt = Nadam(
        lr=0.002,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-08,
        schedule_decay=0.004,
        clipvalue=3,
    )
    model.compile(
        loss={
            "act_output": "categorical_crossentropy",
            "role_output": "categorical_crossentropy",
        },
        optimizer=opt,
    )

    model.summary()

    output_file_path = os.path.join(
        output_folder, "model_{epoch:03d}-{val_loss:.2f}.h5"
    )

    # Saving
    model_checkpoint = ModelCheckpoint(
        output_file_path,
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode="auto",
    )

    early_stopping = EarlyStopping(monitor="val_loss", patience=early_stop)

    model.fit(
        {
            "act_input": vec["prefixes"]["x_ac_inp"],
            "role_input": vec["prefixes"]["x_rl_inp"],
        },
        {
            "act_output": vec["next_evt"]["y_ac_inp"],
            "role_output": vec["next_evt"]["y_rl_inp"],
        },
        validation_split=0.2,
        verbose=2,
        batch_size=5,
        callbacks=[early_stopping, model_checkpoint],
        epochs=epochs,
    )
    return model


def predict_next(log, model):
    prefixes = create_pref_next(log)
    return _predict_next(model, prefixes)


def predict_suffix(model, data):
    prefixes = create_pref_suf(data.test_orig)
    prefixes = _predict_suffix(
        model,
        prefixes,
        100,
        data.logfile.convert_string2int(data.logfile.activity, "end"),
    )
    prefixes = dl_measure(prefixes)

    average_dl = np.sum([x["suffix_dl"] for x in prefixes]) / len(prefixes)

    print("Average DL:", average_dl)
    return average_dl


def vectorization(log):
    """Example function with types documented in the docstring.
    Args:
        log_df (dataframe): event log data.
        ac_index (dict): index of activities.
        rl_index (dict): index of roles.
    Returns:
        dict: Dictionary that contains all the LSTM inputs.
    """
    from tensorflow.keras.utils import to_categorical

    print("Start Vectorization")

    vec = {"prefixes": dict(), "next_evt": dict()}

    train_cases = log.get_cases()
    part_vect_map = partial(vect_map, prefix_size=log.k)
    with mp.Pool(mp.cpu_count()) as p:
        result = np.array(p.map(part_vect_map, train_cases))

    vec["prefixes"]["x_ac_inp"] = np.concatenate(result[:, 0])
    vec["prefixes"]["x_rl_inp"] = np.concatenate(result[:, 1])
    vec["next_evt"]["y_ac_inp"] = np.concatenate(result[:, 2])
    vec["next_evt"]["y_rl_inp"] = np.concatenate(result[:, 3])

    vec["next_evt"]["y_ac_inp"] = to_categorical(
        vec["next_evt"]["y_ac_inp"], num_classes=len(log.values["event"]) + 1
    )
    vec["next_evt"]["y_rl_inp"] = to_categorical(
        vec["next_evt"]["y_rl_inp"], num_classes=len(log.values["role"]) + 1
    )
    return vec


def map_case(x, log_df, case_attr):
    return log_df[log_df[case_attr] == x]


def vect_map(case, prefix_size):
    case_df = case[1]

    x_ac_inps = []
    x_rl_inps = []
    y_ac_inps = []
    y_rl_inps = []
    for row in case_df.iterrows():
        row = row[1]
        x_ac_inp = []
        x_rl_inp = []
        for i in range(prefix_size - 1, 0, -1):
            x_ac_inp.append(row["event_Prev%i" % i])
            x_rl_inp.append(row["role_Prev%i" % i])
        x_ac_inp.append(row["event_Prev0"])
        x_rl_inp.append(row["role_Prev0"])

        x_ac_inps.append(x_ac_inp)
        x_rl_inps.append(x_rl_inp)
        y_ac_inps.append(row["event"])
        y_rl_inps.append(row["role"])
    return [
        np.array(x_ac_inps),
        np.array(x_rl_inps),
        np.array(y_ac_inps),
        np.array(y_rl_inps),
    ]


def create_pref_next(log):
    """Extraction of prefixes and expected suffixes from event log.
    Args:
        df_test (dataframe): testing dataframe in pandas format.
        case_attr: name of attribute containing case ID
        activity_attr: name of attribute containing the activity
    Returns:
        list: list of prefixes and expected sufixes.
    """
    prefixes = []
    print(type(log))
    cases = log.get_cases()
    for case in cases:
        trace = case[1]

        for row in trace.iterrows():
            row = row[1]
            ac_pref = []
            rl_pref = []
            t_pref = []
            for i in range(log.k - 1, -1, -1):
                ac_pref.append(row["event_Prev%i" % i])
                rl_pref.append(row["role_Prev%i" % i])
                t_pref.append(0)
            prefixes.append(
                dict(
                    ac_pref=ac_pref,
                    ac_next=row["event"],
                    rl_pref=rl_pref,
                    rl_next=row["role"],
                    t_pref=t_pref,
                )
            )
    return prefixes


def create_pref_suf(log):
    prefixes = []
    cases = log.get_cases()
    for case in cases:
        trace = case[1]

        trace_ac = list(trace["event"])
        trace_rl = list(trace["role"])

        j = 0
        for row in trace.iterrows():
            row = row[1]
            ac_pref = []
            rl_pref = []
            t_pref = []
            for i in range(log.k - 1, -1, -1):
                ac_pref.append(row["event_Prev%i" % i])
                rl_pref.append(row["role_Prev%i" % i])
                t_pref.append(0)
            prefixes.append(
                dict(
                    ac_pref=ac_pref,
                    ac_suff=[x for x in trace_ac[j + 1 :]],
                    rl_pref=rl_pref,
                    rl_suff=[x for x in trace_rl[j + 1 :]],
                    t_pref=t_pref,
                )
            )
            j += 1
    return prefixes


def _predict_next(model, prefixes):
    """Generate business process suffixes using a keras trained model.
    Args:
        model (keras model): keras trained model.
        prefixes (list): list of prefixes.
    """
    # Generation of predictions
    results = []
    for prefix in tqdm(prefixes):
        # Activities and roles input shape(1,5)

        x_ac_ngram = np.array([prefix["ac_pref"]])
        x_rl_ngram = np.array([prefix["rl_pref"]])

        predictions = model.predict([x_ac_ngram, x_rl_ngram])

        pos = np.argmax(predictions[0][0])

        results.append(
            (
                prefix["ac_next"],
                pos,
                predictions[0][0][pos],
                predictions[0][0][int(prefix["ac_next"])],
            )
        )

    return results


def _predict_suffix(model, prefixes, max_trace_size, end):
    """Generate business process suffixes using a keras trained model.
    Args:
        model (keras model): keras trained model.
        prefixes (list): list of prefixes.
        max_trace_size: maximum length of a trace in the log
        end: value representing the END token
    """
    # Generation of predictions
    for prefix in prefixes:
        # Activities and roles input shape(1,5)
        x_ac_ngram = np.append(np.zeros(5), np.array(prefix["ac_pref"]), axis=0)[
            -5:
        ].reshape((1, 5))

        x_rl_ngram = np.append(np.zeros(5), np.array(prefix["rl_pref"]), axis=0)[
            -5:
        ].reshape((1, 5))

        ac_suf, rl_suf = list(), list()
        for _ in range(1, max_trace_size):
            predictions = model.predict([x_ac_ngram, x_rl_ngram])
            pos = np.argmax(predictions[0][0])
            pos1 = np.argmax(predictions[1][0])
            # Activities accuracy evaluation
            x_ac_ngram = np.append(x_ac_ngram, [[pos]], axis=1)
            x_ac_ngram = np.delete(x_ac_ngram, 0, 1)

            x_rl_ngram = np.append(x_rl_ngram, [[pos1]], axis=1)
            x_rl_ngram = np.delete(x_rl_ngram, 0, 1)

            # Stop if the next prediction is the end of the trace
            # otherwise until the defined max_size
            ac_suf.append(pos)
            rl_suf.append(pos1)

            if pos == end:
                break

        prefix["suff_pred"] = ac_suf
        prefix["rl_suff_pred"] = rl_suf
    return prefixes


def dl_measure(prefixes):
    """Demerau-Levinstain distance measurement.
    Args:
        prefixes (list): list with predicted and expected suffixes.
    Returns:
        list: list with measures added.
    """
    for prefix in prefixes:
        suff_log = str([x for x in prefix["suff"]])
        suff_pred = str([x for x in prefix["suff_pred"]])

        length = np.max([len(suff_log), len(suff_pred)])
        sim = jf.damerau_levenshtein_distance(suff_log, suff_pred)
        sim = 1 - (sim / length)
        prefix["suffix_dl"] = sim
    return prefixes


def train(logfile, train_log, model_folder):
    create_model(
        vectorization(
            train_log.data,
            train_log.trace,
            "event",
            num_classes=len(logfile.values[logfile.activity]) + 1,
        ),
        len(logfile.values[logfile.activity]) + 1,
        len(logfile.values["role"]) + 1,
        model_folder,
    )


def train(log, epochs=200, early_stop=42):
    return create_model(log, "tmp", epochs, early_stop)


def update(model, log):
    vec = vectorization(log)

    model.fit(
        {
            "act_input": vec["prefixes"]["x_ac_inp"],
            "role_input": vec["prefixes"]["x_rl_inp"],
        },
        {
            "act_output": vec["next_evt"]["y_ac_inp"],
            "role_output": vec["next_evt"]["y_rl_inp"],
        },
        validation_split=0.25,
        verbose=2,
        batch_size=5,
        epochs=10,
    )

    return model


def test(model, log):
    return predict_next(log, model)


data = "BPIC2012_FULL.csv"
case_attr = "case"
act_attr = "event"

logfile = LogFile(
    data, ",", 0, None, None, case_attr, activity_attr=act_attr, convert=False, k=10
)
logfile.convert2int()
logfile.filter_case_length(5)
logfile.create_k_context()
train_log, test_log = logfile.splitTrainTest(80, split_case=False, method="test-train")

model = train(train_log, epochs=100, early_stop=10)

acc = test(model, test_log)

# Accuracy
sum = 0
total = 0
for elem in acc:
    if elem[0] == elem[1]:
        sum += 1
    total += 1

print("MM-PRED METRICS")
print(f"ACCURACY: {round((sum / total)*100,2)} %")

metrics_out.write("MM-PRED Accuracy: " + str(round((sum / total) * 100, 2)) + "%\n")

# Precision
correct_predicted = {}
total_predicted = {}
total_value = {}

for elem in acc:
    expected_val = elem[0]
    predicted_val = elem[1]

    if predicted_val not in total_predicted:
        total_predicted[predicted_val] = 0
        total_predicted[predicted_val] += 1

    if expected_val not in total_value:
        total_value[expected_val] = 0
        total_value[expected_val] += 1

    if elem[0] == elem[1]:
        if predicted_val not in correct_predicted:
            correct_predicted[predicted_val] = 0
        correct_predicted[predicted_val] += 1

sum = 0
for val in total_predicted.keys():
    sum += total_value.get(val, 0) * (
        correct_predicted.get(val, 0) / total_predicted[val]
    )

print(f"PRECISION: {round((sum / len(acc))*100,2)} %")

# Recall
correct_predicted = {}
total_value = {}
predicted_vals = []

for elem in acc:
    expected_val = elem[0]
    predicted_val = elem[1]
    predicted_vals.append(elem[1])

    if expected_val not in total_value:
        total_value[expected_val] = 0
    total_value[expected_val] += 1

    if elem[0] == elem[1]:
        if predicted_val not in correct_predicted:
            correct_predicted[predicted_val] = 0
        correct_predicted[predicted_val] += 1

    sum = 0
    for val in total_value.keys():
        sum += total_value[val] * (correct_predicted.get(val, 0) / total_value[val])

print(f"RECALL: {round((sum / len(acc))*100,2)} %")

result_df["MM-PRED"] = pd.Series(predicted_vals)

metrics_out.close()

# Save the results
result_df.to_csv("tool_v3_predictions.csv")
