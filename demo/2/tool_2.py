# # Import modules
import numpy as np
import pandas as pd
import datetime as datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout
from keras.callbacks import ModelCheckpoint
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
import tensorflow as tf
import os
import math
from sklearn.metrics import r2_score, mean_squared_error


warnings.filterwarnings(action="ignore")

# Read the split dataset
# Available on github/data/splits
df_train = pd.read_csv("bpi2017_train.csv")
df_test = pd.read_csv("bpi2017_test.csv")
df_val = pd.read_csv("bpi2017_val.csv")

n = 2000
df_train = df_train[:n]
df_test = df_test[:n]
df_val = df_val[:n]

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


# ## Assign position


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

print(
    "RANDOM FOREST ACCURACY: ",
    metrics.accuracy_score(test["concept:name"], test["predicted_action"]),
)

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


# all_predictions = np.argmax(softmax(lr.decision_function(X_train_st[15].reshape(1,-1))[0]).round(1))
prediction = np.argmax(
    softmax(lr.decision_function(X_train_st[15].reshape(1, -1))[0]).round(1)
)
prediction


lr.predict_proba(X_train_st[:2]).round(3)

# evaluate the model (??)
lr.score(X_test_st, y_test)

# ## Add the predictions to the initial dataframe


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

print("LOGISTIC REGRESSION ACCURACY:")
print(metrics.accuracy_score(df_test["predicted_action"], df_test["true_action"]))

print("----- SVM REGRESSION ------")
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
import re
import datetime as datetime
import seaborn as sns

df_train = pd.read_csv("bpi2017_train.csv", parse_dates=["time:timestamp"])
df_test = pd.read_csv("bpi2017_test.csv", parse_dates=["time:timestamp"])
df_val = pd.read_csv("bpi2017_val.csv", parse_dates=["time:timestamp"])

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

# Remove obsolete columns
df_train = df_train.drop(["index", "Unnamed: 0"], axis=1)
df_val = df_val.drop(["index", "Unnamed: 0"], axis=1)
df_test = df_test.drop(["index", "Unnamed: 0"], axis=1)

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

# %%
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

print("SVM ACCURACY:")
print(regr.score(X_test, y_test))

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
mse = sum((compare_predictions["true"] - compare_predictions["predicted"]) ** 2) / n
print("SVM REGRESSION MSE:")
print(mse)

print("----- VOTING REGRESSION -----")
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.ensemble import GradientBoostingRegressor

df_train = orig_train
df_test = orig_test
df_val = orig_val

df_train = df_train.sort_values(
    by=["case:concept:name", "time:timestamp"]
).reset_index()
df_val = df_val.sort_values(by=["case:concept:name", "time:timestamp"]).reset_index()
df_test = df_test.sort_values(by=["case:concept:name", "time:timestamp"]).reset_index()

# Remove obsolete columns
df_train = df_train.drop(["index", "Unnamed: 0"], axis=1)
df_val = df_val.drop(["index", "Unnamed: 0"], axis=1)
df_test = df_test.drop(["index", "Unnamed: 0"], axis=1)

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

# vals, code_concept_name = pd.factorize(df_train_f['concept:name'])
# df_train_f['concept:name'] = vals

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

print("VOTING REGRESSION ACURACCY: ")
print(ereg.score(X_train, y_train))

result_df.to_csv("tool_2_predictions.csv")
