#!/usr/bin/env python
# coding: utf-8

# # Some Notes: 
# 
# 
# *   The 4-tuple output of the model looks like this: (expected, predicted, prob of predicted, prob of expected). When expected == predicted a correct prediction was made.
# 
# *   For the renaming: Use ProM to convert the .xes file to .csv, by using ProM the case and event attributes are automatically resolved. The values for the case should start with Application, while the values for the event should start with a capital letter and an underscore. The resource column is manually named role instead. 
# 
# *   The k parameter is the size for the prefix to use (the amount of historical events that have to be used for predicting the next event. Padding is used when not enough events are present, should be the same behaviour as in other prediction papers). For selecting the k value: you have to test some settings. Depends on the data, and often also on the model. In this case, it was used to speed up training, and to allow running an update with only very limited number of training data.
# 
# * On the other hand, a suffix is the remaining sequence of events. 
# 
# *   The split_case parameter allows for cases to be split between train and test set. When set to False, cases will be kept together. This can however result in the unwanted result that some of the events in the test set occured before some events in the train set.
# 
# *   In the MM-Pred paper, we are absolutely not able to recreate their noted accuracies in Table 2. It is very likely that they used different attributes/features (but they failed to mention which ones, which could require some additional programming to incorporate these attributes/features in the model).
# 
# 
# 
# 
# 
# 
# 

# # Main breakdown of scripts (Adaptor, Model, Modulator)
# - **Adapter**: Adapter class to allow it to work with the Method class
# - **Model**: Contains the model itself (there is a version in there where I was experimenting with using CUDNN, so there might be some duplicate code (the basics of this code was taken from the implementation of Camargo et al. so there should be lots of similarities in the way I used some functions etc.
# - **Modulator**: contains the implementation of the custom layer, refer to the Tensorflow documentation for the meaning of all these functions etc

# Imports
import multiprocessing as mp
import os
from functools import partial
import jellyfish as jf
import numpy as np
from tensorflow.keras.layers import Layer
from tensorflow import multiply, sigmoid, concat, transpose, matmul
import copy
import multiprocessing as mp
import pandas as pd
from dateutil.parser import parse
import math
import random
from operator import itemgetter
import matplotlib.pyplot as plt
import networkx as nx
import scipy
from scipy.stats import pearsonr


# Utils
class LogFile:

    def __init__(self, filename, delim, header, rows, time_attr, trace_attr, activity_attr = None, values = None, integer_input = False, convert = True, k = 1, dtype=None):
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
            if dtype is not None:
                self.data = pd.read_csv(self.filename, header=header, nrows=rows, delimiter=delim, encoding='latin-1', dtype=dtype)
            else:
                self.data = pd.read_csv(self.filename, header=header, nrows=rows, delimiter=delim, encoding='latin-1')

            # Determine types for all columns - numerical or categorical
            for col_type in self.data.dtypes.iteritems():
                if col_type[1] == 'float64':
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
        self.convert2ints("../converted_ints.csv")

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

        if self.isNumericAttribute(x.name):
            return x

        if self.time is not None and x.name == self.time:
            return x

        print("PREPROCESSING: Converting", x.name)
        if x.name not in self.values:
            x = x.astype("str")
            self.values[x.name], y = np.unique(x, return_inverse=True)
            return y + 1
        else:
            x = x.astype("str")
            self.values[x.name] = np.append(self.values[x.name], test(np.unique(x), self.values[x.name]))

            print("PREPROCESSING: Substituting values with ints")
            xsorted = np.argsort(self.values[x.name])
            ypos = np.searchsorted(self.values[x.name][xsorted], x)
            indices = xsorted[ypos]

        return indices + 1

    def convert_string2int(self, column, value):
        if column not in self.values:
            return value
        vals = self.values[column]
        found = np.where(vals==value)
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
            result = p.map(self.create_trace_attribute_case, self.data.groupby([self.trace]))
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
            # result = map(self.create_k_context_trace, self.data.groupby([self.trace]))

            with mp.Pool(mp.cpu_count()) as p:
                result = p.map(self.create_k_context_trace, self.data.groupby([self.trace]))

            # result = map(self.create_k_context_trace, self.data.groupby([self.trace]))

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
            self.contextdata['duration_%i' %(i)] = self.contextdata.apply(self.calc_duration, axis=1, args=(i,))
            self.numericalAttributes.add("duration_%i" % (i))

    def calc_duration(self, row, k):
        if row[self.time + "_Prev%i" % (k)] != 0:
            startTime = parse(self.convert_int2string(self.time, int(row[self.time + "_Prev%i" % (k)])))
            endTime = parse(self.convert_int2string(self.time,int(row[self.time])))
            return (endTime - startTime).total_seconds()
        else:
            return 0

    def discretize(self,row, bins=25):
        if isinstance(bins, int):
            labels = [str(i) for i in range(1,bins+1)]
        else:
            labels = [str(i) for i in range(1,len(bins))]
        if self.isNumericAttribute(row):
            self.numericalAttributes.remove(row)
            self.categoricalAttributes.add(row)
            self.contextdata[row], binned = pd.cut(self.contextdata[row], bins, retbins=True, labels=labels)
            #self.contextdata[row] = self.contextdata[row].astype(str)
            #self.contextdata[row] = self.convert_column2ints(self.contextdata[row])
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
                train_inds = random.sample(range(self.contextdata.shape[0]), k=round(self.contextdata.shape[0] * train_percentage))
                test_inds = list(set(range(self.contextdata.shape[0])).difference(set(train_inds)))
            elif method == "train-test":
                train_inds = np.arange(0, self.contextdata.shape[0] * train_percentage)
                test_inds = list(set(range(self.contextdata.shape[0])).difference(set(train_inds)))
            else:
                test_inds = np.arange(0, self.contextdata.shape[0] * (1 - train_percentage))
                train_inds = list(set(range(self.contextdata.shape[0])).difference(set(test_inds)))
        else:
            train_inds = []
            test_inds = []
            cases = self.contextdata[self.trace].unique()
            if method == "random":
                train_cases = random.sample(list(cases), k=round(len(cases) * train_percentage))
                test_cases = list(set(cases).difference(set(train_cases)))
            elif method == "train-test":
                train_cases = cases[:round(len(cases) * train_percentage)]
                test_cases = cases[round(len(cases) * train_percentage):]
            else:
                train_cases = cases[round(len(cases) * (1 - train_percentage)):]
                test_cases = cases[:round(len(cases) * (1 - train_percentage))]

            for train_case in train_cases:
                train_inds.extend(list(self.contextdata[self.contextdata[self.trace] == train_case].index))
            for test_case in test_cases:
                test_inds.extend(list(self.contextdata[self.contextdata[self.trace] == test_case].index))

        train = self.contextdata.loc[train_inds]
        test = self.contextdata.loc[test_inds]

        print("Train:", len(train_inds))
        print("Test:", len(test_inds))

        train_logfile = LogFile(None, None, None, None, self.time, self.trace, self.activity, self.values, False, False)
        train_logfile.filename = self.filename
        train_logfile.values = self.values
        train_logfile.contextdata = train
        train_logfile.categoricalAttributes = self.categoricalAttributes
        train_logfile.numericalAttributes = self.numericalAttributes
        train_logfile.data = self.data.loc[train_inds]
        train_logfile.k = self.k

        test_logfile = LogFile(None, None, None, None, self.time, self.trace, self.activity, self.values, False, False)
        test_logfile.filename = self.filename
        test_logfile.values = self.values
        test_logfile.contextdata = test
        test_logfile.categoricalAttributes = self.categoricalAttributes
        test_logfile.numericalAttributes = self.numericalAttributes
        test_logfile.data = self.data.loc[test_inds]
        test_logfile.k = self.k

        return train_logfile, test_logfile


    def split_days(self, date_format, num_days=1):
        from datetime import datetime

        self.contextdata["days"] = self.contextdata[self.time].map(lambda l: str(datetime.strptime(l, date_format).isocalendar()[:3]))
        days = {}
        for group_name, group in self.contextdata.groupby("days"):
            new_logfile = LogFile(None, None, None, None, self.time, self.trace, self.activity, self.values, False, False)
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
        from datetime import datetime

        self.contextdata["year_week"] = self.contextdata[self.time].map(lambda l: str(datetime.strptime(l, date_format).isocalendar()[:2]))
        weeks = {}
        for group_name, group in self.contextdata.groupby("year_week"):
            new_logfile = LogFile(None, None, None, None, self.time, self.trace, self.activity, self.values, False, False)
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
        from datetime import datetime
        self.contextdata["month"] = self.contextdata[self.time].map(lambda l: str(datetime.strptime(l, date_format).strftime("%Y/%m")))

        months = {}
        for group_name, group in self.contextdata.groupby("month"):
            new_logfile = LogFile(None, None, None, None, self.time, self.trace, self.activity, self.values, False, False)
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
        from datetime import datetime

        self.contextdata["year_week"] = self.contextdata[self.time].map(lambda l: str(datetime.strptime(l, date_format).isocalendar()[:2]))

        if from_week:
            train = self.contextdata[(self.contextdata["year_week"] >= from_week) & (self.contextdata["year_week"] < year_week)]
        else:
            train = self.contextdata[self.contextdata["year_week"] < year_week]
        test = self.contextdata[self.contextdata["year_week"] == year_week]

        train_logfile = LogFile(None, None, None, None, self.time, self.trace, self.activity, self.values, False, False)
        train_logfile.filename = self.filename
        train_logfile.values = self.values
        train_logfile.contextdata = train
        train_logfile.categoricalAttributes = self.categoricalAttributes
        train_logfile.numericalAttributes = self.numericalAttributes
        train_logfile.data = train[self.attributes()]
        train_logfile.k = self.k

        test_logfile = LogFile(None, None, None, None, self.time, self.trace, self.activity, self.values, False, False)
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

            logfile = LogFile(None, None, None, None, self.time, self.trace, self.activity, self.values, False, False)
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
        train_logfile = LogFile(None, None, None, None, self.time, self.trace, self.activity, self.values, False, False)
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
                    stop_value = min(len(trace), i+window)

                for fol_act in set(trace[i+1:stop_value+1]):
                    if fol_act not in follow_counts[act]:
                        follow_counts[act][fol_act] = 0
                    follow_counts[act][fol_act] += 1


        follows = {}
        for a in range(1, len(self.values[self.activity])+1):
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
        for act in range(1, len(self.values[self.activity])+1):
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
        for act_1 in range(1, len(self.values[self.activity])+1):
            for act_2 in range(1, len(self.values[self.activity]) + 1):
                num_events = len(trace)
                if act_1 in follows and act_2 in follows[act_1]:
                    p_aFb = follows[act_1][act_2] / value_counts.get(act_1, 0)
                else:
                    p_aFb = 0

                if act_1 not in value_counts:
                    p_a = 0
                else:
                    p_a = value_counts.get(act_1, 0)/ num_events

                if act_2 not in value_counts:
                    p_b = 0
                else:
                    p_b = value_counts.get(act_2, 0) / num_events

                j_value = 0
                if p_aFb != 0 and p_b != 0:
                    j_value += p_aFb * math.log(p_aFb / p_b, 2)

                if p_aFb != 1 and p_b != 1:
                    j_value += (1-p_aFb) * math.log((1-p_aFb) / (1-p_b), 2)

                j_measure.append(p_a * j_value)

        return j_measure


    def get_j_measure(self, window=5):
        traces = self.get_traces()
        # return [np.mean(self.get_j_measure_trace(trace, window)) for trace in traces]
        return [self.get_j_measure_trace(trace, window) for trace in traces]
        # j_measures = np.asarray([self.get_j_measure_trace(trace, window) for trace in traces])
        # avg_j_measures = [np.mean(j_measures[:,i]) for i in range(len(j_measures[0]))]
        # return avg_j_measures


def combine(logfiles):
    if len(logfiles) == 0:
        return None

    log = copy.deepcopy(logfiles[0])
    for i in range(1, len(logfiles)):
        log = log.extend_data(logfiles[i])
    return log


# Modulator
REPR_DIM = 100

class Modulator(Layer):
    def __init__(self, attr_idx, num_attrs, time, **kwargs):
        self.attr_idx = attr_idx
        self.num_attrs = num_attrs  # Number of extra attributes used in the modulator (other than the event)
        self.time_step = time

        super(Modulator, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="Modulator_W", shape=(self.num_attrs+1, (self.num_attrs + 2) * REPR_DIM), initializer="uniform", trainable=True)
        self.b = self.add_weight(name="Modulator_b", shape=(self.num_attrs + 1, 1), initializer="zeros", trainable=True)

        #super(Modulator, self).build(input_shape)
        self.built = True

    def call(self, x):
        # split input to different representation vectors
        representations = []
        for i in range(self.num_attrs + 1):
            representations.append(x[:,((i + 1) * self.time_step) - 1,:])

        # Calculate z-vector
        tmp = []
        for elem_product in range(self.num_attrs + 1):
            if elem_product != self.attr_idx:
                tmp.append(multiply(representations[self.attr_idx],representations[elem_product], name="Modulator_repr_mult_" + str(elem_product)))
        for attr_idx in range(self.num_attrs + 1):
            tmp.append(representations[attr_idx])
        z = concat(tmp, axis=1, name="Modulator_concatz")
        # Calculate b-vectors
        b = sigmoid(matmul(self.W,transpose(z), name="Modulator_matmulb") + self.b, name="Modulator_sigmoid")

        # Use b-vectors to output
        tmp = transpose(multiply(b[0,:], transpose(x[:,(self.attr_idx * self.time_step):((self.attr_idx+1) * self.time_step),:])), name="Modulator_mult_0")
        for i in range(1, self.num_attrs + 1):
             tmp = tmp + transpose(multiply(b[i,:], transpose(x[:,(i * self.time_step):((i+1) * self.time_step),:])), name="Modulator_mult_" + str(i))

        return tmp

    def compute_output_shape(self, input_shape):
        return (None, self.time_step, REPR_DIM)

    def get_config(self):
        config = {'attr_idx': self.attr_idx, 'num_attrs': self.num_attrs, 'time': self.time_step}
        base_config = super(Modulator, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# Model
def create_model_cudnn(vec, vocab_act_size, vocab_role_size, output_folder):
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    from tensorflow.keras.layers import Input, Embedding, Dropout, Concatenate, LSTM, Dense, BatchNormalization
    from tensorflow.keras.models import Model, load_model
    from tensorflow.keras.optimizers import Nadam

    # Create embeddings + Concat
    act_input = Input(shape = (vec['prefixes']['x_ac_inp'].shape[1],), name="act_input")
    role_input = Input(shape = (vec['prefixes']['x_rl_inp'].shape[1],), name="role_input")

    act_embedding = Embedding(vocab_act_size, 100, input_length=vec['prefixes']['x_ac_inp'].shape[1],)(act_input)
    act_dropout = Dropout(0.2)(act_embedding)
    act_e_lstm_1 = LSTM(32, return_sequences=True)(act_dropout)
    act_e_lstm_2 = LSTM(100, return_sequences=True)(act_e_lstm_1)


    role_embedding = Embedding(vocab_role_size, 100, input_length=vec['prefixes']['x_rl_inp'].shape[1],)(role_input)
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

    act_output = Dense(vocab_act_size, name="act_output", activation='softmax')(act_d_lstm_2)
    role_output = Dense(vocab_role_size, name="role_output", activation="softmax")(role_d_lstm_2)

    model = Model(inputs=[act_input, role_input], outputs=[act_output, role_output])

    opt = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999,
                epsilon=1e-08, schedule_decay=0.004, clipvalue=3)
    model.compile(loss={'act_output': 'categorical_crossentropy', 'role_output': 'categorical_crossentropy'}, optimizer=opt)

    model.summary()

    output_file_path = os.path.join(output_folder, 'model_rd_{epoch:03d}-{val_loss:.2f}.h5')

    # Saving
    model_checkpoint = ModelCheckpoint(output_file_path,
                                       monitor='val_loss',
                                       verbose=1,
                                       save_best_only=True,
                                       save_weights_only=False,
                                       mode='auto')

    early_stopping = EarlyStopping(monitor='val_loss', patience=42)

    model.fit({'act_input':vec['prefixes']['x_ac_inp'],
               'role_input':vec['prefixes']['x_rl_inp']},
              {'act_output':vec['next_evt']['y_ac_inp'],
               'role_output':vec['next_evt']['y_rl_inp']},
              validation_split=0.2,
              verbose=2,
              batch_size=5,
              callbacks=[early_stopping, model_checkpoint],
              epochs=200)

def create_model(log, output_folder, epochs, early_stop):
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    from tensorflow.keras.layers import Input, Embedding, Dropout, Concatenate, LSTM, Dense, BatchNormalization
    from tensorflow.keras.models import Model, load_model
    from tensorflow.keras.optimizers import Nadam

    vec = vectorization(log)
    vocab_act_size = len(log.values["event"]) + 1
    vocab_role_size = len(log.values["role"]) + 1

    # Create embeddings + Concat
    act_input = Input(shape=(vec['prefixes']['x_ac_inp'].shape[1],), name="act_input")
    role_input = Input(shape=(vec['prefixes']['x_rl_inp'].shape[1],), name="role_input")

    act_embedding = Embedding(vocab_act_size, 100, input_length=vec['prefixes']['x_ac_inp'].shape[1],)(act_input)
    act_dropout = Dropout(0.2)(act_embedding)
    act_e_lstm_1 = LSTM(32, return_sequences=True)(act_dropout)
    act_e_lstm_2 = LSTM(100, return_sequences=True)(act_e_lstm_1)


    role_embedding = Embedding(vocab_role_size, 100, input_length=vec['prefixes']['x_rl_inp'].shape[1],)(role_input)
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

    act_output = Dense(vocab_act_size, name="act_output", activation='softmax')(act_d_lstm_2)
    role_output = Dense(vocab_role_size, name="role_output", activation="softmax")(role_d_lstm_2)

    model = Model(inputs=[act_input, role_input], outputs=[act_output, role_output])

    opt = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999,
                epsilon=1e-08, schedule_decay=0.004, clipvalue=3)
    model.compile(loss={'act_output': 'categorical_crossentropy', 'role_output': 'categorical_crossentropy'}, optimizer=opt)

    model.summary()

    output_file_path = os.path.join(output_folder, 'model_{epoch:03d}-{val_loss:.2f}.h5')

    # Saving
    model_checkpoint = ModelCheckpoint(output_file_path,
                                       monitor='val_loss',
                                       verbose=1,
                                       save_best_only=True,
                                       save_weights_only=False,
                                       mode='auto')

    early_stopping = EarlyStopping(monitor='val_loss', patience=early_stop)

    model.fit({'act_input':vec['prefixes']['x_ac_inp'],
               'role_input':vec['prefixes']['x_rl_inp']},
              {'act_output':vec['next_evt']['y_ac_inp'],
               'role_output':vec['next_evt']['y_rl_inp']},
              validation_split=0.2,
              verbose=2,
              batch_size=5,
              callbacks=[early_stopping, model_checkpoint],
              epochs=epochs)
    return model


def predict_next(log, model):
    prefixes = create_pref_next(log)
    return _predict_next(model, prefixes)


def predict_suffix(model, data):
    prefixes = create_pref_suf(data.test_orig)
    prefixes = _predict_suffix(model, prefixes, 100, data.logfile.convert_string2int(data.logfile.activity, "end"))
    prefixes = dl_measure(prefixes)

    average_dl = (np.sum([x['suffix_dl'] for x in prefixes]) / len(prefixes))

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

    vec = {'prefixes': dict(), 'next_evt': dict()}

    train_cases = log.get_cases()
    part_vect_map = partial(vect_map, prefix_size=log.k)
    with mp.Pool(mp.cpu_count()) as p:
        result = np.array(p.map(part_vect_map, train_cases))

    vec['prefixes']['x_ac_inp'] = np.concatenate(result[:, 0])
    vec['prefixes']['x_rl_inp'] = np.concatenate(result[:, 1])
    vec['next_evt']['y_ac_inp'] = np.concatenate(result[:, 2])
    vec['next_evt']['y_rl_inp'] = np.concatenate(result[:, 3])

    vec['next_evt']['y_ac_inp'] = to_categorical(vec['next_evt']['y_ac_inp'], num_classes=len(log.values["event"])+1)
    vec['next_evt']['y_rl_inp'] = to_categorical(vec['next_evt']['y_rl_inp'], num_classes=len(log.values["role"])+1)
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
    return [np.array(x_ac_inps), np.array(x_rl_inps), np.array(y_ac_inps), np.array(y_rl_inps)]


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
            prefixes.append(dict(ac_pref=ac_pref,
                                 ac_next=row["event"],
                                 rl_pref=rl_pref,
                                 rl_next=row["role"],
                                 t_pref=t_pref))
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
            prefixes.append(dict(ac_pref=ac_pref,
                                 ac_suff=[x for x in trace_ac[j + 1:]],
                                 rl_pref=rl_pref,
                                 rl_suff=[x for x in trace_rl[j + 1:]],
                                 t_pref=t_pref))
            j += 1
    return prefixes

def _predict_next(model, prefixes):
    from tqdm import tqdm 
    """Generate business process suffixes using a keras trained model.
    Args:
        model (keras model): keras trained model.
        prefixes (list): list of prefixes.
    """
    # Generation of predictions
    results = []
    for prefix in tqdm(prefixes):
        # Activities and roles input shape(1,5)

        x_ac_ngram = np.array([prefix['ac_pref']])
        x_rl_ngram = np.array([prefix['rl_pref']])

        predictions = model.predict([x_ac_ngram, x_rl_ngram])

        pos = np.argmax(predictions[0][0])
        # print(prefix['ac_next'])
        # print(pos)
        # print(predictions)

        results.append((prefix["ac_next"], pos, predictions[0][0][pos], predictions[0][0][int(prefix["ac_next"])]))

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
        x_ac_ngram = np.append(
            np.zeros(5),
            np.array(prefix['ac_pref']),
            axis=0)[-5:].reshape((1, 5))

        x_rl_ngram = np.append(
            np.zeros(5),
            np.array(prefix['rl_pref']),
            axis=0)[-5:].reshape((1, 5))

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

        prefix['suff_pred'] = ac_suf
        prefix['rl_suff_pred'] = rl_suf
    return prefixes


def dl_measure(prefixes):
    """Demerau-Levinstain distance measurement.
    Args:
        prefixes (list): list with predicted and expected suffixes.
    Returns:
        list: list with measures added.
    """
    for prefix in prefixes:
        suff_log = str([x for x in prefix['suff']])
        suff_pred = str([x for x in prefix['suff_pred']])

        length = np.max([len(suff_log), len(suff_pred)])
        sim = jf.damerau_levenshtein_distance(suff_log,
                                              suff_pred)
        sim = (1 - (sim / length))
        prefix['suffix_dl'] = sim
    return prefixes

def train(logfile, train_log, model_folder):
    create_model(vectorization(train_log.data, train_log.trace, "event", num_classes=len(logfile.values[logfile.activity]) + 1), len(logfile.values[logfile.activity]) + 1, len(logfile.values["role"]) + 1, model_folder)


# # Loading Data, Adaptor, and Model Training
df1 = pd.read_csv('../bpi_2017.csv')

df1 = df1.rename(columns = {"concept:name": 'event', "case:concept:name": 'case', "org:resource": 'role'})
df1 = df1.drop(columns=['Unnamed: 0'])

df1.to_csv('BPIC2017_FULL.csv', index=False)

def train(log, epochs=200, early_stop=42):
    return create_model(log, "tmp", epochs, early_stop)


def update(model, log):
    vec = vectorization(log)

    model.fit({'act_input':vec['prefixes']['x_ac_inp'],
               'role_input':vec['prefixes']['x_rl_inp']},
              {'act_output':vec['next_evt']['y_ac_inp'],
               'role_output':vec['next_evt']['y_rl_inp']},
              validation_split=0.2,
              verbose=2,
              batch_size=5,
              epochs=10)

    return model


def test(model, log):
    return predict_next(log, model)

data = '../BPIC2017_FULL.csv'
case_attr = "case"
act_attr = "event"

logfile = LogFile(data, ",", 0, None, None, case_attr,
                  activity_attr=act_attr, convert=False, k=10)
logfile.convert2int()
logfile.filter_case_length(5)
logfile.create_k_context()
train_log, test_log = logfile.splitTrainTest(80, split_case=False, method="test-train")

model = train(train_log, epochs=100, early_stop=10)

acc = test(model, test_log)
print(acc)
print(len(acc))

# Accuracy
sum = 0 
total = 0
for elem in acc: 
  if elem[0] == elem[1]:
    sum += 1
  total += 1

f'Accuracy: {sum / total}'

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
  sum += total_value.get(val, 0) * (correct_predicted.get(val, 0) / total_predicted[val])

f'Precision: {sum / len(acc)}'

# Recall
correct_predicted = {}
total_value = {}

for elem in acc:
  expected_val = elem[0]
  predicted_val = elem[1]

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

f'Recall: {sum / len(acc)}'