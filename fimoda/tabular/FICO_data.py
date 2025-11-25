#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Load and pre-process FICO Challenge dataset
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_and_process_FICO(path, normalize=False):
    # Read raw data
    data = pd.read_csv(path)

    # Separate and encode label
    y = data.pop('RiskPerformance')
    le = LabelEncoder()
    y = le.fit_transform(y)

    # Drop rows where all data is missing
    y = y[(data > -9).any(axis=1)]
    data = data.loc[(data > -9).any(axis=1)]

    # MSinceMostRecentDelq == -7 means at least 84 months since delinquency, set to 84
    # MSinceMostRecentDelq == -8 means no delinquency data, imputing with mean seems to yield best univariate fit
    MSinceMostRecentDelqMean = data.loc[data['MSinceMostRecentDelq']>=0, 'MSinceMostRecentDelq'].mean()
    data.loc[data['MSinceMostRecentDelq']==-7, 'MSinceMostRecentDelq'] = data['MSinceMostRecentDelq'].max() + 1
    data.loc[data['MSinceMostRecentDelq']==-8, 'MSinceMostRecentDelq'] = MSinceMostRecentDelqMean

    # MSinceMostRecentInqexcl7days == -7 means inquiry within last 7 days, set to 0
    # MSinceMostRecentInqexcl7days == -8 means more than 24 months since inquiry, set to 25
    data.loc[data['MSinceMostRecentInqexcl7days']==-7, 'MSinceMostRecentInqexcl7days'] = 0
    data.loc[data['MSinceMostRecentInqexcl7days']==-8, 'MSinceMostRecentInqexcl7days'] = data['MSinceMostRecentInqexcl7days'].max() + 1

    # "other" values in MaxDelq2PublicRecLast12M are actually 7 (current and never delinquent) based on MaxDelqEver
    data.loc[data['MaxDelq2PublicRecLast12M'] > 7, 'MaxDelq2PublicRecLast12M'] = 7

    # Impute remaining special values with mean
    # May not be best for NetFractionRevolvingBurden, NumRevolvingTradesWBalance, NumBank2NatlTradesWHighUtilization
    # because -8 seems to be worse than mean for these features
    data = data.replace(-8, data[data >= 0].mean())
    data = data.replace(-9, data[data >= 0].mean())

    if normalize:
        ss = StandardScaler()
        data = pd.DataFrame(ss.fit_transform(data), columns=data.columns)

    return data, y
