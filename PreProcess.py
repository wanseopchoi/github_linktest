# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 11:23:45 2024

@author: younjae.lee
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 13:50:17 2024

@author: younjae.lee
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


# from sklearn.preprocessing import MinMaxScaler
# from sklearn.preprocessing import StandardScaler
# import seaborn as sns
# import time
# from datetime import datetime
# from scipy import interpolate
# import scipy.interpolate as spi
# from scipy.signal import find_peaks
# from csaps import csaps
# from scipy.signal import butter, lfilter
# import statistics
# from scipy import signal
# from scipy.fft import fftshift
# from sklearn.cluster import KMeans


def FeatureRename(data, Name):
    """
        데이터프레임의 열 이름을 변경하는 함수
        :param data: 입력 데이터프레임
        :param Name: 새로운 열 이름 리스트
        :return: 열 이름이 변경된 데이터프레임
        """
    data.columns = Name

    return data


def CalBase(data):
    """
        데이터프레임의 기본 통계 값을 계산하는 함수
        :param data: 입력 데이터프레임
        :return: 기본 통계 값을 포함하는 시리즈
        """
    Base = data.mean(numeric_only=True)
    Base['PM10'] = data['PM10'].min(numeric_only=True)
    Base['SGP4x_3'] = data['SGP4x_3'].max(numeric_only=True)

    return Base


def RawDataInterpolation(RawData):
    TIndex = pd.to_datetime(RawData['datetime'])
    df = pd.DataFrame(RawData.values, index=TIndex, columns=RawData.columns)
    df = df.drop('datetime', axis=1)
    df = df.astype(float)

    rdf = df.resample(rule='S').mean()
    Int_df = rdf.interpolate(method='cubic')

    return Int_df


def EventDetection(label_data):
    Event = pd.DataFrame()
    for i in range(len(label_data)):
        if i > 0:
            print(label_data['label_4'][i - 1], label_data['label_4'][i])
            if label_data['label_4'][i - 1] != label_data['label_4'][i]:
                Event = pd.concat([Event, label_data.iloc[i, :]], axis=1)
    Event = Event.transpose()
    return Event


def MakeEvent_df(Event, Int_df):
    EIndex = pd.to_datetime(Event['datetime'])
    df = pd.DataFrame(Event.values, index=EIndex, columns=Event.columns)
    df = df.drop('datetime', axis=1)
    df = df.label_4.astype(float)

    StartTime = Int_df.index[0]
    EndTime = Int_df.index[len(Int_df) - 1]
    df.loc[StartTime] = 0
    df.loc[EndTime] = 0

    rdf = df.resample(rule='S').mean()
    rdf = rdf.fillna(method='pad')

    return rdf, StartTime, EndTime


def Cal_Sdata(RawData):
    data = RawData.copy()
    data['R_SGP40_0'] = 11.0 * np.exp(data['SGP4x_0'] / 2871.0)
    data['R_SGP40_1'] = 11.0 * np.exp(data['SGP4x_1'] / 2871.0)
    data['R_SGP40_2'] = 11.0 * np.exp(data['SGP4x_2'] / 2871.0)
    data['R_SGP40_3'] = 11.0 * np.exp(data['SGP4x_3'] / 2871.0)
    Base = []
    Base.append(data.R_SGP40_0.max(numeric_only=True))
    Base.append(data.R_SGP40_1.max(numeric_only=True))
    Base.append(data.R_SGP40_2.max(numeric_only=True))
    Base.append(data.R_SGP40_3.max(numeric_only=True))

    data['S_SGP40_0'] = data['R_SGP40_0'] / Base[0]
    data['S_SGP40_1'] = data['R_SGP40_1'] / Base[1]
    data['S_SGP40_2'] = data['R_SGP40_2'] / Base[2]
    data['S_SGP40_3'] = data['R_SGP40_3'] / Base[3]

    data['PM10'] = data['PM10'] / 100

    feature = ['datetime', 'PM10', 'S_SGP40_0', 'S_SGP40_1', 'S_SGP40_2', 'S_SGP40_3']
    data = data[feature]

    return data, Base


def add_Ratio(data):
    data['Ratio_SGP40_0'] = data['S_SGP40_0'] / data['PM10']
    data['Ratio_SGP40_1'] = data['S_SGP40_1'] / data['PM10']
    data['Ratio_SGP40_2'] = data['S_SGP40_2'] / data['PM10']
    data['Ratio_SGP40_3'] = data['S_SGP40_3'] / data['PM10']

    return data


def add_Diff(data):
    n = 60
    data['D_SGP40_0'] = data['S_SGP40_0'].diff(periods=n)
    data['D_SGP40_1'] = data['S_SGP40_1'].diff(periods=n)
    data['D_SGP40_2'] = data['S_SGP40_2'].diff(periods=n)
    data['D_SGP40_3'] = data['S_SGP40_3'].diff(periods=n)
    data['D_PM10'] = data['PM10'].diff(periods=n)
    data = data.dropna()

    return data


def Process(data, label):
    data['datetime'] = data.DATE + ' ' + data.TIME
    label['datetime'] = label.date_start + ' ' + label.time_start

    feature_d = ['datetime', ' P-PM10', ' unMEMS_SGP4x_Data0]0]', ' unMEMS_SGP4x_Data0]1]', ' unMEMS_SGP4x_Data0]2]',
                 ' unMEMS_SGP4x_Data0]3]']
    feature_l = ['datetime', 'time_end', 'label_4']
    RawData = data[feature_d]
    label_data = label[feature_l]

    RawData = FeatureRename(RawData, ['datetime', 'PM10', 'SGP4x_0', 'SGP4x_1', 'SGP4x_2', 'SGP4x_3'])
    SData, Base = Cal_Sdata(RawData)
    SData = add_Ratio(SData)
    SData = add_Diff(SData)

    Int_df = RawDataInterpolation(SData)
    Event = EventDetection(label_data)
    Event_df, StartTime, EndTime = MakeEvent_df(Event, Int_df)

    Int_df['class'] = Event_df
    Int_df.plot()
    plt.title(Event_df.index[0])

    return Int_df, Event_df


path = './All/'
file_list = os.listdir(path)
file_list_py = [file for file in file_list if file.endswith('.csv')]

RawData = []
Label = []

for i in file_list_py:
    print(i)
    fname = i.split('_')
    data = pd.read_csv(path + i, encoding='cp949')
    t = fname[0]
    d = fname[1].split('.')[0]
    if t == 'data':
        RawData.append(data)
    elif t == 'label':
        Label.append(data)

All_df = pd.DataFrame()
for i in range(len(RawData)):
    print(i)
    data = RawData[i]
    label = Label[i]

    Int_df, Event_df = Process(data, label)
    All_df = pd.concat([All_df, Int_df])

All_df.to_csv('All.csv')


