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


def FeatureRename(data, Name):
    data.columns = Name

    return data


def CalBase(data):
    Base = data.mean(numeric_only=True)
    Base['PM10'] = data.PM10.min(numeric_only=True)
    Base['SGP4x_3'] = data.SGP4x_3.max(numeric_only=True)

    return Base


def RawDataInterpolation(RawData):
    TIndex = pd.to_datetime(RawData['datetime'])
    df = pd.DataFrame(RawData.values, index=TIndex, columns=RawData.columns)
    df = df.drop('datetime', axis=1)
    df = df.astype(float)

    rdf = df.resample(rule='S').mean()
    # Int_df=rdf.interpolate('cubic')
    Int_df = rdf.interpolate()

    return Int_df


def EventDetection(label_data):
    Event = pd.DataFrame()
    for i in range(len(label_data)):
        if i > 0:
            print(label_data.label_4[i - 1], label_data.label_4[i])
            if label_data.label_4[i - 1] != label_data.label_4[i]:
                Event = pd.concat([Event, label_data.iloc[i, :]], axis=1)
    Event = Event.transpose()
    return Event


def MakeEvent_df(Event, Int_df):
    EIndex = pd.to_datetime(Event['datetime'])
    df = pd.DataFrame(Event.values, index=EIndex, columns=Event.columns)
    df = df.drop('datetime', axis=1)
    df = df.drop('time_end', axis=1)
    # df=df.label_4.astype(float)
    df = df.astype(float)

    StartTime = Int_df.index[0]
    EndTime = Int_df.index[len(Int_df) - 1]
    df.loc[StartTime] = 0
    df.loc[EndTime] = 0

    rdf = df.resample(rule='S').mean()
    rdf = rdf.fillna(method='pad')

    return rdf, StartTime, EndTime


def Cal_Sdata(RawData, FeatureList):
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

    data['Ratio_NC0_5_0'] = data['NC1_0'] / data[' NC0_5']
    data['Ratio_NC0_5_1'] = data[' NC2_5'] / data[' NC0_5']
    data['Ratio_NC0_5_2'] = data[' NC4_0'] / data[' NC0_5']
    data['Ratio_NC0_5_3'] = data[' NC10_0'] / data[' NC0_5']
    data['Ratio_NC1_0_0'] = data[' NC0_5'] / data['NC1_0']
    data['Ratio_NC1_0_1'] = data[' NC2_5'] / data['NC1_0']
    data['Ratio_NC1_0_2'] = data[' NC4_0'] / data['NC1_0']
    data['Ratio_NC1_0_3'] = data[' NC10_0'] / data['NC1_0']
    data['Ratio_NC2_5_0'] = data[' NC0_5'] / data[' NC2_5']
    data['Ratio_NC2_5_1'] = data['NC1_0'] / data[' NC2_5']
    data['Ratio_NC2_5_2'] = data[' NC4_0'] / data[' NC2_5']
    data['Ratio_NC2_5_3'] = data[' NC10_0'] / data[' NC2_5']
    data['Ratio_NC4_0_0'] = data[' NC0_5'] / data[' NC4_0']
    data['Ratio_NC4_0_1'] = data['NC1_0'] / data[' NC4_0']
    data['Ratio_NC4_0_2'] = data[' NC2_5'] / data[' NC4_0']
    data['Ratio_NC4_0_3'] = data[' NC10_0'] / data[' NC4_0']
    data['Ratio_NC10_0_0'] = data[' NC0_5'] / data[' NC10_0']
    data['Ratio_NC10_0_1'] = data['NC1_0'] / data[' NC10_0']
    data['Ratio_NC10_0_2'] = data[' NC2_5'] / data[' NC10_0']
    data['Ratio_NC10_0_3'] = data[' NC4_0'] / data[' NC10_0']

    feature = ['datetime', 'rPM10', 'rPM25', 'rPM1', 'PM10', 'PM25', 'PM1', 'OPT_PM01', 'OPT_PM02', 'OPT_PM04',
               'OPT_PM10', 'Temp', 'Humi',
               'S_SGP40_0', 'S_SGP40_1', 'S_SGP40_2', 'S_SGP40_3', 'OPT_NCu5', 'OPT_NC01', 'OPT_NC02', 'OPT_NC04',
               'OPT_NC10',
               ' NC0_5', 'NC1_0', ' NC2_5', ' NC4_0', ' NC10_0', 'final_output_tVOC']
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


def add_Ratio_NC(data):
    data['Ratio_NC0_5_0'] = data['NC1_0'] / data[' NC0_5']
    data['Ratio_NC0_5_1'] = data[' NC2_5'] / data[' NC0_5']
    data['Ratio_NC0_5_2'] = data[' NC4_0'] / data[' NC0_5']
    data['Ratio_NC0_5_3'] = data[' NC10_0'] / data[' NC0_5']
    data['Ratio_NC1_0_0'] = data[' NC0_5'] / data['NC1_0']
    data['Ratio_NC1_0_1'] = data[' NC2_5'] / data['NC1_0']
    data['Ratio_NC1_0_2'] = data[' NC4_0'] / data['NC1_0']
    data['Ratio_NC1_0_3'] = data[' NC10_0'] / data['NC1_0']
    data['Ratio_NC2_5_0'] = data[' NC0_5'] / data[' NC2_5']
    data['Ratio_NC2_5_1'] = data['NC1_0'] / data[' NC2_5']
    data['Ratio_NC2_5_2'] = data[' NC4_0'] / data[' NC2_5']
    data['Ratio_NC2_5_3'] = data[' NC10_0'] / data[' NC2_5']
    data['Ratio_NC4_0_0'] = data[' NC0_5'] / data[' NC4_0']
    data['Ratio_NC4_0_1'] = data['NC1_0'] / data[' NC4_0']
    data['Ratio_NC4_0_2'] = data[' NC2_5'] / data[' NC4_0']
    data['Ratio_NC4_0_3'] = data[' NC10_0'] / data[' NC4_0']
    data['Ratio_NC10_0_0'] = data[' NC0_5'] / data[' NC10_0']
    data['Ratio_NC10_0_1'] = data['NC1_0'] / data[' NC10_0']
    data['Ratio_NC10_0_2'] = data[' NC2_5'] / data[' NC10_0']
    data['Ratio_NC10_0_3'] = data[' NC4_0'] / data[' NC10_0']

    return data


def Process(data, label):
    data['datetime'] = data.DATE + ' ' + data.TIME
    label['datetime'] = label.date_start + ' ' + label.time_start

    feature_d = ['datetime', 'PM10 [ug/m3]', 'PM2.5 [ug/m3]', 'PM1 [ug/m3]', ' P-PM10', ' P-PM2.5', ' P-PM1.0'
        , 'OPT_PM01', 'OPT_PM02', 'OPT_PM04', 'OPT_PM10', ' fTemperature', ' fHumid',
                 ' unMEMS_SGP4x_Data0]0]', ' unMEMS_SGP4x_Data0]1]', ' unMEMS_SGP4x_Data0]2]', ' unMEMS_SGP4x_Data0]3]',
                 'OPT_NCu5', 'OPT_NC01', 'OPT_NC02', 'OPT_NC04', 'OPT_NC10', ' NC0_5', 'NC1_0', ' NC2_5', ' NC4_0',
                 ' NC10_0',
                 'final_output_tVOC']
    feature_l = ['datetime', 'time_end', 'label_4', 'cooks_code', 'vent']
    RawData = data[feature_d]
    label_data = label[feature_l]

    FeatureList = ['datetime', 'rPM10', 'rPM25', 'rPM1', 'PM10', 'PM25', 'PM1',
                   'OPT_PM01', 'OPT_PM02', 'OPT_PM04', 'OPT_PM10', 'Temp', 'Humi',
                   'SGP4x_0', 'SGP4x_1', 'SGP4x_2', 'SGP4x_3', 'OPT_NCu5', 'OPT_NC01', 'OPT_NC02', 'OPT_NC04',
                   'OPT_NC10',
                   ' NC0_5', 'NC1_0', ' NC2_5', ' NC4_0', ' NC10_0', 'final_output_tVOC']
    RawData = FeatureRename(RawData, FeatureList)
    SData, Base = Cal_Sdata(RawData, FeatureList)
    SData = add_Ratio(SData)
    # SData=add_Diff(SData)
    SData = add_Ratio_NC(SData)

    Int_df = RawDataInterpolation(SData)
    Event = EventDetection(label_data)
    Event_df, StartTime, EndTime = MakeEvent_df(Event, Int_df)

    Int_df['label_4'] = Event_df['label_4']
    Int_df['Cook'] = Event_df['cooks_code']
    Int_df['Vent'] = Event_df['vent']

    Int_df.plot()
    plt.title(Event_df.index[0])

    return Int_df, Event_df, Event


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

    Int_df, Event_df, Event = Process(data, label)
    All_df = pd.concat([All_df, Int_df])

filtered_All_df = All_df[(All_df['label_4'] == 1) & (All_df['Cook'].isin([1, 3]))]  # 유증기 요리만 추출, 삼겹살1, 고등어3
# filtered_All_df = All_df[(All_df['label_4'] == 3) & (All_df['Cook'].isin([42,43,45,47,48]))]  # 유증기 요리만 추출, 삼겹살과 고등어만
print(filtered_All_df)

filtered_All_df.to_csv('All_oilcode.csv')

# All_df.to_csv('All_10.csv')
