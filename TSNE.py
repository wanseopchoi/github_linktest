import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import matplotlib
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D

font_path = "C:\Windows\Fonts\malgun.ttf"
font = matplotlib.font_manager.FontProperties(fname=font_path).get_name()
matplotlib.rc('font', family=font)

#삼겹살, 고등어 파일 (공청기), inf를 nan으로 바꾸고, nan 제거,
df = pd.read_csv('All_oilcode.csv', header=0 , parse_dates=True).drop(columns=['datetime']).replace([np.inf, -np.inf], np.nan).dropna()
print(df.shape[0])

X= df.loc[:, ['Ratio_NC0_5_0', 'Ratio_NC0_5_1', 'Ratio_NC0_5_2', 'Ratio_NC0_5_3',
              'Ratio_NC1_0_0','Ratio_NC1_0_1','Ratio_NC1_0_2','Ratio_NC1_0_3',
              'Ratio_NC2_5_0','Ratio_NC2_5_1','Ratio_NC2_5_2','Ratio_NC2_5_3',
              'Ratio_NC4_0_0','Ratio_NC4_0_1','Ratio_NC4_0_2','Ratio_NC4_0_3',
              'Ratio_NC10_0_0','Ratio_NC10_0_1','Ratio_NC10_0_2','Ratio_NC10_0_3',
              'Ratio_SGP40_0','Ratio_SGP40_1', 'Ratio_SGP40_2','Ratio_SGP40_3']
               ].values
y = df.loc[:, 'Cook'].values

#Scaling
# scaler = StandardScaler()
# scaler.fit(X)
# X_scaled = scaler.transform(X)
#
# plt.figure(figsize=(12, 4))
# plt.plot(X)
# plt.legend(['Ratio_NC0_5_0', 'Ratio_NC1_0_1', 'Ratio_NC2_5_2' ])

X_scaled = X[:25000] # 26500_삼겹살 고등어만 모을경우
y = y[:25000]
mask = X_scaled[:, 0] > 0.006
X_scaled = X_scaled[mask]
y = y[mask]

df_mask=df[mask]
print(df_mask.shape[0])

# t-SNE 적용
Data = pd.DataFrame(df_mask, columns= [
                'Ratio_NC0_5_0','Ratio_NC0_5_1','Ratio_NC0_5_2','Ratio_NC0_5_3',
                'Ratio_NC1_0_0','Ratio_NC1_0_1','Ratio_NC1_0_2','Ratio_NC1_0_3',
                'Ratio_NC2_5_0','Ratio_NC2_5_1','Ratio_NC2_5_2','Ratio_NC2_5_3',
                'Ratio_NC4_0_0','Ratio_NC4_0_1','Ratio_NC4_0_2','Ratio_NC4_0_3',
                'Ratio_NC10_0_0','Ratio_NC10_0_1','Ratio_NC10_0_2','Ratio_NC10_0_3',
               'Ratio_SGP40_0','Ratio_SGP40_1', 'Ratio_SGP40_2','Ratio_SGP40_3','Cook'])
print(Data)

train_df = df[['Ratio_NC0_5_0','Ratio_NC1_0_1','Ratio_NC2_5_2',  ]]  # 선택 인자들만 별도로.. 여러개 가능
# 2차원 t-SNE 임베딩
tsne_np = TSNE(n_components=2, n_iter=1000, random_state=42).fit_transform (train_df)  # 최적화 반복회수,
# numpy array -> DataFrame 변환
tsne_df=pd.DataFrame(tsne_np, columns= ['component 0', 'component 1'])
print(tsne_df)
# class target 정보 불러오기
tsne_df['Cook'] = df['Cook']
# target 별 분리
tsne_df_1 = tsne_df[tsne_df['Cook'] == 1]
tsne_df_3 = tsne_df[tsne_df['Cook'] == 3]
# target 별 시각화
plt.scatter(tsne_df_1['component 0'], tsne_df_1['component 1'], color = 'red', label = 'pork')
plt.scatter(tsne_df_3['component 0'], tsne_df_3['component 1'], color = 'yellow', label = 'fish')

plt.xlabel('component 0')
plt.ylabel('component 1')
plt.legend()
plt.show()

# # 3차원 t-SNE 임베딩
# tsne_np = TSNE(n_components = 3, n_iter=1000).fit_transform(train_df)
#
# # numpy array -> DataFrame 변환
# tsne_df = pd.DataFrame(tsne_np, columns = ['component 0', 'component 1', 'component 2'])
# print(tsne_df)
# # 3차원 그래프 세팅
# fig = plt.figure(figsize=(9, 6))
# ax = fig.add_subplot(111, projection='3d')
# # class target 정보 불러오기
# tsne_df['Cook'] = df['Cook']
# # target 별 분리
# tsne_df_1 = tsne_df[tsne_df['Cook'] == 1]
# tsne_df_3 = tsne_df[tsne_df['Cook'] == 3]
#
# print(tsne_df_1['component 0'].shape, tsne_df_1['component 1'].shape, tsne_df_1['component 2'].shape)
# # target 별 시각화
# plt.scatter(tsne_df_1['component 0'], tsne_df_1['component 1'], tsne_df_1['component 2'], color = 'red', label = 'pork')
# plt.scatter(tsne_df_3['component 0'], tsne_df_3['component 1'], tsne_df_3['component 2'], color = 'yellow', label = 'fish')
#
# ax.set_xlabel('component 0')
# ax.set_ylabel('component 1')
# ax.set_zlabel('component 2')
# ax.legend()
# plt.show()