from src.library import *
from src.constant_eda import *
from src.visualize_data_eda import plot_single_data_only, plot_single_data_with_labels
from scipy.signal import savgol_filter


class ReadPreprocessAnalyzeDataset3:
    """Read all files and preprocess them at once for Dataset_3(w/4-steps label)"""

    def __init__(
            self,
            columns_name=None,  # column names
            columns_idx=np.r_[:],  # column index
            paths_data=None,  # path for data file
            paths_label=None,  # path for label file
            num_classes=None,  # the number of classes for each label levels ex. [5, 2, 2, 4]
            time_cutout=1,  # time for cutting out abnormal data from start. [min.]
            sg_windows_length=31,  # window length for sav-gol filter
            sg_poly_order=7,  # polynomial order for sav-gol filter
            sg_mode='nearest',  # mode for sav-gol filter
            sampling_period_in_seconds=15,  # sampling period [sec.]

    ):
        self.columns = columns_name
        self.paths_data = paths_data
        self.paths_label = paths_label
        self.df_all = None
        self.columns_idx = columns_idx

        self.time_cutout = timedelta(minutes=time_cutout)
        self.sg_windows_length = sg_windows_length
        self.sg_poly_order = sg_poly_order
        self.sg_mode = sg_mode
        self.sampling_period = sampling_period_in_seconds
        self.ulim_max_scaling = 3000  # upper limit for max-scaling EC sensor data

        self.num_classes = num_classes
        self.num_multi_label = 1 if type(self.num_classes) == int else len(
            self.num_classes)  # the number of levels of multi-label hierarchy
        self.name_col_label = None  # column names for labels
        self.label_with_time = None  # list of all ((time_start, time_end), label) information

        self.save_fig = False
        self.suffix_savefile = 'suffix'
        pd.set_option('display.max_columns', None)

    def read_and_eda(self, plot=False, savefig=False):
        """read each file as a whole in the list of files, execute EDA for each files"""
        # 1. read files
        # 2. view information related to the files
        # 3. plot raw data

        self.suffix_savefile = datetime.now().strftime("%y%m%d%H%M%S")

        for path_data in self.paths_data:
            print("="*50, f"\nReading a file : {path_data} ")
            path_label = path_data.parent / path_data.name.replace('data', 'label')

            df = self.read_file_single(path_data, display=False)

            print(df.info())
            print(df.describe())

            if plot:
                plot_single_data_only(df, title=path_data.stem, savefig=savefig, suffix=self.suffix_savefile)

            # df = self.add_labels(df, path_label=path_label, num_column_label=self.num_multi_label)
            # print(df.loc[:, self.name_col_label].info())

    def read_and_preprocess_data(self, plot=False, savefig=False):
        """
        Read each file as a whole in the list of files, pre-process and plot if requested
        Preprocess as follows: cut_out > smoothing(filtering) > scaling > sampling
        """

        self.suffix_savefile = datetime.now().strftime("%y%m%d%H%M%S")

        for path_data in self.paths_data:
            print(f"Reading a file : {path_data} ")
            path_label = path_data.parent / path_data.name.replace('data', 'label')

            df = self.read_file_single(path_data, display=False)
            df = self.cut_out(df)

            df[COLUMN_Grimm] = self.smoothing_data(df[COLUMN_Grimm])  # grimm 계측기 같이 포함된 data일 경우 사용
            df[COLUMN_Grimm] = self.scaling_data(df[COLUMN_Grimm], type='max')  # global max scaling # ->10은 오버

            df[COLUMN_Puri] = self.smoothing_data(df[COLUMN_Puri])
            df[COLUMN_Puri] = self.scaling_data(df[COLUMN_Puri], type='max_global')

            df[COLUMN_Puri_MEMS] = self.smoothing_data(df[COLUMN_Puri_MEMS])
            df[COLUMN_Puri_MEMS] = 1 - self.scaling_data(df[COLUMN_Puri_MEMS], type='max')

            df[COLUMN_MOX] = self.smoothing_data(df[COLUMN_MOX])
            df[COLUMN_MOX] = 1 - self.scaling_data(df[COLUMN_MOX], type='max')  # 초기 max  # 수정 max_global

            df[COLUMN_OPT] = self.smoothing_data(df[COLUMN_OPT])
            df[COLUMN_OPT] = self.scaling_data(df[COLUMN_OPT], type='min-max')  # global max scaling # 초기 max_limit_upper #수정 min-max -> 확인 용이

            df[COLUMN_EC] = self.smoothing_data(df[COLUMN_EC])
            df[COLUMN_EC] = self.scaling_data(df[COLUMN_EC], type='max')  # global max scaling # 초기 max_limit_upper #수정 max

            df[COLUMN_SGP] = self.smoothing_data(df[COLUMN_SGP])
            df[COLUMN_SGP] = 1 - self.scaling_data(df[COLUMN_SGP], type='max')  # 초기 m.ax # 수정 min-max

            df[COLUMN_PM] = self.smoothing_data(df[COLUMN_PM])
            df[COLUMN_PM] = self.scaling_data(df[COLUMN_PM], type='max')  # global max scaling # 초기 max_limit_upper #수정 min-max

            df[COLUMN_HT] = self.scaling_data(df[COLUMN_HT], type='max_global')

            df = self.sampling_data(df, by='interval')  # interval_min=SAMPLING_INTERVAL)

            df = self.add_labels(df, path_label=path_label, num_column_label=self.num_multi_label)
            option_label_adjust = 1 # 옵션 바꿔주는 코드
            df = self.adjust_labels(df, option=option_label_adjust)
            self.adjust_label_with_time(option=option_label_adjust)
            # df = self.simplify_labels(df)
            df['file_name'] = path_data.name

            if plot:
                plot_single_data_with_labels(df, self.label_with_time, title=path_data.stem, savefig=savefig,
                                             suffix=self.suffix_savefile)

            # concat into df_all
            if self.df_all is None:
                self.df_all = df.reset_index().copy(deep=True)
            else:
                self.df_all = pd.concat([self.df_all, df.reset_index()])

    def read_file_single(self, path, header=None, display=True):
        df = pd.read_csv(path, header=header, low_memory=False)
        df.set_index(pd.to_datetime(df.iloc[:, 0] + " " + df.iloc[:, 1]),
                     inplace=True)  # set datetime info as index
        df = df.iloc[:, self.columns_idx]
        df.columns = self.columns

        if display:
            df.info()
            print(df.describe())
        return df

    def cut_out(self, df):
        # cut-out abnormal data for the first 'time_cutout' minutes
        return df.loc[df.index > df.index[0] + self.time_cutout, :]

    def smoothing_data(self, df):
        return df.apply(lambda x: self.execute_savgol(x))

    def execute_savgol(self, data_column):
        return savgol_filter(data_column, window_length=self.sg_windows_length, polyorder=self.sg_poly_order,
                             mode=self.sg_mode)

    def scaling_data(self, df, type=None):
        if type == "min-max":
            df = df.apply(lambda x: (x - x.min()) / (x.max() - x.min()))
        if type == "max":
            df = df.apply(lambda x: x / x.max() if x.max() else 0)
            # df = df.apply(lambda x: x / (x.max() + 1e-9))  # avoid ZeroDivisionError
        if type == "max_global":
            df = df.apply(lambda x: x / df.max().max())
        if type == "max_limit_upper":
            upper_limit = self.ulim_max_scaling  # 각 센서 별 upper limit
            df = df.apply(lambda x: x / upper_limit)
        if type == "cum_avg_6h":
            # 6h interval data에 대한 cumulative average 값으로 calibration // 상용 공청기에 적용되는 방식
            pass
        return df

    def sampling_data(self, df, by='interval', rate=1):
        if by == 'interval':
            # Sampling data per minutes using resample
            df_sampled = df.resample(f'{self.sampling_period}s').first().dropna().copy()
            return df_sampled  # time_selecting 이후 resampling 할 경우 dropna() 필요
        elif by == 'rate':
            # Sampling at a constant rate --> change into ways of sampling data per minutes
            return df[::rate]

    def add_labels(self, df, path_label=None, num_column_label=1):
        """Add label columns to df"""

        df['BD'] = int(path_label.stem.split('_')[3].replace('BD', ''))
        # label_with_time : 구간 별 인식을 위해 label_with_time 필요
        self.read_situation_time(path_label, num_column_label)

        if num_column_label == 1:  # data with single column labels
            self.name_col_label = ['label']
            df[self.name_col_label] = -1
            for (ts, te), label in self.label_with_time:
                df.loc[ts:te, 'label'] = int(label)
            df = df[df.label != -1]
            # print(df.value_counts('label'))

        elif num_column_label > 1:  # data with multiple column labels
            self.name_col_label = ['label' + str(i) for i in range(1, num_column_label + 1)]
            df[self.name_col_label] = -1
            for (ts, te), labels in self.label_with_time:
                for idx, label in enumerate(labels):
                    df.loc[ts:te, self.name_col_label[idx]] = int(label)
        return df

    def select_sensors(self, df):
        return pd.concat([df[self.columns], df.loc[:, ['label', 'label']]], axis=1)

    def adjust_labels(self, df, option=None, MERGE_LABELS_6222=None ):
        """
        Modify labels in df from original label structure into simplified classes.
        User can select types how to adjust labels manually by 'option' arg.
        Args:
            df: dataframe with original labels
            option: types of label adjustment.
                - 0 for normal [6, 2, 2, 2] clf. Simplification details are described in MERGE_LABELS in constants.py
                - 1 for oil-vapor clf. ['no-action', 'cook_dining_oil', 'cook_dining_non_oil'], ['vent', 'not-vent']
        Returns: df with modified labels
        """
        if self.num_classes[1]:
            # Vent label adjustment. 01: single-window-vent, 02: (two-window) cross-vent
            df.loc[(df.label2 // 10) & (df.label2 % 10 == 0), 'label2'] = df.label2 // 10  # vent code 변경 시 적용 (앞자리 코드)
            df.loc[(df.label2 // 10) & (df.label2 % 10 == 1), 'label2'] = 0  # vent code 변경 시 적용 (뒷자리 코드)
            df.loc[(df.label2 // 10) & (df.label2 % 10 == 2), 'label2'] = df.label2 // 10  # v ent code 변경 시 적용 (뒷자리 코드)

        if not option:
            for idx, num_class in enumerate(self.num_classes):
                if num_class:
                    if idx == 0:
                        df.label1 = (df.label1 // 10).astype(int)  # label1 0, 10, 20, ... --> 0, 1, 2 변환
                    col = 'label' + str(idx + 1)
                    for label_col in df[col].unique():
                        df.loc[df[col] == label_col, col] = MERGE_LABELS_6222[col][label_col]
        elif option == 1:
            if self.num_classes[0] and self.num_classes[1]:   # Situation1, 2를 모두 사용하는 경우
                # not-vent 이외 모든 경우 label2 == 1
                # df.loc[(df.label2 != 0), 'label2'] = 1  # vent

                # cook, meal 중 구이, 튀김류 label1 == 1
                df.loc[(df.label1 < 30) & (df.label1 >= 10) & (
                        (df.label1 % 10 == 1) | (df.label1 % 10 == 4)), 'label1'] = 1
                # cook, meal 중 구이, 튀김류 외 전체 label1 == 2
                df.loc[(df.label1 < 30) & (df.label1 >= 10) & ~(
                        (df.label1 % 10 == 1) | (df.label1 % 10 == 4)), 'label1'] = 2
                # 그 밖에 모든 label1 == 'others' or 'no-action'
                df.loc[~df.label1.isin([0, 1, 2]), 'label1'] = 3

            elif self.num_classes[0] and not self.num_classes[1]:   # Situation1 만 사용하는 경우
                # cook, meal 중 구이, 튀김류 label1 == 1
                df.loc[(df.label1 < 30) & (df.label1 >= 10) & (
                        (df.label1 % 10 == 1) | (df.label1 % 10 == 4)), 'label1'] = 1
                # cook, meal 중 구이, 튀김류 외 전체 label1 == 2
                df.loc[(df.label1 < 30) & (df.label1 >= 10) & ~(
                        (df.label1 % 10 == 1) | (df.label1 % 10 == 4)), 'label1'] = 2
                # 그 밖에 모든 label1 == 'others' or 'no-action'
                df.loc[~df.label1.isin([0, 1, 2]), 'label1'] = 3

            elif not self.num_classes[0] and self.num_classes[1]:  # Situation2 만 사용하는 경우
                # not-vent 이외 모든 경우 label2 == 1
                # df.loc[(df.label2 != 0), 'label2'] = 1  # vent
                pass

            else:
                print("Any label doesn't exist under neither Situation 1 nor 2")

        return df

    def adjust_label_with_time(self, option=1):
        """
        Modify label_with_time((t_start, t_end), labels) tuple) in place to simplify classes.
        User can select types how to adjust labels manually by 'option' arg.
        Args:
            option: types of label adjustment.
                - 0 for normal [6, 2, 2, 2] clf. Simplification details are described in MERGE_LABELS in constants.py
                - 1 for oil-vapor clf. ['no-action', 'cook_dining_oil', 'cook_dining_non_oil'], ['vent', 'not-vent']
        Returns: None
        """
        for idx, ((ts, te), labels) in enumerate(self.label_with_time):
            labels = list(labels)
            # Vent label adjustment. 01: single-window-vent, 02: (two-window) cross-vent
            if (labels[1] // 10) and (labels[1] % 10 == 0):
                labels[1] = labels[1] // 10  # vent code 변경 시  -> 3 앞자리 변경시 적용
            elif (labels[1] // 10) and (labels[1] % 10 == 1):
                labels[1] = 0  # vent code 변경 시 -> 뒷자리 변경 시적용 (아래 코드와 같이 연계 스위치로)
            elif (labels[1] // 10) and (labels[1] % 10 == 2):
                labels[1] = labels[1] // 10  # vent code 변경 시 -> 뒷자리 변경 시적용 (위 코드와 같이 연계 스위치로)

            if not option:
                labels[0] = labels[0] // 10
                # adjust labels according to predefined rules in constants.py
                for jdx, key in enumerate(sorted(MERGE_LABELS_6222.keys())):
                    labels[jdx] = MERGE_LABELS_6222[key][jdx]

            elif option == 1:
                # not-vent 이외 모든 경우 label2 == 1
                # if labels[1] != 0:
                #     labels[1] = 1   # 환기 코드 5하고 같이 연관되는 코드 (안쓸거면 활성화 처리)

                # cook, meal 중 구이, 튀김류 label1 == 1
                if (labels[0] < 30) and (labels[0] >= 10) and (
                        (labels[0] % 10 == 1) or (labels[0] % 10 == 4)):
                    labels[0] = 1
                # cook, meal 중 구이, 튀김류 외 전체 label1 == 2
                if (labels[0] < 30) and (labels[0] >= 10) and not (
                        (labels[0] % 10 == 1) or (labels[0] % 10 == 4)):
                    labels[0] = 2
                # 그 밖에 모든 label1 == 'others' or 'no-action'
                if not ((labels[0] == 0) or (labels[0] == 1) or (labels[0] == 2)):
                    labels[0] = 3  # 설거지나 40, 50-unknown 등도 유증기에선 모두

            # final assignment with adjusted label values
            self.label_with_time[idx] = ((ts, te), (labels[0], labels[1], labels[2], labels[3]))

    def read_situation_time(self, file_path, num_column_label=1):
        if num_column_label == 1:  # data with single column labels
            df_info = pd.read_csv(file_path, header=0, encoding='cp949')
            # df_info['dt_s'] = pd.to_datetime(df_info.date_start + " " + df_info.time_start)
            # df_info['dt_e'] = pd.to_datetime(df_info.date_end + " " + df_info.time_end)
            ts_lst = pd.to_datetime(df_info.date_start + " " + df_info.time_start).to_list()
            te_lst = pd.to_datetime(df_info.date_end + " " + df_info.time_end).to_list()
            label_lst = df_info.iloc[:, 0].to_list()
            self.label_with_time = list(zip(zip(ts_lst, te_lst), label_lst))

        elif num_column_label > 1:  # data with multiple columns labels
            df_info = pd.read_csv(file_path, header=0, encoding='cp949')
            ts_lst = pd.to_datetime(df_info.date_start + " " + df_info.time_start).to_list()
            te_lst = pd.to_datetime(df_info.date_end + " " + df_info.time_end).to_list()
            name_col_label = ['situation' + str(i) for i in range(1, num_column_label + 1)]
            # df_info.situation1 = (df_info.situation1 // 10).astype(int)  # sit1 label 0, 10, 20, ... --> 0, 1, 2 변환
            label_lst = [[] for _ in range(num_column_label)]
            for idx, col in enumerate(name_col_label):
                label_lst[idx] = df_info.loc[:, col].to_list()
            self.label_with_time = list(zip(zip(ts_lst, te_lst), zip(*label_lst)))
        return None

    # def simplify_labels(self, df):
    #     MERGE_LABELS = {
    #         'label1': {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5},
    #         'label2': {0: 0, 1: 0, 2: 0, 3: 1, 4: 0, 5: 1, 6: 1, 7: 1},
    #         'label3': {0: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1},
    #         'label4': {0: 0, 1: 1, 2: 1, 3: 1},
    #     }
    #
    #     if self.num_multi_label == 1:
    #         pass
    #     elif self.num_multi_label > 1:
    #         for col in self.name_col_label:
    #             for label_col in df[col].unique():
    #                 df.loc[df[col] == label_col, col] = MERGE_LABELS[col][label_col]
    #     return df
