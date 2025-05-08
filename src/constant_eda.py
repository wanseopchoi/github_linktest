import numpy as np

PATH_DATA = "data"
PATH_RESULT = "result"

# data preprocess parameters for SG-filtering, sampling and cut-out
N_COL_INIT = 30
WINDOW_LENGTH = 31
POLY_ORDER = 7
SAMPLING_INTERVAL = 1   # MIN.
SAMPLING_RATE = 25   # 분당 1pts 수준 --> resample 사용 시 필요 없음
TIME_CUTOUT = 60   # min

COLORS_PLOT = (
    "red", "orange", 'yellow', 'green', 'blue',
    'indigo', 'purple', "sienna", "grey", "magenta",
    "navy", "olive", "green", "orchid", "plum",
    "turquoise",  "black", "blue", "brown", "yellowgreen",
    "silver", "teal", "darkorange", "darkblue",
)

NAME_LABEL_1 = ['absence', 'presence', 'vent', 'cook', 'meal', 'cleaning', 'pet', 'others']
NAME_LABEL_4 = [
    ['no_action', 'cook', 'meal', 'cleaning', 'others', 'unknown'],
    ['none', 'CL', 'AC', 'VT', 'CL-AC', 'CL-VT', 'AC-VT', 'CL-AC-VT'],
    ['0', '1', '2', '3', '4', '5'],
    ['none', 'pet_exist', 'pet_waste', 'pet_meal'],
]

MERGE_LABELS_6222 = {
    'label1': {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5},
    'label2': {0: 0, 1: 0, 2: 0, 3: 1, 4: 0, 5: 1, 6: 1, 7: 1},
    'label3': {0: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1},
    'label4': {0: 0, 1: 1, 2: 1, 3: 1},
}

# constants for column indexing
COLUMN_MOX = ['SC_' + f'{x:02}' for x in range(1, 10)]
COLUMN_OPT = ['CO2']
COLUMN_EC = ['EC_' + f'{x:02}' for x in range(1, 6)]
COLUMN_PM = ['PM_' + f'{x:02}' for x in range(1, 5)]
COLUMN_SGP = ['SGP_' + f'{x:02}' for x in range(1, 7)]
COLUMN_HT = ['TEMP', 'RH']
COLUMN_Grimm = ['Grimm_' + f'{x}' for x in range(1, 4)]  # grimm 계측기 같이 포함된 경우 사용
COLUMN_Puri = ['Puri_' + f'{x}' for x in range(1, 4)]  # Puri 표현 방식 정의 ->3가지
COLUMN_Puri_MEMS = ['PMEMS_' + f'{x:02}' for x in range(1, 7)]  # puri mems ->6가지

NAME_SC_HW2 = ['VOC', 'VOC', 'Tol', 'H2S', 'NH3', 'H2', 'CO', 'O3', 'CH4']
NAME_EC_HW2 = ['NH3', 'H2S', 'VOC', 'SO2', 'C2H4']
COLUMN_MOX = [x + '_' + name for (x, name) in zip(COLUMN_MOX, NAME_SC_HW2)]
COLUMN_EC = [x + '_' + name for (x, name) in zip(COLUMN_EC, NAME_EC_HW2)]
NAME_Grimm_HW2 = ['10', '2.5', '1.0']  # grimm 계측기 같이 포함된 경우 사용
NAME_Puri_HW2 = ['10', '2.5', '1.0']  # Puri 표현 방식 정의 ->3가지
COLUMN_Grimm = [x + '_' + name for (x, name) in zip(COLUMN_Grimm, NAME_Grimm_HW2)]  # grimm 포함된 경우 사용
COLUMN_Puri = [x + '_' + name for (x, name) in zip(COLUMN_Puri, NAME_Puri_HW2)]

# SEN01 SEN02 ... SEN9 // CO2 NH3 H2S VOC // TEMP RH // SGP01...SGP08 // TEMP HUMIDITY // SPS30 // SO2 C2H4
# number : 9 + 4 + 2 + 8 + 2 + 2 = 27
COLUMN_NAMES_HW2_27 = COLUMN_MOX + COLUMN_OPT + COLUMN_EC[:3] + COLUMN_HT + COLUMN_SGP + COLUMN_PM + COLUMN_EC[3:]
COLUMN_IDX_HW2_27 = np.r_[2:11, 11, 12, 15, 18, 21:23, 23:29, 33:37, 37, 40]
COLUMN_NAMES_HW2_25 = COLUMN_MOX + COLUMN_OPT + COLUMN_EC + COLUMN_SGP + COLUMN_PM + COLUMN_EC[3:]
COLUMN_IDX_HW2_25 = np.r_[2:11, 11, 12, 15, 18, 23:29, 33:37, 37, 40]
COLUMN_NAMES_HW2_30 = COLUMN_Grimm + COLUMN_MOX + COLUMN_OPT + COLUMN_EC[:3] + COLUMN_HT + COLUMN_SGP + COLUMN_PM + COLUMN_EC[3:]
COLUMN_IDX_HW2_30 = np.r_[2:5, 5:14, 14, 15, 18, 21, 24:26, 26:32, 36:40, 40, 43]  # 센서 순서 Grimm, MOX, OPT, EC, HT, SGP, PM, EC
COLUMN_NAMES_HW2_33 = COLUMN_Grimm + COLUMN_Puri + COLUMN_MOX + COLUMN_OPT + COLUMN_EC[:3] + COLUMN_HT + COLUMN_SGP + COLUMN_PM + COLUMN_EC[3:]  # + COLUMN_Puri
COLUMN_IDX_HW2_33 = np.r_[2:5, 5:8, 16:25, 25, 26, 29, 32, 35:37, 37:43, 47:51, 51, 54]  # 센서 컬럼 순서 Grimm, Puri, MOX~ // PMEMS, 10:16
COLUMN_NAMES_HW2_37 = COLUMN_Grimm + COLUMN_Puri + COLUMN_Puri_MEMS + COLUMN_MOX + COLUMN_OPT + COLUMN_EC[:3] + COLUMN_HT + COLUMN_SGP + COLUMN_PM + COLUMN_EC[3:]  # + COLUMN_Puri_MEMS
COLUMN_IDX_HW2_37 = np.r_[2:5, 5:8, 10:16, 16:25, 25, 26, 29, 32, 35:37, 37:43, 47:51, 51, 54]  # 센서 컬럼 순서 Grimm, Puri, MOX, OPT, EC, HT, SGP,PMEMS, PM, EC 5:8, 10:16
COLUMN_NAMES_HW2_38 = COLUMN_MOX + COLUMN_OPT + COLUMN_EC[:3] + COLUMN_HT + COLUMN_SGP + COLUMN_PM + COLUMN_EC[3:] + COLUMN_Grimm + COLUMN_Puri + COLUMN_Puri_MEMS
COLUMN_IDX_HW2_38 = np.r_[2:11, 11, 12, 15, 18, 21:23, 23:29, 33:37, 37, 40, 41:44, 44:47, 49:55]
COLUMN_NAMES_HW2_39 = COLUMN_MOX + COLUMN_OPT + COLUMN_EC[:3] + COLUMN_HT + COLUMN_SGP + COLUMN_PM + COLUMN_EC[3:] + COLUMN_Grimm + COLUMN_Puri + COLUMN_Puri_MEMS
COLUMN_IDX_HW2_39 = np.r_[2:11, 11, 12, 15, 18, 21:23, 23:29, 33:37, 37, 40, 74:77, 66:69, 60:66]