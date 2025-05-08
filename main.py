from src.library import *
from src.constant_eda import *
from src.read_preprocess_eda import ReadPreprocessAnalyzeDataset3

pd.set_option('display.max_columns', None)

def eda_ds3(name_folder: str, num_classes: list):
    """
    Do EDA on dataset_3. Read and Check data integrity first, and visualize them
    Args:
        name_folder: folder name containing data and label csv files
        num_classes (list): number of classes for each target in list (ex) [6, 2, 0, 0]
    Returns: None
    """

    paths = Path(PATH_DATA) / name_folder
    paths_data = [csvfile for csvfile in paths.iterdir()
                  if (csvfile.is_file()) and (csvfile.suffix == '.csv') and (csvfile.name.split('_')[0] == 'data')]
    paths_label = [csvfile for csvfile in paths.iterdir()
                   if (csvfile.is_file()) and (csvfile.suffix == '.csv') and (csvfile.name.split('_')[0] == 'label')]

    rp_all = ReadPreprocessAnalyzeDataset3(
        columns_name=COLUMN_NAMES_HW2_39, columns_idx=COLUMN_IDX_HW2_39,  # 27 일반 컬럼, 38 grimm+ puri 3종 혼합 컬럼  39 신규 NC set
        paths_data=paths_data, paths_label=paths_label, num_classes=num_classes,
    )

    rp_all.read_and_eda(plot=False, savefig=False)  # plot True & False 로 label 합친 그림 확인 가능
    rp_all.read_and_preprocess_data(plot=True, savefig=False)


if __name__ == "__main__":
    folder_name = 'test_250429_homecook'
    num_classes = [6, 2, 2, 2]
    # folder_name = 'test_1_label'
    # label_steps = 1

    eda_ds3(folder_name, num_classes=num_classes)
