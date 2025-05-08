from src.library import *
from src.constant_eda import *


def plot_single_data_only(df, title: str = None, savefig: bool = False, suffix: str = None):
    """
    plot single file data with label information for 2nd HW / field data (실환경 데이터)
    display label as horizontal lines with different y-values and colors

    Args:
        df: dataframe with label info
        title: plot figure title
        savefig: bool. whether to save the plot figure or not
        suffix: str. suffix added to the end of file name

    Returns: None
    """
    figsize = (20, 12)  # 그래프 사이즈 -컬럼 증가 시 y축 늘려야 함
    date_fmt = mdates.DateFormatter("%H:%M:%S")
    # col_lst = [COLUMN_PM, COLUMN_OPT + [COLUMN_EC[0]], COLUMN_EC[1:], COLUMN_MOX, COLUMN_SGP, COLUMN_HT]  # COLUMN_Grimm, COLUMN_Puri,
    # ylabel_lst = ['PM', 'CO2/NH3', 'EC', 'MOX', 'SGP', 'RH/TEMP']  # 'Grimm' Puri
    # gridspec = [1, 1, 1, 1.5, 1.5, 1]  # 그래프 개별 창 크기
    col_lst = [COLUMN_Puri, COLUMN_PM, COLUMN_OPT + [COLUMN_EC[0]], COLUMN_EC[1:], COLUMN_MOX, COLUMN_SGP,
               COLUMN_HT]  # COLUMN_Grimm,
    ylabel_lst = ['Puri', 'PM', 'CO2/NH3', 'EC', 'MOX', 'SGP', 'RH/TEMP']  # 'Grimm',
    gridspec = [1, 1, 1, 1, 1.5, 1.5, 1]  # 그래프 개별 창 크기 순서대로 #Grimm 추가하려면 앞에 1 넣을 것

    n_subplot = len(col_lst)
    fig, axs = plt.subplots(n_subplot, 1, figsize=figsize, sharex=True,
                            gridspec_kw={'height_ratios': gridspec})
    fig.suptitle(title, y=0.95)

    for idx, col in enumerate(col_lst):
        axs[idx].plot(df.loc[:, col], label=col)
        axs[idx].xaxis.set_major_formatter(date_fmt)
        axs[idx].legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize='x-small', ncol=2 if len(col) > 5 else 1)
        axs[idx].set_ylabel(ylabel_lst[idx])

    fig.align_ylabels(axs)

    if savefig:
        plt.savefig(Path(PATH_RESULT) / f"plot_{title}_{suffix}_raw.png")
        plt.close(fig)
        print(f"Saving the plot figure: plot_{title}_{suffix}_raw.png")

    plt.show()


def plot_single_data_with_labels(
        df, label_with_time_interval: list, title: str = None, savefig: bool = False, suffix: str = None
):
    """
    Plot single file data with labels for 2nd HW / field data (실환경 데이터).
    Display label as horizontal lines with different y-values and colors.
    This can cover multi-label dataset as well as single-label ones.

    Args:
        df: dataframe. data with label info
        label_with_time_interval: list. ((time_start, time_end), label) information
        title: str. plot figure title
        savefig: bool. whether to save the plot figure or not
        suffix: str. suffix added to the end of file name

    Returns: None
    """
    label_with_time_interval = [((t1, t2), ls[:-2]) for (t1, t2), ls in label_with_time_interval]  # del pet label temporarily -1이면 situ.4만 제거 -2면 situ3,4 제거
    date_fmt = mdates.DateFormatter("%H:%M:%S")
    # col_lst = [COLUMN_PM, COLUMN_OPT + [COLUMN_EC[0]], COLUMN_EC[1:], COLUMN_MOX, COLUMN_SGP, COLUMN_HT]  # COLUMN_Grimm, COLUMN_Puri,
    # ylabel_lst = ['PM', 'CO2/NH3', 'EC', 'MOX', 'SGP', 'RH/TEMP']  # 'Grimm', 'Puri',
    # n_col_data = len(col_lst)
    # gridspec = [1, 1, 1, 1.5, 1.5, 1]   # height ratio of subplots for each data
    col_lst = [COLUMN_Puri, COLUMN_PM, COLUMN_OPT + [COLUMN_EC[0]], COLUMN_EC[1:], COLUMN_MOX, COLUMN_SGP,
               COLUMN_HT]  # COLUMN_Grimm,
    ylabel_lst = ['Puri', 'PM', 'CO2/NH3', 'EC', 'MOX', 'SGP', 'RH/TEMP']  # 'Grimm',
    n_col_data = len(col_lst)
    gridspec = [1, 1, 1, 1, 1.5, 1.5, 1]  # height ratio of subplots for each data # 'Grimm' 추가하려면 앞에 1 넣을 것

    if type(label_with_time_interval[0][1]) == int:
        n_col_label = 1
        figsize = (20, 12)
        gridspec += [1.5]
    else:
        n_col_label = len(label_with_time_interval[0][1])
        figsize = (20, 12)
        gridspec += [1.2, 1.2, 0.5, 0.3][:n_col_label]  # n_col_label을 4개 이하로 사용할 경우도 포함 - # label 창 크기
        # gridspec += [1.0, 1.2, 1.2, 0.6][:n_col_label]  # n_col_label을 4개 이하로 사용할 경우도 포함

    n_subplot = len(col_lst) + n_col_label
    fig, axs = plt.subplots(
        n_subplot, 1, figsize=figsize, sharex='all', gridspec_kw={'height_ratios': gridspec}
    )
    fig.suptitle(title, y=0.95)

    if n_col_label == 1:
        for (ts, te), label in label_with_time_interval:
            axs[-1].axvline(x=ts, color='grey', linestyle=':', linewidth=1)
            axs[-1].hlines(y=label, xmin=ts, xmax=te, color=COLORS_PLOT[label], linestyles='-', linewidth=6)
        axs[-1].set_yticks(np.arange(0, 8))
        axs[-1].set_yticklabels(NAME_LABEL_1)
        axs[-1].grid(axis='y')

        for idx, col in enumerate(col_lst):
            axs[idx].plot(df.loc[:, col], label=col)
            axs[idx].xaxis.set_major_formatter(date_fmt)
            axs[idx].legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize='x-small')

            for (ts, te), label in label_with_time_interval:
                axs[idx].axvline(x=ts, color='k', linestyle=':', linewidth=1)
    else:
        for (ts, te), labels in label_with_time_interval:
            for idx, label in enumerate(labels):
                axs[idx + n_col_data].axvline(x=ts, color='grey', linestyle=':', linewidth=1)
                axs[idx + n_col_data].hlines(y=label, xmin=ts, xmax=te, color=COLORS_PLOT[label], linestyles='-', linewidth=6)
        for idx in range(n_col_label):
            axs[idx + n_col_data].set_yticks(np.arange(0, len(NAME_LABEL_4[idx])))
            axs[idx + n_col_data].set_yticklabels(NAME_LABEL_4[idx])
            axs[idx + n_col_data].grid(axis='y')
            axs[idx + n_col_data].set_ylabel('label'+str(idx+1))

        for idx, col in enumerate(col_lst):
            axs[idx].plot(df.loc[:, col], label=col)
            axs[idx].xaxis.set_major_formatter(date_fmt)

            axs[idx].legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize='x-small', ncol=2 if len(col) > 5 else 1)
            axs[idx].set_ylabel(ylabel_lst[idx])

            for (ts, te), label in label_with_time_interval:
                axs[idx].axvline(x=ts, color='k', linestyle=':', linewidth=1)

    fig.align_ylabels(axs)

    if savefig:
        plt.savefig(Path(PATH_RESULT) / f"plot_{title}_{suffix}_labeled.png")
        plt.close(fig)
        print(f"Saving the plot figure: plot_{title}_{suffix}_labeled.png")

    plt.show()
