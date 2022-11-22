import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt


def read_data_with_intervals(files_with_data):
    data_rows = []
    eps = 1e-04

    for file in files_with_data:
        data_r = genfromtxt(file, delimiter=';')
        data_rows.append([[val[0] - eps, val[0] + eps] for val in data_r][1:201])
    return data_rows


def dif_drift_component(interval_d, drift_params, start_enums=None):
    if start_enums is None:
        start_enums = [0, 0]
    new_list = []
    for list_num, list_ in enumerate(interval_d):
        new_list_ = []
        for num, interval in enumerate(list_, start=start_enums[list_num]):
            new_list_.append([interval[0] - (num + 1) * drift_params[list_num][1],
                              interval[1] - (num + 1) * drift_params[list_num][0]])
        new_list.append(new_list_)
    return new_list


def multi_jaccard_metric(interval_list):
    res_inter = interval_list[0]
    res_union = interval_list[0]
    for i in range(1, len(interval_list), 1):
        res_inter = [max(res_inter[0], interval_list[i][0]), min(res_inter[1], interval_list[i][1])]
        res_union = [min(res_union[0], interval_list[i][0]), max(res_union[1], interval_list[i][1])]
    return (res_inter[1] - res_inter[0]) / (res_union[1] - res_union[0])


def get_inner_r(data_list, R_out, need_save=True, save_path=''):
    step_count = 1000
    r_step = (R_out[1] - R_out[0]) / step_count
    start = R_out[0]
    jaccar_list = []
    while start <= R_out[1]:
        data_l = data_list[0].copy()
        data_l += [[start * data_item[0], start * data_item[1]] for data_item in data_list[1]]
        jaccar_list.append((start, multi_jaccard_metric(data_l)))
        start += r_step
    plt.plot([R[0] for R in jaccar_list], [R[1] for R in jaccar_list], label='Jaccard metric by R')
    optimal = [(R[0], R[1]) for R in jaccar_list if R[1] >= 0]
    optimal_m = None
    if optimal:
        plt.plot(optimal[0][0], optimal[0][1], 'ro', label=f'minR={optimal[0][0]:.5f}')
        plt.plot(optimal[-1][0], optimal[-1][1], 'ro', label=f'maxR={optimal[-1][0]:.5f}')
        argmaxR = max(optimal, key=lambda opt: opt[1])
        plt.plot(argmaxR[0], argmaxR[1], 'go', label=f'optR=({argmaxR[0]:.5f}, {argmaxR[1]:.5f})')
        optimal_m = max(optimal, key=lambda opt: opt[1])
        print('optimal: ', max(optimal, key=lambda opt: opt[1]))
    plt.xlabel("R")
    plt.ylabel("JK")
    plt.legend(frameon=False)
    plt.title(f'Jaccard metric')
    if need_save:
        plt.savefig(f'{save_path}/Jaccard_metric.png')
    plt.show()
    return optimal_m


def draw_all_intervals(intervals, optimal_m, need_save=True, save_path=''):
    data_len = (len(intervals[0]))
    x = np.arange(1, data_len + 1, 1, dtype=int)
    y_err = [(interval[1] - interval[0]) / 2 for interval in intervals[0]]
    y = [interval[1] - y_err[num] for num, interval in enumerate(intervals[0])]
    plt.errorbar(x, y, yerr=y_err, ecolor='cyan', label='intervals_ch_1')
    y_err = [optimal_m * (interval[1] - interval[0]) / 2 for interval in intervals[1]]
    y = [optimal_m * interval[1] - y_err[num] for num, interval in enumerate(intervals[1])]
    plt.errorbar(x, y, yerr=y_err, ecolor='red', label='intervals_ch_2')
    plt.legend(frameon=False)
    plt.title(f'Intervals intersection')
    plt.xlabel("n")
    plt.ylabel("mV")
    if need_save:
        plt.savefig(f'{save_path}/weighted_intervals.png')
    plt.show()


if __name__ == "__main__":
    data_postfix = '800nm_0.23mm.csv'
    interval_data = read_data_with_intervals([f'./data/ch_1/Канал 1_{data_postfix}', f'./data/ch_2/Канал 2_{data_postfix}'])
    save_p = f'./pictures/intervals'
    intervals_regression_drift_params = [[3.4551e-06, 4.2070e-06], [5.1628e-06, 6.2094e-06]]
    intervals_w_o_drift = dif_drift_component(interval_data, intervals_regression_drift_params)
    opt_m = get_inner_r(intervals_w_o_drift, [0.9377, 0.9385], True, save_path=save_p)
    draw_all_intervals(intervals_w_o_drift, opt_m[0], True, save_path=save_p)


