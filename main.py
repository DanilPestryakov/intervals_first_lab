import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt
from sklearn.linear_model import LinearRegression


def read_data():
    files_with_data = ['./data/Kanal_1_600nm_0_23mm.csv', './data/Kanal_2_600nm_0_23_mm.csv']

    data_rows = []

    for file in files_with_data:
        data_r = genfromtxt(file, delimiter=';')
        data_rows.append([val[0] for val in data_r])
    return data_rows


def draw_plot(data_list, need_save=True):
    x = np.arange(0, len(data_list[0]), 1, dtype=int)
    fig, ax = plt.subplots()
    for num, data_l in enumerate(data_list, start=1):
        ax.plot(x, data_l, label=f'ch_{num}')
    plt.xlabel("n")
    plt.ylabel("mV")
    plt.ylim(0.51, 0.57)
    ax.legend(frameon=False)
    ax.set_title('Data from experiment')
    if need_save:
        plt.savefig('./pictures/two_chanel.png')
    plt.show()


def get_out_coeff(data_list):
    out_coeff_l = []
    for i in range(len(data_list[0])):
        out_coeff_l.append(data_list[0][i] / data_list[1][i])
    return out_coeff_l


def draw_intervals(data_list, need_save=True):
    x = np.arange(0, len(data_list[0]), 1, dtype=int)
    for num, data_l in enumerate(data_list, start=1):
        plt.errorbar(x, data_l, yerr=10**(-4), marker='o', linestyle='none', ecolor='k', elinewidth=0.8, capsize=4,
                     capthick=1)
        plt.xlabel("n")
        plt.ylabel("mV")
        plt.title(f'Ch_{num} with intervals')
        if need_save:
            plt.savefig(f'./pictures/error_ch_{num}.png')
        plt.show()


def get_linear_regression(data_list, need_save=True):
    x = np.arange(0, len(data_list[0]), 1, dtype=int)
    x_r = np.array(x).reshape((-1, 1))
    lsm_params = []
    for num, data_l in enumerate(data_list, start=1):
        model = LinearRegression().fit(x_r, np.array(data_l))
        k = model.coef_
        b = model.intercept_
        plt.errorbar(x, data_l, yerr=10 ** (-4), marker='o', linestyle='none', ecolor='k', elinewidth=0.8, capsize=4,
                     capthick=1, label=f'intervals_ch_{num}')
        plt.plot(x, k * x + b, label=f'regression_ch_{num}')
        plt.xlabel("n")
        plt.ylabel("mV")
        plt.title(f'Ch_{num} with intervals')
        plt.legend(frameon=False)
        if need_save:
            plt.savefig(f'./pictures/linear_regression_ch_{num}.png')
        plt.show()
        lsm_params.append((k, b))
    return lsm_params


def get_w_histogram(data_list, lsm_p_list, need_save=True):
    eps = 10**(-4)
    w_list_arr = []
    for num, data_l in enumerate(data_list, start=1):
        w_list = []
        lsm_y = lsm_p_list[num - 1][0] * np.arange(0, len(data_l), 1, dtype=int) + lsm_p_list[num - 1][1]
        for x_num, x in enumerate(data_l, start=0):
            start = x - eps
            end = x + eps
            y = lsm_y[x_num]
            if start > y:
                w = (eps + (start - y)) / eps
                w_list.append(w)
            elif end < y:
                w = (eps + (y - end)) / eps
                w_list.append(w)
            else:
                w_list.append(1)
        w_list_arr.append(w_list)
        plt.hist(w_list, label=f'w_{num}')
        plt.xlabel(f'w_{num}')
        plt.legend(frameon=False)
        plt.title(f'Ch_{num} w histogram')
        if need_save:
            plt.savefig(f'./pictures/w_hist_ch_{num}.png')
        plt.show()
    return w_list_arr


def get_linear(data_list, w_list, lsm_p_list, need_save=True):
    data_len = len(data_list[0])
    x = np.arange(0, data_len, 1, dtype=int)
    eps = 10**(-4)
    yerr = [eps] * data_len
    linear_data = []
    for num, data_l in enumerate(data_list, start=0):
        for i in range(200):
            yerr[i] *= w_list[num][i]
        data_l = np.array(data_l)
        data_l -= x * lsm_p_list[num][0]
        linear_data.append(data_l)
        plt.errorbar(x, data_l, yerr=yerr, marker='o', linestyle='none', ecolor='k', elinewidth=0.8, capsize=4,
                     capthick=1, label='intervals')
        b = [lsm_p_list[num][1]] * data_len
        plt.plot(x, b, label=f'{b[0]}')
        plt.xlabel("n")
        plt.ylabel("mV")
        plt.title(f'Ch_{num + 1} linear')
        plt.legend(frameon=False)
        if need_save:
            plt.savefig(f'./pictures/liner_ch_{num + 1}.png')
        plt.show()
    return linear_data


def multi_jaccard_metric(interval_list):
    res_inter = interval_list[0]
    res_union = interval_list[0]
    for i in range(1, len(interval_list), 1):
        res_inter = [max(res_inter[0], interval_list[i][0]), min(res_inter[1], interval_list[i][1])]
        res_union = [min(res_union[0], interval_list[i][0]), max(res_union[1], interval_list[i][1])]
    return (res_inter[1] - res_inter[0]) / (res_union[1] - res_union[0])


def get_inner_r(data_list, w_list, R_out, need_save=True):
    step_count = 1000
    eps = 10**(-4)
    r_step = (R_out[1] - R_out[0]) / step_count
    start = R_out[0]
    jaccar_list = []
    while start <= R_out[1]:
        data_l = [[data_item - eps * w_list[0][num], data_item + eps * w_list[0][num]] for num, data_item in enumerate(list(data_list[0]))]
        data_l += [[start * (data_item - eps * w_list[1][num]), start * (data_item + eps * w_list[1][num])] for num, data_item in enumerate(list(data_list[1]))]
        jaccar_list.append((start, multi_jaccard_metric(data_l)))
        start += r_step
    plt.plot([R[0] for R in jaccar_list], [R[1] for R in jaccar_list], label='Jaccard metric by R')
    optimal = [(R[0], R[1]) for R in jaccar_list if R[1] >= 0]
    plt.plot(optimal[0][0], optimal[0][1], 'ro', label=f'minR={optimal[0][0]}')
    plt.plot(optimal[-1][0], optimal[-1][1], 'ro', label=f'maxR={optimal[-1][0]}')
    plt.xlabel("R")
    plt.ylabel("JK")
    plt.legend(frameon=False)
    plt.title(f'Jaccard metric')
    if need_save:
        plt.savefig(f'./pictures/Jaccard_metric.png')
    plt.show()


if __name__ == "__main__":
    data = read_data()
    draw_plot(data, True)
    out_coeff = get_out_coeff(data)
    R_outer = [min(out_coeff), max(out_coeff)]
    print(f'R21 = [{min(out_coeff)}, {max(out_coeff)}]')
    draw_intervals(data, True)
    lsm_p = get_linear_regression(data, True)
    w_l = get_w_histogram(data, lsm_p, True)
    lin_data = get_linear(data, w_l, lsm_p, need_save=True)
    get_inner_r(lin_data, w_l, R_outer)
