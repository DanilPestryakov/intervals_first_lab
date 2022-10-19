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
        if need_save:
            plt.savefig(f'./pictures/linear_regression_ch_{num}.png')
        plt.legend(frameon=False)
        plt.show()
        lsm_params.append((k, b))
    return lsm_params


if __name__ == "__main__":
    data = read_data()
    #draw_plot(data, True)
    out_coeff = get_out_coeff(data)
    print(f'R21 = [{min(out_coeff)}, {max(out_coeff)}]')
    #draw_intervals(data, True)
    lsm_p = get_linear_regression(data, False)

