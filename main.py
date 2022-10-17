import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt


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


if __name__ == "__main__":
    data = read_data()
    draw_plot(data)

