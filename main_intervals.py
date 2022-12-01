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
    plt.errorbar(x, y, yerr=y_err, ecolor='cyan', label='intervals_ch_1', elinewidth=0.8, capsize=4,
                 capthick=1)
    y_err = [optimal_m * (interval[1] - interval[0]) / 2 for interval in intervals[1]]
    y = [optimal_m * interval[1] - y_err[num] for num, interval in enumerate(intervals[1])]
    plt.errorbar(x, y, yerr=y_err, ecolor='red', label='intervals_ch_2', elinewidth=0.8, capsize=4,
                 capthick=1)
    plt.legend(frameon=False)
    plt.title(f'Intervals intersection')
    plt.xlabel("n")
    plt.ylabel("mV")
    if need_save:
        plt.savefig(f'{save_path}/weighted_intervals.png')
    plt.show()


def draw_all_intervals_edge(intervals, start, optimal_m, need_save=True, save_path=''):
    data_len = (len(intervals[0]))
    x = np.arange(start, start + data_len, 1, dtype=int)
    y_err = [(interval[1] - interval[0]) / 2 for interval in intervals[0]]
    y = [interval[1] - y_err[num] for num, interval in enumerate(intervals[0])]
    plt.errorbar(x, y, yerr=y_err, ecolor='cyan', label='intervals_ch_1', elinewidth=0.8, capsize=4,
                 capthick=1)
    y_err = [optimal_m * (interval[1] - interval[0]) / 2 for interval in intervals[1]]
    y = [optimal_m * interval[1] - y_err[num] for num, interval in enumerate(intervals[1])]
    plt.errorbar(x, y, yerr=y_err, ecolor='red', label='intervals_ch_2', elinewidth=0.8, capsize=4,
                 capthick=1)
    plt.legend(frameon=False)
    plt.title(f'Intervals intersection')
    plt.xlabel("n")
    plt.ylabel("mV")
    if need_save:
        plt.savefig(f'{save_path}/weighted_intervals.png')
    plt.show()


def get_regression_intervals(params, len_of_intervals=200):
    x = np.arange(1, len_of_intervals + 1, 1, dtype=int)
    intervals = []
    for x_ in x:
        intervals.append([params[0][0] * x_ + params[1][0], params[0][1] * x_ + params[1][1]])
    return intervals


def draw_interval_regression(intervals, params, need_save=True, save_path=''):
    data_len = (len(intervals[0]))
    x = np.arange(1, data_len + 1, 1, dtype=int)
    for num, data_l in enumerate(intervals, start=0):
        y_err = [(interval[1] - interval[0]) / 2 for interval in intervals[num]]
        y = [interval[1] - y_err[num] for num, interval in enumerate(intervals[num])]
        plt.errorbar(x, y, yerr=y_err, ecolor='cyan', label=f'intervals_ch_{num + 1}', elinewidth=0.8, capsize=4,
                     capthick=1)
        reg_intervals = get_regression_intervals(params=params[num])
        y_err = [(interval[1] - interval[0]) / 2 for interval in reg_intervals]
        y = [interval[1] - y_err[num] for num, interval in enumerate(reg_intervals)]
        plt.errorbar(x, y, yerr=y_err, ecolor='red', label='interval regression', elinewidth=0.8, capsize=4,
                     capthick=1)
        plt.legend(frameon=False)
        plt.title(f'Chanel_{num + 1} interval regression')
        plt.xlabel("n")
        plt.ylabel("mV")
        if need_save:
            plt.savefig(f'{save_path}/regression_intervals_ch_{num + 1}.png')
        plt.show()


def draw_interval_with_edge(intervals, edge_points, need_save=True, save_path=''):
    data_len = (len(intervals[0]))
    x = np.arange(1, data_len + 1, 1, dtype=int)
    for num, data_l in enumerate(intervals, start=0):
        y_err = [(interval[1] - interval[0]) / 2 for interval in intervals[num]]
        y = [interval[1] - y_err[num] for num, interval in enumerate(intervals[num])]
        plt.errorbar(x[:edge_points[num][0]], y[:edge_points[num][0]], yerr=y_err[:edge_points[num][0]], ecolor='cyan',
                     label=f'set 1, {edge_points[num][0] - 1}', elinewidth=0.8, capsize=4,
                     capthick=1)
        plt.errorbar(x[edge_points[num][0]:edge_points[num][1]], y[edge_points[num][0]:edge_points[num][1]],
                     yerr=y_err[edge_points[num][0]:edge_points[num][1]], ecolor='green',
                     label=f'set {edge_points[num][0]}, {edge_points[num][1] - 1}', elinewidth=0.8, capsize=4,
                     capthick=1)
        plt.errorbar(x[edge_points[num][1]:], y[edge_points[num][1]:], yerr=y_err[edge_points[num][1]:],
                     ecolor='red',
                     label=f'set {edge_points[num][1]}, 200', elinewidth=0.8, capsize=4,
                     capthick=1)
        plt.legend(frameon=False)
        plt.title(f'Chanel_{num + 1} sets of intervals')
        plt.xlabel("n")
        plt.ylabel("mV")
        if need_save:
            plt.savefig(f'{save_path}/sets_of_intervals_ch_{num + 1}.png')
        plt.show()


def dif_drift_component_edge(interval_d, edge_points, drift_params_3):
    new_list = []
    for list_num, list_ in enumerate(interval_d):
        new_list__ = []
        for num, drift_param in enumerate(drift_params_3[list_num]):
            if num == 0:
                new_list_ = list_[:edge_points[list_num][0]]
                start = 0
            elif num == 1:
                new_list_ = list_[edge_points[list_num][0]:edge_points[list_num][1]]
                start = edge_points[list_num][0]
            else:
                new_list_ = list_[edge_points[list_num][1]:]
                start = edge_points[list_num][1]
            for num_, interval in enumerate(new_list_, start=start):
                new_list__.append([interval[0] - (num_ + 1) * drift_param[0][1],
                                   interval[1] - (num_ + 1) * drift_param[0][0]])
        new_list.append(new_list__)
    return new_list


def get_regression_intervals_edge(edge_points, params, len_of_intervals=200):
    intervals = []
    for x in range(edge_points[0]):
        intervals.append([params[0][0][0] * (x + 1) + params[0][1][0], params[0][0][1] * (x + 1) + params[0][1][1]])
    for x in range(edge_points[0], edge_points[1]):
        intervals.append([params[1][0][0] * (x + 1) + params[1][1][0], params[1][0][1] * (x + 1) + params[1][1][1]])
    for x in range(edge_points[1], len_of_intervals):
        intervals.append([params[2][0][0] * (x + 1) + params[2][1][0], params[2][0][1] * (x + 1) + params[2][1][1]])
    return intervals


def draw_interval_regression_edge(intervals, edge_points, params3, need_save=True, save_path=''):
    data_len = (len(intervals[0]))
    x = np.arange(1, data_len + 1, 1, dtype=int)
    for num, data_l in enumerate(intervals, start=0):
        y_err = [(interval[1] - interval[0]) / 2 for interval in intervals[num]]
        y = [interval[1] - y_err[num] for num, interval in enumerate(intervals[num])]
        plt.errorbar(x, y, yerr=y_err, ecolor='cyan', label=f'intervals_ch_{num + 1}', elinewidth=0.8, capsize=4,
                     capthick=1)
        reg_intervals = get_regression_intervals_edge(edge_points[num], params3[num])
        y_err = [(interval[1] - interval[0]) / 2 for interval in reg_intervals]
        y = [interval[1] - y_err[num] for num, interval in enumerate(reg_intervals)]
        plt.errorbar(x, y, yerr=y_err, ecolor='red', label='interval regression', elinewidth=0.8, capsize=4,
                     capthick=1)
        plt.legend(frameon=False)
        plt.title(f'Chanel_{num + 1} interval regression with 3 section')
        plt.xlabel("n")
        plt.ylabel("mV")
        if need_save:
            plt.savefig(f'{save_path}/regression_intervals_with_3_section_ch_{num + 1}.png')
        plt.show()


def get_out_coeff(data_list):
    def get_point(interval):
        interval_r = (interval[1] - interval[0]) / 2
        return interval[1] - interval_r

    out_coeff_l = []
    for i in range(len(data_list[0])):
        out_coeff_l.append(get_point(data_list[0][i]) / get_point(data_list[1][i]))
    return out_coeff_l


def draw_residuals(residuals_l, intersection_, title='', save_path=None):
    from draw_data_status import get_rad, get_mid
    data_len = (len(residuals_l))
    x = np.arange(1, data_len + 1, 1, dtype=int)
    y_err = [get_rad(interval) for interval in residuals_l]
    y = [get_mid(interval) for interval in residuals_l]
    plt.errorbar(x, y, yerr=y_err, ecolor='cyan', label=f'residuals', elinewidth=0.8, capsize=4,
                 capthick=1)
    x1, y1 = [1, 200], [intersection_[0], intersection_[0]]
    x2, y2 = [1, 200], [intersection_[1], intersection_[1]]
    x3, y3 = [1, 200], [(intersection_[1] + intersection_[0]) / 2, (intersection_[1] + intersection_[0]) / 2]
    plt.plot(x1, y1, 'r--', label=f'[{y1[0]:.2e}, {y2[1]:.2e}]')
    plt.plot(x2, y2, 'r--')
    plt.plot(x3, y3, 'k--', label=f'mid={y3[0]:.2e}')
    plt.legend(frameon=False)
    plt.title(title)
    plt.yticks(fontsize=5)
    plt.xlabel("n")
    plt.ylabel("residual")
    if save_path is not None:
        plt.savefig(f'{save_path}/regression_intervals_with_3_section_ch_1.png')
    plt.show()


def regularization(residuals, edge_point, intersection_, need_visualize=False, title='', label=''):
    w_list = [1] * len(residuals)
    left_intervals = residuals[:edge_point].copy()
    for num, interval in enumerate(left_intervals):
        mid = (left_intervals[num][1] + left_intervals[num][0]) / 2
        if interval[0] > intersection_[0]:
            left_intervals[num][0] = intersection_[0]
            left_intervals[num][1] = mid + (mid - intersection_[0])
            w_list[num] = (mid - intersection_[0]) / ((interval[1] - interval[0]) / 2)
        if interval[1] < intersection_[1]:
            left_intervals[num][1] = intersection_[1]
            left_intervals[num][0] = mid - (intersection_[1] - mid)
            w_list[num] = (intersection_[1] - mid) / ((interval[1] - interval[0]) / 2)
    if need_visualize:
        plt.hist(w_list, label=label)
        plt.xlabel(label)
        plt.legend(frameon=False)
        plt.title(title)
        plt.show()
    return left_intervals + residuals[edge_point:]


def draw_params(params_1, params_2=None, title='', param_name='', param_name_2=''):
    x = np.arange(1, len(params_1) + 1, 1, dtype=int)
    fig, ax = plt.subplots()
    if params_2:
        ax.plot(x, np.fabs(np.array(params_1)), label=param_name)
        ax.plot(x, 1 - np.array(params_2), label=param_name_2)
    else:
        ax.plot(x, params_1, label=param_name)
    plt.xlabel("n")
    plt.ylabel(param_name)
    ax.legend()
    ax.set_title(title)
    plt.show()


def get_matlab_form(intervals):
    res_str = '['
    for interval in intervals:
        res_str = f'{res_str}[{interval[0]}, {interval[1]}];'
    print(f'{res_str[:-1]}]')


def draw_mode(mode_data, title='Mode'):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    data_len = len(mode_data)
    for num, mode_data_inst in enumerate(mode_data):
        x1, y1 = [mode_data_inst[0][0], mode_data_inst[0][1]], [mode_data_inst[1], mode_data_inst[1]]
        ax.plot(x1, y1, 'r')
        if num < data_len - 1:
            x2, y2 = [mode_data_inst[0][1], mode_data_inst[0][1]], [mode_data_inst[1], mode_data[num + 1][1]]
            ax.plot(x2, y2, 'r')
    ax.set_xlabel('ticks')
    ax.set_ylabel('frequency')
    ax.set_title(title)
    plt.show()


if __name__ == "__main__":
    data_postfix = '800nm_0.23mm.csv'
    interval_data = read_data_with_intervals(
        [f'./data/ch_1/Канал 1_{data_postfix}', f'./data/ch_2/Канал 2_{data_postfix}'])
    save_p = f'./pictures/intervals'
    # intervals_regression_drift_params = [[3.4551e-06, 4.2070e-06], [5.1628e-06, 6.2094e-06]]
    # intervals_regression_params = [([3.4551e-06, 4.2070e-06], [4.7202e-01, 4.7214e-01]), ([5.1628e-06, 6.2094e-06],
    #                                                                                       [5.0301e-01, 5.0312e-01])]
    # draw_interval_regression(interval_data, intervals_regression_params, True, save_path=save_p)
    # intervals_w_o_drift = dif_drift_component(interval_data, intervals_regression_drift_params)

    # opt_m = get_inner_r(intervals_w_o_drift, [0.9377, 0.9385], True, save_path=save_p)
    # draw_all_intervals(intervals_w_o_drift, opt_m[0], True, save_path=save_p)
    # draw_interval_with_edge(interval_data, [[43, 178], [41, 190]], True, save_path=save_p)
    edge_points_ = [[43, 178], [41, 190]]
    intervals_regression_params_3 = [[([0, 1.2058e-05], [0.4720, 0.4721]),
                                      ([1.6296e-06, 4.2911e-06], [4.7202e-01, 4.7227e-01]),
                                      ([3.5000e-06, 1.0823e-05], [4.7078e-01, 4.7213e-01])],
                                     [([2.7222e-06, 1.1479e-05], [5.0296e-01, 5.0313e-01]),
                                      ([3.9064e-06, 6.1838e-06], [5.0301e-01, 5.0328e-01]),
                                      ([0.0, 3.6175e-05], [0.4971, 0.5042])]]
    # intervals_reg_w_o_drift = dif_drift_component_edge(interval_data, edge_points_, intervals_regression_params_3)
    # draw_interval_regression_edge(interval_data, edge_points_, intervals_regression_params_3, True, save_path=save_p)
    # out_coeff = get_out_coeff([interval_data[0][:41], interval_data[1][:41]])
    # R_outer = [min(out_coeff), max(out_coeff)]
    # print(R_outer)
    # opt_m = get_inner_r([intervals_reg_w_o_drift[0][:41], intervals_reg_w_o_drift[1][:41]], R_outer, True,
    #                    save_path=save_p+'/40')
    # draw_all_intervals_edge([intervals_reg_w_o_drift[0][:41], intervals_reg_w_o_drift[1][:41]], 0, opt_m[0], True,
    #                         save_path=save_p+'/40')
    # out_coeff = get_out_coeff([interval_data[0][41:177], interval_data[1][41:177]])
    # R_outer = [min(out_coeff), max(out_coeff)]
    # print(R_outer)
    # opt_m = get_inner_r([intervals_reg_w_o_drift[0][41:177], intervals_reg_w_o_drift[1][41:177]], R_outer, True,
    #                     save_path=save_p + '/177')
    # draw_all_intervals_edge([intervals_reg_w_o_drift[0][41:177], intervals_reg_w_o_drift[1][41:177]], 41, opt_m[0], True,
    #                         save_path=save_p + '/177')
    # out_coeff = get_out_coeff([interval_data[0][177:], interval_data[1][177:]])
    # R_outer = [min(out_coeff), max(out_coeff)]
    # print(R_outer)
    # opt_m = get_inner_r([intervals_reg_w_o_drift[0][177:], intervals_reg_w_o_drift[1][177:]], R_outer, True,
    #                     save_path=save_p + '/200')
    # draw_all_intervals_edge([intervals_reg_w_o_drift[0][177:], intervals_reg_w_o_drift[1][177:]], 177, opt_m[0], True,
    #                         save_path=save_p + '/200')
    from draw_data_status import get_residuals, add_point, draw_data_status_template, get_influences, get_residuals_1, \
        get_intersections_wrong_int, get_intersections

    intervals_residuals = get_residuals(interval_data, edge_points_, intervals_regression_params_3)
    intervals_residuals_1 = get_residuals_1(interval_data, [[intervals_regression_params_3[0][1]],
                                                            [intervals_regression_params_3[1][1]]])

    # for num_, res_list in enumerate(intervals_residuals_1):
    #     inters = get_intersections(intervals_residuals_1[num_][edge_points_[num_][0]:edge_points_[num_][1]])
    #     infls, intersection = get_influences(res_list, inters)
    #     draw_residuals(res_list, intersection, title=f'Residuals with central regression, ch_{num_ + 1}')
    #     m_l = max([res[0] for res in infls])
    #     fig_, ax_ = draw_data_status_template([0, max(m_l, 2)], title=f'Influences with central regression, ch_{num_ + 1}')
    #     for infl in infls:
    #         add_point(infl, ax_)
    #     fig_.show()
    #
    # for num_, res_list in enumerate(intervals_residuals):
    #     inter_w = get_intersections_wrong_int(intervals_residuals[num_][edge_points_[num_][0]:edge_points_[num_][1]])
    #     print(inter_w)
    #     inters = get_intersections(intervals_residuals[num_][edge_points_[num_][0]:edge_points_[num_][1]])
    #     infls, intersection = get_influences(res_list, inters)
    #     draw_residuals(res_list, intersection, title=f'Residuals with central regression, ch_{num_ + 1}')
    #     m_l = max([res[0] for res in infls])
    #     fig_, ax_ = draw_data_status_template([0, max(m_l, 2)], title=f'Influences with central regression, ch_{num_ + 1}')
    #     for infl in infls:
    #         add_point(infl, ax_)
    #     fig_.show()

    # intervals_regression_params_3_1 = intervals_regression_params_3.copy()
    # intervals_regression_params_3_1[0][2] = intervals_regression_params_3_1[0][1]
    # intervals_regression_params_3_1[1][2] = intervals_regression_params_3_1[1][1]

    intervals_residuals = get_residuals_1(interval_data, [[intervals_regression_params_3[0][1]],
                                                            [intervals_regression_params_3[1][1]]])
    # for num_, res_list in enumerate(intervals_residuals):
    #     inter_w = get_intersections_wrong_int(intervals_residuals[num_][edge_points_[num_][0]:edge_points_[num_][1]])
    #     print(inter_w)
    #     inters = get_intersections(intervals_residuals[num_][edge_points_[num_][0]:edge_points_[num_][1]])
    #     infls, intersection = get_influences(res_list, inters)
    #     draw_residuals(res_list, intersection, title=f'Residuals with all regressions, ch_{num_}')
    #     m_l = max([res[0] for res in infls])
    #     fig_, ax_ = draw_data_status_template([0, max(m_l, 2)], title=f'Influences with all regressions, ch_{num_}')
    #     for infl in infls:
    #         add_point(infl, ax_)
    #     fig_.show()

    # for num_, res_list in enumerate(intervals_residuals):
    #     inter_w = get_intersections_wrong_int(intervals_residuals[num_][edge_points_[num_][0]:edge_points_[num_][1]])
    #     print(inter_w)
    #     inters = get_intersections(intervals_residuals[num_][edge_points_[num_][0]:edge_points_[num_][1]])
    #     new_res_list = regularization(res_list, edge_points_[num_][0], inters, True,
    #                                   f'Weight of regularization, ch_{num_ + 1}', f'w_ch_{num_ + 1}')
    #     infls, intersection = get_influences(new_res_list, inters)
    #     draw_residuals(new_res_list, intersection, title=f'Residuals with central regression, ch_{num_ + 1}')
    #     m_l = max([res[0] for res in infls])
    #     draw_params([infl[0] for infl in infls], title=f'High leverage, ch_{num_ + 1}', param_name='l')
    #     draw_params([infl[1] for infl in infls], title=f'Relative residual, ch_{num_ + 1}', param_name='|r|',
    #                 params_2=[infl[0] for infl in infls], param_name_2='1-l')
    #     fig_, ax_ = draw_data_status_template([0, max(m_l, 2)], title=f'Influences with central regression, ch_{num_ + 1}')
    #     for infl in infls:
    #         add_point(infl, ax_)
    #     fig_.show()

    draw_mode([([1, 1.5], 1), ([1.5, 2], 2), ([2, 3], 1)])