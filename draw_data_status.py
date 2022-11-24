import matplotlib.pyplot as plt
import numpy as np


def draw_data_status_template(x_lims=(0, 2), title='Influences'):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.patch.set_facecolor('yellow')
    ax.set_xlim(x_lims[0], x_lims[1])
    ax.set_ylim(-(x_lims[1] + 1), x_lims[1] + 1)
    # draw green triangle zone
    x1, y1 = [0, 1], [-1, 0]
    x2, y2 = [0, 1], [1, 0]
    ax.plot(x1, y1, 'k', x2, y2, 'k')
    ax.fill_between(x1, y1, y2, facecolor='green')

    # draw others zones
    x1, y1 = [0, x_lims[1]], [-1, -(x_lims[1] + 1)]
    x2, y2 = [0, x_lims[1]], [1, x_lims[1] + 1]

    ax.plot(x1, y1, 'k', x2, y2, 'k')
    x = np.arange(0.0, x_lims[1], 0.01)
    y1 = x + 1
    y2 = [x_lims[1] + 1] * len(x)
    ax.fill_between(x, y1, y2, facecolor='red')
    y2 = [-(x_lims[1] + 1)] * len(x)
    ax.fill_between(x, -y1, y2, facecolor='red')

    x1, y1 = [1, 1], [-(x_lims[1] + 1), x_lims[1] + 1]
    ax.plot(x1, y1, 'k--')
    ax.set_xlabel('l(x, y)')
    ax.set_ylabel('r(x, y)')
    ax.set_title(title)
    return fig


fig_ = draw_data_status_template()
fig_.show()
