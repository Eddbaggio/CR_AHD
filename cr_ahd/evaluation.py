import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator


def bar_plot_with_errors(solomon_list: list, column: str):
    eval = pd.DataFrame()
    for solomon in solomon_list:
        file_name = f'../data/Output/Custom/{solomon}/{solomon}_3_15_ass_eval.csv'
        df = pd.read_csv(file_name)
        eval = eval.append(df)
    eval = eval.set_index(['solomon_base', 'rand_copy', 'algorithm'])
    grouped = eval[column].unstack('solomon_base').groupby('algorithm')

    # plotting parameters
    width = 1 / (grouped.ngroups + 2)
    ind = np.arange(len(solomon_list))
    cmap = plt.get_cmap('Paired')
    cmap_median = ['#C9D9E7', '#91B2CD',
                   '#EDBBA7', '#EDBBA7',
                   '#CED1BC', '#CED1BC',
                   '#E9D4AC', '#E9D4AC',
                   '#B9CEC9', '#B9CEC9']
    cmap_coolors = ['#F94144',
                    '#F3722C',
                    '#F8961E',
                    '#F9C74F',
                    '#90BE6D',
                    '#43AA8B',
                    '#577590',
                    ]
    fig: plt.Figure
    ax: plt.Axes
    fig, ax = plt.subplots()

    # grouped.boxplot(rot=90, sharex=True, sharey=True, subplots=False)

    # plotting
    i = 0
    for name, group in grouped:
        ax.bar(
            x=ind + i * width,
            height=group.mean(),
            width=width,
            lw=1,
            color=cmap(i),
            # edgecolor=cmap(i * 2 + 1),
            label=name,
            yerr=group.std(),
            # capsize=width * 15,
            # error_kw=dict(elinewidth=width * 5, ecolor='#7F7F7F'),
        )
        i += 1

    # x axis format
    # ax.set_xlim(0 - 2 * width, grouped.ngroups + 6 * width)
    ax.set_xticks(ind + width * (grouped.ngroups/2-1))
    ax.set_xticklabels(solomon_list)
    ax.set_axisbelow(True)
    minor_locator = AutoMinorLocator(2)
    # ax.xaxis.set_minor_locator(minor_locator)
    # ax.grid(which='minor')
    ax.grid(which='major', axis='y')

    # y axis format
    ax.set_ylabel(column)

    # legend, title, size, saving
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.1),
        fancybox=True,
        shadow=True,
        ncol=5
    )
    ax.set_title(f'Mean + Std of {column} per algorithm ')
    fig.set_size_inches(10.5, 4.5)
    fig.savefig(f'../data/Output/Custom/bar_plot.pdf', bbox_inches='tight')
    plt.show()

# if __name__ == '__main__':
#     bar_plot_with_errors('cost')
