import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from typing import List

from solving.routing_visitor import RoutingVisitor
from helper.utils import univie_cmap, path_output_custom


def bar_plot_with_errors(solomon_list: list, algorithms: List[str], columns: List[str], fig_size: tuple = (10.5, 4.5)):
    """
    Reads and combines the evaluation csv files of each instance type in solomon_list. Filters for only the given
    algorithms and saves an individual comparison bar plot for each given column.

    :param fig_size: Figure size (default is for A4 wide); (7, 4.5) for half slide PPT 16:9; ()
    :param algorithms: filter the algorithms that are compared in each bar plot
    :param solomon_list: the solomon base instances that shall be compared
    :param columns: the columns for which an individual bar plot is saved
    """
    eval = combine_eval_files(algorithms, solomon_list)
    for col in columns:
        grouped = eval[col].unstack('solomon_base').groupby(['initializing_visitor', 'routing_visitor', 'finalizing_visitor' ])

        # plotting parameters
        width = 1 / (grouped.ngroups + 2)
        ind = np.arange(len(solomon_list))
        cmap = univie_cmap
        # plt.get_cmap('Set1')  # 'Paired', 'Set1',
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
                x=ind + i * width * 1.1,  # *1.1 for small gaps between bars, alternatively do width =0.9
                height=group.mean(),
                width=width,
                lw=1,
                color=cmap(i),
                # edgecolor=cmap(i * 2 + 1),
                label=name,
                yerr=group.std(),
                capsize=width * 15,
                error_kw=dict(elinewidth=width * 5,
                              # ecolor='#7F7F7F',
                              ),
            )
            i += 1

        # x axis format
        # ax.set_xlim(0 - 2 * width, grouped.ngroups + 6 * width)
        ax.set_xticks(ind + width * (grouped.ngroups / 2 - 1))
        ax.set_xticklabels(solomon_list)
        ax.set_axisbelow(True)
        minor_locator = AutoMinorLocator(2)
        # ax.xaxis.set_minor_locator(minor_locator)
        # ax.grid(which='minor')
        ax.grid(which='major', axis='y')

        # y axis format
        ax.set_ylabel(col)

        # legend, title, size, saving
        ax.legend(
            loc='upper center',
            bbox_to_anchor=(0.5, -0.1),
            fancybox=True,
            shadow=True,
            ncol=5
        )
        ax.set_title(f'Mean + Std of {col} per algorithm ')
        fig.set_size_inches(fig_size)
        fig.savefig(f'../data/Output/Custom/bar_plot_{col}.pdf', bbox_inches='tight')
        fig.savefig(f'../data/Output/Custom/bar_plot_{col}.png', bbox_inches='tight')
        # plt.show()


def combine_eval_files(algorithms, solomon_list, save: bool = True):
    """


    :param algorithms:
    :param solomon_list:
    :param save:
    :return:
    """
    eval = pd.DataFrame()
    for solomon in solomon_list:
        file_name = path_output_custom.joinpath(solomon, f'{solomon}_3_15_ass_eval').with_suffix('.csv')
        df = pd.read_csv(file_name)
        eval = eval.append(df)
    eval = eval[eval.routing_visitor.isin(algorithms)]
    eval = eval.set_index(['solomon_base', 'rand_copy', 'initializing_visitor', 'routing_visitor',
                           'finalizing_visitor'])
    if save:
        eval.to_csv(path_output_custom.joinpath('Evaluation').with_suffix('.csv'))
    return eval


if __name__ == '__main__':
    solomon_list = ['C101', 'C201', 'R101', 'R201', 'RC101', 'RC201']
    # no need to solve the instances before evaluation, the plotting function will read the solution json files

    bar_plot_with_errors(solomon_list,
                         algorithms=[rs.__name__ for rs in RoutingVisitor.__subclasses__()],
                         columns=['num_act_veh',
                                  'cost',
                                  ],
                         )
#     bar_plot_with_errors('cost')
