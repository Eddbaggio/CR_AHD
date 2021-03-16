import itertools
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib.ticker import AutoMinorLocator

from src.cr_ahd.utility_module.utils import univie_cmap, path_output_custom, univie_colors_100


# =================================================================================================
# PLOTLY
# =================================================================================================

def plotly_plot(solomon_list: List, columns: List[str], ):
    df: pd.DataFrame = combine_eval_files(solomon_list).reset_index()
    df2 = df.groupby(['solution_algorithm', 'num_carriers', 'solomon_base'])[columns].agg('mean')
    df3 = df2.reset_index()
    for col in columns:
        fig = go.Figure()
        colors = itertools.cycle(univie_colors_100)
        for name, group in df3.groupby('solution_algorithm'):
            x_axis_1, x_axis_2 = zip(*group[['solomon_base', 'num_carriers']].values)
            fig.add_bar(x=[x_axis_1, x_axis_2],
                        y=group[col],
                        name=name,
                        text=group[col],
                        marker_color=next(colors),
                        hovertemplate='%{y:.2f}'
                        )

        # fig = px.bar(df3,
        #              x='solomon_base',
        #              y=col,
        #              color='solution_algorithm',
        #              barmode='group',
        #              facet_row='num_carriers',
        #              text=col,
        #              )
        fig.update_traces(texttemplate='%{text:.0f}', textposition='outside', textangle=-90,)
        # fig.update_layout()
        path = path_output_custom.joinpath(f'plotly_bar_plot_{col}.html')
        fig.write_html(str(path), auto_open=True)
    pass


# =================================================================================================
# MATPLOTLIB
# =================================================================================================

def bar_plot_with_errors(solomon_list: list, columns: List[str], filter_conditions=None, fig_size: tuple = (10.5, 4.5)):
    """
    Reads and combines the evaluation csv files of each instance type in solomon_list. Filters for only the given
    algorithms and saves an individual comparison bar plot for each given column.

    :param filter_conditions: list of tuples that define values for equality(!) conditions that must hold for the
    specified columns. Only rows that fulfill one of these conditions will be plotted
    :param fig_size: Figure size (default is for A4 wide); (7, 4.5) for half slide PPT 16:9; ()
    :param solomon_list: the solomon base instances that shall be compared
    :param columns: the columns for which an individual bar plot is saved
    """
    if filter_conditions is None:
        filter_conditions = []
    evaluation: pd.DataFrame = combine_eval_files(solomon_list)

    for col in columns:
        grouped = evaluation[col].unstack('solomon_base').groupby(['solution_algorithm',
                                                                   'num_carriers'])

        # plotting parameters
        n_filtered_groups = grouped.ngroups if filter_conditions == [] else len(filter_conditions)
        width = 1 / (n_filtered_groups + 2)
        ind = np.arange(len(solomon_list))
        cmap = univie_cmap  # univie_cmap_paired
        fig: plt.Figure
        ax: plt.Axes
        fig, ax = plt.subplots()

        # grouped.boxplot(rot=90, sharex=True, sharey=True, subplots=False)

        # plotting
        i = 0
        for name, group in grouped:
            if name in filter_conditions:
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
        ax.set_xticks(ind + width * (n_filtered_groups / 2 - 1))
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
            # ncol=5
        )
        ax.set_title(f'Mean + Std of {col} per algorithm ')
        fig.set_size_inches(fig_size)
        fig.savefig(path_output_custom.joinpath(f'bar_plot_{col}.pdf'), bbox_inches='tight')
        fig.savefig(path_output_custom.joinpath(f'bar_plot_{col}.png'), bbox_inches='tight')
        # plt.show()


def combine_eval_files(solomon_list, save: bool = True):
    """
    iterates through the dir

    :param solomon_list:
    :param save:
    :return:
    """
    evaluation = pd.DataFrame()
    for solomon in solomon_list:
        file_name = next(path_output_custom.joinpath(solomon).glob('*eval.csv'))  # there should only be one eval file
        df = pd.read_csv(file_name)
        evaluation = evaluation.append(df)
    evaluation = evaluation.set_index(['solomon_base', 'rand_copy', 'solution_algorithm', 'num_carriers'])
    if save:
        evaluation.to_csv(path_output_custom.joinpath('Evaluation').with_suffix('.csv'))
    return evaluation


if __name__ == '__main__':
    solomon_list = ['C101', 'C201', 'R101', 'R201', 'RC101', 'RC201']
    columns = ['num_act_veh', 'cost', ]
    # bar_plot_with_errors(solomon_list, columns)
    plotly_plot(solomon_list, columns)
