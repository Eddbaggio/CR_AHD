import itertools
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, FactorRange
from matplotlib.ticker import AutoMinorLocator
from bokeh.plotting import figure, show, output_file

from src.cr_ahd.utility_module.utils import univie_cmap, path_output_custom, univie_colors_100

labels = {
    'num_carriers': 'Number of Carriers',
    'solomon_base': 'Instance',
    'cost': 'Routing Costs',
    'num_act_veh': 'Number of Vehicles',
    'StaticI1Insertion': 'Static, no collaboration',
    'StaticI1InsertionWithAuction': 'Static, with collaboration',
    'StaticSequentialInsertion': 'Static Sequential, no collaboration',
    'DynamicI1Insertion': 'Dynamic, no collaboration',
    'DynamicI1InsertionWithAuctionA': 'Dynamic, with collaboration (A)',
    'DynamicI1InsertionWithAuctionB': 'Dynamic, with collaboration (B)',
    'DynamicI1InsertionWithAuctionC': 'Dynamic, with collaboration (C)',
    'DynamicSequentialInsertion': 'Dynamic  Sequential, no collaboration',
}


# =================================================================================================
# BOKEH
# =================================================================================================

def bokeh_plot(solomon_list: List, attributes: List[str]):
    df: pd.DataFrame = combine_eval_files(solomon_list).reset_index()
    df.num_carriers = df.num_carriers.astype(str)
    x_range = [(nc, sol) for nc in np.unique(df['num_carriers']) for sol in np.unique(df['solomon_base'])]
    for col in attributes:
        subplots = []
        for name, group in df.groupby('solution_algorithm'):
            p = figure(x_range=FactorRange(*x_range),
                       sizing_mode='stretch_width')
            grouped = group.groupby(['num_carriers', 'solomon_base'])
            source = ColumnDataSource(grouped)
            p.vbar(source=source,
                   x='solomon_base_num_carriers',
                   top=f'{col}_mean',
                   width=0.9,
                   legend_label=name,
                   )
            subplots.append(p)

        path = path_output_custom.joinpath(f'bokeh_{col}.html')
        output_file(str(path), f'bokeh_{col}')
        show(column(*subplots))
    pass


# =================================================================================================
# PLOTLY
# =================================================================================================

def plotly_bar_plot(solomon_list: List, attributes: List[str], ):
    df: pd.DataFrame = combine_eval_files(solomon_list)[attributes]
    for attr in attributes:
        attr_df = df[attr].unstack('solution_algorithm').reset_index('rand_copy', drop=True).groupby(level=[0, 1]).agg(
            'mean')
        fig = go.Figure()
        colors = itertools.cycle([univie_colors_100[0]]+univie_colors_100[2:])
        for col in attr_df.columns:
            fig.add_bar(x=list(zip(*attr_df.index)),
                        y=attr_df[col],
                        name=labels[col],
                        marker_color=next(colors),
                        hovertemplate='%{y:.2f}',
                        texttemplate='%{y:.0f}',
                        textposition='outside',
                        textangle=-90,
                        textfont_size=14
                        )
        for v_line_x in np.arange(1.5, 10.5, 2):
            fig.add_vline(x=v_line_x, line_width=1, line_color="grey")
        fig.update_layout(title=f'Mean {labels[attr]}',
                          xaxis_title=f'{labels["num_carriers"]} // {labels["solomon_base"]}',
                          # line break with <br>, but then its outside the plot, needed to adjust margins then
                          template='plotly_white',
                          uniformtext_minsize=14, uniformtext_mode='show')

        # multicategory axis not supported by plotly express
        # fig = px.bar(attr_df, x=attr_df.index.names, y=attr_df.columns)
        path = path_output_custom.joinpath(f'plotly_bar_plot_{attr}.html')
        fig.write_html(str(path), auto_open=True)
        # path = path_output_custom.joinpath(f'plotly_bar_plot_{attr}.svg')
        # fig.write_image(str(path))
    pass


# =================================================================================================
# MATPLOTLIB
# =================================================================================================

def bar_plot_with_errors(solomon_list: list, attributes: List[str], filter_conditions=None,
                         fig_size: tuple = (10.5, 4.5)):
    """
    Reads and combines the evaluation csv files of each instance type in solomon_list. Filters for only the given
    algorithms and saves an individual comparison bar plot for each given column.

    :param filter_conditions: list of tuples that define values for equality(!) conditions that must hold for the
    specified columns. Only rows that fulfill one of these conditions will be plotted
    :param fig_size: Figure size (default is for A4 wide); (7, 4.5) for half slide PPT 16:9; ()
    :param solomon_list: the solomon base instances that shall be compared
    :param attributes: the columns for which an individual bar plot is saved
    """
    if filter_conditions is None:
        filter_conditions = []
    evaluation: pd.DataFrame = combine_eval_files(solomon_list)

    for col in attributes:
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
        fig.savefig(path_output_custom.joinpath(f'matplotlib_bar_plot_{col}.pdf'), bbox_inches='tight')
        fig.savefig(path_output_custom.joinpath(f'matplotlib_bar_plot_{col}.png'), bbox_inches='tight')
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
    attributes = ['num_act_veh', 'cost', ]
    # bar_plot_with_errors(solomon_list, attributes)
    # plotly_plot(solomon_list, attributes)
    # bokeh_plot(solomon_list, attributes)
    plotly_bar_plot(solomon_list, attributes)
