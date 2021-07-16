import warnings
from pathlib import Path
from typing import Sequence

import pandas as pd
import plotly.express as px
from plotly import graph_objects as go
from plotly.subplots import make_subplots

import src.cr_ahd.utility_module.utils as ut

labels = {
    'num_carriers': 'Number of Carriers',
    'solomon_base': 'Instance',
    'travel_distance': 'Travel Distance',
    'travel_duration': 'Travel Duration',
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

config = dict({'scrollZoom': True})


# =================================================================================================
# PLOTLY
# =================================================================================================
def bar_chart(df: pd.DataFrame,
              values,
              category,
              color,
              facet_row,
              facet_col,
              barmode='group',
              title: str = f"<b>n</b>: Number of requests per carrier<br>"
                           f"<b>rad</b>: Radius of the carriers' operational area around the depot<br>",
              width=None,
              height=None,
              show: bool = True,
              html_path=None,
              ):
    """
    MultiIndex Hierarchy is:
    rad, n, run, solution_algorithm, carrier_id_

    :param df: multi-index dataframe
    :return:
    """

    # group and aggregate
    # sum over necessary levels and values
    grouped = df.groupby([x for x in [facet_col, facet_row, color, category] if x])
    agg_dict = {col: sum for col in df.columns}
    agg_dict['acceptance_rate'] = 'mean'
    df = grouped.agg(agg_dict)

    # prepare for use in plotly express
    df: pd.DataFrame = df.reset_index()
    df = df.round(2)

    '''
    # bar plot
    fig = make_subplots(rows=df[facet_row].nunique(),
                        cols=df[facet_col].nunique(),
                        specs=[[{"secondary_y": True}] * df[facet_col].nunique()] * df[facet_row].nunique(),
                        column_titles=[f'{facet_col}={val}' for val in df[facet_col].unique()],
                        row_titles=[f'{facet_row}={val}' for val in df[facet_row].unique()],
                        shared_xaxes='all',
                        shared_yaxes='all',
                        vertical_spacing=0.1,
                        horizontal_spacing=0.1,
                        # subplot_titles=("Plot 1", "Plot 2", ...)
                        )
    colormap = ut.map_to_univie_colors(df[color].unique())

    for col_idx, col in enumerate(df[facet_col].unique(), start=1):
        for row_idx, row in enumerate(df[facet_row].unique(), start=1):
            for legend_idx, legend_group in enumerate(df[color].unique()):
                data = df.loc[(df[facet_col] == col) & (df[facet_row] == row) & (df[color] == legend_group)]
                fig.add_bar(
                    x=data[category],
                    y=data[values],
                    marker=dict(color=colormap[legend_group], opacity=0.6),
                    text=data[values],
                    row=row_idx,
                    col=col_idx,
                    name=legend_group,
                    legendgroup=legend_group,
                )

                fig.add_scatter(
                    x=data[category] - 0.2 + legend_idx * 0.5,
                    y=data['num_tours'],
                    mode='markers',
                    marker=dict(color=colormap[legend_group],
                                size=10,
                                opacity=1),
                    row=row_idx,
                    col=col_idx,
                    name=legend_group,
                    legendgroup=legend_group,
                    showlegend=False,
                    secondary_y=True,

                )

    # fig.update_yaxes(range=[0, 10], secondary_y=True)
    fig.update_layout(title_text=f'{values}', template='plotly_white')
    '''

    # bar plot
    fig = px.bar(df,
                 x=category,
                 y=values,
                 title=title,
                 color=color,
                 color_discrete_sequence=ut.univie_colors_100,
                 facet_row=facet_row,
                 facet_col=facet_col,
                 text=values,
                 template='plotly_white',
                 hover_data=df.columns.values,
                 barmode=barmode,
                 category_orders={'solution_algorithm': [
                     'IsolatedPlanningNoTW',
                     'IsolatedPlanning',
                     'CollaborativePlanningNoTW',
                     'CollaborativePlanning',
                     # 'CentralizedPlanning'
                 ],
                     'rad': [150, 200, 300]},
                 width=width,
                 height=height,
                 )
    fig.update_yaxes(range=[0, 12000])
    fig.update_xaxes(type='category')
    if show:
        fig.show(config=config)

    if html_path:
        fig.write_html(html_path, )


def print_top_level_stats(df: pd.DataFrame):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 0):
        # aggregate the 3 carriers
        grouped = df.groupby(['rad', 'n', 'solution_algorithm', 'run']).agg({'sum_profit': sum,
                                                                             'sum_travel_distance': sum,
                                                                             'sum_travel_duration': sum,
                                                                             'sum_load': sum,
                                                                             'sum_revenue': sum,
                                                                             'num_tours': sum,
                                                                             'acceptance_rate': 'mean',
                                                                             })

        print('=============/ stats per instance-algorithm combination /=============')
        print(grouped, '\n')

        print('=============/ number of solved instances per algorithm /=============')
        for name, group in grouped.groupby('solution_algorithm'):
            print(f'{group["num_tours"].astype(bool).sum(axis=0)}/{len(group)} solved by {name}')
        print('\n')

        # aggregate the 20 runs
        print('=============/ average over runs  /=============')
        print(grouped.groupby(['rad', 'n', 'solution_algorithm']).agg('mean'), '\n')

        # # csv
        # bar_chart = grouped.groupby('n')
        # print('=============/ CSV: rad and algorithm /=============')
        # for name, group in bar_chart:
        #     print(f'Group: n={name}')
        #     print(group.reset_index('n')['sum_profit'].unstack('solution_algorithm')[
        #               ['IsolatedPlanning',
        #                'CollaborativePlanning',
        #                'IsolatedPlanningNoTW',
        #                'CollaborativePlanningNoTW',
        #                # 'CentralizedPlanning',
        #                ]].to_csv(), '\n')

        # aggregate the instance types

        # collaboration gain
        print('=============/ collaboration gains /=============')
        g = grouped.groupby('solution_algorithm').agg('mean')
        for pair in [('CollaborativePlanning', 'IsolatedPlanning'),
                     ('CollaborativePlanningNoTW', 'IsolatedPlanningNoTW')]:
            print(f'{pair[0]} & {pair[1]}:')
            for stat in ['sum_profit', 'num_tours']:
                try:
                    gain = g.loc[pair[0], stat] / g.loc[pair[1], stat] - 1
                    print(f'\tCollaboration gain {stat}: {gain}')
                except:
                    continue

        print('=============/ consistency check: collaborative better than isolated? /=============')
        for name, group in grouped.groupby(['rad', 'n', 'run'], as_index=False):
            d = group.reset_index(['rad', 'n', 'run'], True)
            try:
                assert d.loc['CollaborativePlanning', 'sum_profit'] >= d.loc[
                    'IsolatedPlanning', 'sum_profit']
            except AssertionError:
                warnings.warn(f'{name}: Collaborative is worse than Isolated!')
            except KeyError:
                None
            try:
                assert d.loc['CollaborativePlanningNoTW', 'sum_profit'] >= d.loc[
                    'IsolatedPlanningNoTW', 'sum_profit']
            except AssertionError:
                warnings.warn(f'{name}: CollaborativeNoTW is worse than IsolatedNoTW!')
            except KeyError:
                None


if __name__ == '__main__':
    df = pd.read_csv(
        "C:/Users/Elting/ucloud/PhD/02_Research/02_Collaborative Routing for Attended Home "
        "Deliveries/01_Code/data/Output/Gansterer_Hartl/evaluation_carrier_#004.csv",
        index_col=['rad',
                   'n',
                   'run',
                   'carrier_id_',  # only if agg_level of the writer was 'carrier'
                   'rad',
                   'n',
                   'run',
                   'solution_algorithm',
                   'tour_construction',
                   'tour_improvement',
                   'time_window_management',
                   'time_window_offering',
                   'time_window_selection',
                   'auction_tour_construction',
                   'auction_tour_improvement',
                   'request_selection',
                   'reopt_and_improve_after_request_selection',
                   'bundle_generation',
                   'bidding',
                   'winner_determination',
                   ])
    print_top_level_stats(df)
    bar_chart(df,
              title='',
              values='sum_profit',
              category='time_window_management',
              color='solution_algorithm',
              facet_col='rad',
              facet_row='n',
              show=True,
              # width=700,
              # height=450,
              html_path=ut.unique_path(ut.output_dir_GH, 'CAHD_#{:03d}.html').as_posix()
              )
    # boxplot(df,
    #         show=True,
    #         category='n',
    #         facet_col=None,
    #         facet_row='rad'
    #         )
