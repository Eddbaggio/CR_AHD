import warnings
from typing import Sequence, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px

import src.cr_ahd.utility_module.utils as ut

labels = {'num_carriers': 'Number of Carriers',
          'travel_distance': 'Travel Distance',
          'travel_duration': 'Travel Duration',
          'rad': 'Radius of Service Area',
          'n': 'Number of requests per carrier',
          }

category_orders = {'solution_algorithm': ['IsolatedPlanning',
                                          'CollaborativePlanning',
                                          # 'CentralizedPlanning'
                                          ],
                   'rad': [150,
                           200,
                           300],
                   'solution_algorithm-request_selection': [('IsolatedPlanning', 'None'),
                                                            ('CollaborativePlanning', 'SpatialCluster'),
                                                            ('CollaborativePlanning', 'TemporalRangeCluster')
                                                            ],
                   'num_auction_bundles': [
                       # 50,
                       100,
                       # 200,
                       300,
                       # 500
                   ],
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

    df = drop_single_value_index(df, ut.flatten([category, color, facet_row, facet_col]))
    splitters = dict(category=category, color=color, facet_row=facet_row, facet_col=facet_col, )

    agg_dict = {col: sum for col in df.columns}
    agg_dict['acceptance_rate'] = 'mean'

    # aggregate carriers
    if 'carrier_id_' in df.index.names:
        # aggregate the 3 carriers
        solution_df = df.groupby(df.index.names.difference(['carrier_id_']),
                                 dropna=False).agg({'sum_profit': sum,
                                                    'sum_travel_distance': sum,
                                                    'sum_travel_duration': sum,
                                                    'sum_load': sum,
                                                    'sum_revenue': sum,
                                                    'num_tours': sum,
                                                    'acceptance_rate': 'mean',
                                                    })

    # if any of facet_col, facet_row, color, category is a sequence, merge the levels into one
    already_joined = []
    for k, v in splitters.items():
        if isinstance(v, (List, Tuple)):
            splitters[k] = '-'.join(v)
            if v not in already_joined:
                solution_df = merge_index_levels(solution_df, v)
                already_joined.append(v)

    # group by facets, colors and categories
    px_ready = solution_df.groupby([x for x in set(splitters.values()) if x], dropna=False).agg('mean')

    # prepare for use in plotly express
    px_ready: pd.DataFrame = px_ready.reset_index()
    px_ready = px_ready.round(2)

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
    fig = px.bar(px_ready,
                 x=splitters['category'],
                 y=values,
                 title=title,
                 color=splitters['color'],
                 color_discrete_sequence=ut.univie_colors_100,
                 facet_row=splitters['facet_row'],
                 facet_col=splitters['facet_col'],
                 text=values,
                 template='plotly_white',
                 hover_data=px_ready.columns.values,
                 barmode=barmode,
                 category_orders=category_orders,
                 width=width,
                 height=height,
                 )
    fig.update_yaxes(range=[0, 10000])
    fig.update_xaxes(type='category')
    if show:
        fig.show(config=config)

    if html_path:
        fig.write_html(html_path, )


def drop_single_value_index(df: pd.DataFrame, keep: Sequence):
    """drops all index levels that contain the same unique value for all records OR only one unique value and NaN"""
    if len(df) == 1:
        warnings.warn('Dataframe has only a single row, thus all indices will only have a single value and will '
                      'therefore be dropped')
    for idx_level in df.index.names:
        if idx_level in keep:
            continue
        if len(df.index.unique(idx_level).difference([np.NaN, float('nan'), None, 'None'])) <= 1:
            df = df.droplevel(idx_level, axis=0)
    return df


def merge_index_levels(df: pd.DataFrame, levels: Sequence):
    """merges two or more levels of a pandas Multiindex Dataframe into a single level"""
    df['-'.join(levels)] = df.reset_index(df.index.names.difference(levels)).index.to_flat_index()
    df.set_index('-'.join(levels), append=True, drop=True, inplace=True)
    for level in levels:
        df = df.droplevel(level)
    return df


def print_top_level_stats(carrier_df: pd.DataFrame, secondary_parameters: List[str]):
    if len(carrier_df) > 1:
        carrier_df = drop_single_value_index(carrier_df, ['rad', 'n', 'run', 'carrier_id_', 'solution_algorithm'])

    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 0):

        # aggregate the 3 carriers
        solution_df = carrier_df.groupby(carrier_df.index.names.difference(['carrier_id_']),
                                         dropna=False).agg({'sum_profit': sum,
                                                            'sum_travel_distance': sum,
                                                            'sum_travel_duration': sum,
                                                            'sum_load': sum,
                                                            'sum_revenue': sum,
                                                            'num_tours': sum,
                                                            'acceptance_rate': 'mean',
                                                            })

        print('=============/ stats per solution /=============')
        print(solution_df, '\n')

        # aggregate the 20 runs
        print('=============/ average over runs  /=============')
        print(solution_df.groupby(solution_df.index.names.difference(['run']), dropna=False).agg('mean'), '\n')

        print('=============/ number of solved instances per algorithm /=============')
        for name, group in solution_df.groupby(solution_df.index.names.difference(['rad', 'n', 'run']), dropna=False):
            print(f'{group["num_tours"].astype(bool).sum(axis=0)}/{len(group)} solved by {name}')
        print('\n')

        # # csv
        # bar_chart = solution_df.groupby('n')
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
        for name1, group1 in solution_df.groupby(['solution_algorithm', *secondary_parameters], dropna=False):
            for name2, group2 in solution_df.groupby(['solution_algorithm', *secondary_parameters], dropna=False):
                if name1 == name2:
                    continue
                print(f"{name1}/{name2}")
                for stat in ['sum_profit', 'num_tours']:
                    print(f"{stat}:\t{group1.agg('mean')[stat] / group2.agg('mean')[stat]}")
                print('\n')

        print('=============/ consistency check: collaborative better than isolated? /=============')
        consistency_df = solution_df.droplevel(level=[x for x in solution_df.index.names if x.startswith('auction_')])
        for name, group in consistency_df.groupby(
                consistency_df.index.names.difference(['solution_algorithm', *secondary_parameters]),
                as_index=False,
                dropna=False):
            isolated = group.xs('IsolatedPlanning', level='solution_algorithm').reset_index().iloc[0]
            for index, row in group.xs('CollaborativePlanning', level='solution_algorithm').iterrows():
                if isolated['sum_profit'] > row['sum_profit']:
                    warnings.warn(f'{name}: Collaborative is worse than Isolated!')


if __name__ == '__main__':
    df = pd.read_csv(
        "C:/Users/Elting/ucloud/PhD/02_Research/02_Collaborative Routing for Attended Home "
        "Deliveries/01_Code/data/Output/Gansterer_Hartl/evaluation_carrier_#091.csv",
    )
    df.fillna('None', inplace=True)
    df.set_index(['rad', 'n', 'run', 'carrier_id_'] + ut.solver_config, inplace=True)
    print_top_level_stats(df, ['tour_improvement'])
    bar_chart(df,
              title='',
              values='sum_profit',
              color=['solution_algorithm', 'tour_improvement'],
              category='run', facet_col='rad',
              # category='rad', facet_col=None,
              facet_row='n',
              show=True,
              # width=700,
              # height=450,
              # html_path=ut.unique_path(ut.output_dir_GH, 'CAHD_#{:03d}.html').as_posix()
              )
    # boxplot(df,
    #         show=True,
    #         category='n',
    #         facet_col=None,
    #         facet_row='rad'
    #         )
