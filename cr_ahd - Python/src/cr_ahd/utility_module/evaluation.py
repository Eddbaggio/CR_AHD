import time
import warnings
from pathlib import Path
from typing import Sequence, List, Tuple, Union
from itertools import zip_longest
import numpy as np
import pandas as pd
import plotly.express as px
import datetime as dt

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
              title: str = '',
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

    splitters = dict(category=category,
                     color=color,
                     facet_row=facet_row,
                     facet_col=facet_col, )

    # create annotation
    annotation, _ = single_and_zero_value_indices(df.index, list(splitters.values()))
    annotation = [f"{i}={df.index.unique(i).dropna().difference([np.NaN, float('nan'), None, 'None'])[0]}"
                  for i in annotation]
    annotation = '<br>'.join(annotation)
    # annotation = '<br>'.join(('; '.join(x) for x in zip_longest(annotation[::2], annotation[1::2], fillvalue='')))

    # drop indices with single values or only None values
    svi, zvi = single_and_zero_value_indices(df.index, ut.flatten([category, color, facet_row, facet_col]))
    df.droplevel([*svi, *zvi], axis=0)

    # aggregate carriers if necessary to obtain a df containing data per solution rather than per carrier
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
    else:
        solution_df = df

    # if any of facet_col, facet_row, color, category is a sequence, merge the levels into one
    already_joined = []
    for k, v in splitters.items():
        if isinstance(v, (List, Tuple)):
            if len(v) > 1:
                splitters[k] = '-'.join(v)
                if v not in already_joined:
                    solution_df = merge_index_levels(solution_df, v)
                    already_joined.append(v)
            else:
                splitters[k] = v[0]

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
                 color_discrete_sequence=ut.univie_colors_100 + ut.univie_colors_60,
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
    # add annotation
    fig.add_annotation(
        x=1,
        y=0,
        xanchor='left',
        yanchor='bottom',
        xref='paper',
        yref='paper',
        showarrow=False,
        align='left',
        text=annotation)

    # fig.update_yaxes(range=[0, 10000])
    fig.update_xaxes(type='category')
    if 7 < dt.datetime.now().hour < 20:
        template = 'plotly_white'
    else:
        template = 'plotly_dark'
    fig.update_layout(
        title_font_size=12,
        template=template)

    if show:
        fig.show(config=config)

    if html_path:
        fig.write_html(html_path, )

    # violin plot
    fig = px.box(solution_df.reset_index(),
                 # points='all',
                 x=splitters['category'],
                 y=values,
                 title=title,
                 color=splitters['color'],
                 color_discrete_sequence=ut.univie_colors_100 + ut.univie_colors_60,
                 facet_row=splitters['facet_row'],
                 facet_col=splitters['facet_col'],
                 template='plotly_white',
                 hover_data=px_ready.columns.values,
                 boxmode='group',
                 category_orders=category_orders,
                 width=width,
                 height=height,
                 )
    # add annotation
    # fig.add_annotation(
    #     x=1,
    #     y=0,
    #     xanchor='left',
    #     yanchor='bottom',
    #     xref='paper',
    #     yref='paper',
    #     showarrow=False,
    #     align='left',
    #     text=annotation)

    # fig.update_yaxes(range=[0, 10000])
    fig.update_xaxes(type='category')
    if 7 < dt.datetime.now().hour < 20:
        template = 'plotly_white'
    else:
        template = 'plotly_dark'
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.1,
            xanchor="center",
            x=0.5)
    #     title_font_size=12,
    #     template=template
        )

    if show:
        fig.show(config=config)


def single_and_zero_value_indices(multiindex: pd.MultiIndex, ignore=None):
    """
    identify the indices that contain a single value (not counting None, NaN, etc.)
    or no value (not counting None, NaN, etc.). Zero-value-indices can only happen if all values of that index are
    None, NaN, etc.

    :param ignore: indices to ignore even if they only have a single value or no value
    :return: two lists of levels from the DataFrame's MultiIndex that contain (1) only a single value and (2) no values
    """
    if ignore is None:
        ignore = []

    if len(multiindex) == 1:
        warnings.warn('Dataframe has only a single row, thus all indices will only have a single value!')

    single_value_indices = []
    zero_value_indices = []
    for index in multiindex.names:
        if index in ignore:
            continue
        index_length = len(multiindex.unique(index).difference([np.NaN, float('nan'), None, 'None']))
        if index_length == 1:
            single_value_indices.append(index)
        elif index_length == 0:
            zero_value_indices.append(index)
    return single_value_indices, zero_value_indices


def merge_index_levels(df: pd.DataFrame, levels: Sequence):
    """merges two or more levels of a pandas Multiindex Dataframe into a single level"""
    df['-'.join(levels)] = df.reset_index(df.index.names.difference(levels)).index.to_flat_index()
    df.set_index('-'.join(levels), append=True, drop=True, inplace=True)
    for level in levels:
        df = df.droplevel(level)
    return df


def print_top_level_stats(df: pd.DataFrame, secondary_parameters: List[str]):
    if len(df) > 1:
        drop_indices = single_and_zero_value_indices(
            df.index, ['rad', 'n', 'run', 'carrier_id_', 'solution_algorithm', *secondary_parameters])
        df.droplevel([*drop_indices[0], *drop_indices[1]], axis=0)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 0):
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
        else:
            solution_df = df

        print('=============/ stats per solution /=============')
        print(solution_df, '\n')

        # aggregate the 20 runs
        print('=============/ average over runs  /=============')
        print(solution_df.groupby(solution_df.index.names.difference(['run']), dropna=False).agg('mean'), '\n')

        print('=============/ number of solved instances per algorithm /=============')
        for name, group in solution_df.groupby(solution_df.index.names.difference(['rad', 'n', 'run']), dropna=False):
            print(f'{group["num_tours"].astype(bool).sum(axis=0)}/{len(group)} solved by {name}')
        print('\n')

        # collaboration gain
        if 'solution_algorithm' in solution_df.columns and solution_df['solution_algorithm'].nunique() == 2:
            print('=============/ collaboration gains /=============')
            columns = ['sum_profit', 'sum_travel_distance', 'num_tours', 'runtime_total']
            records = []
            grouped = solution_df.groupby('solution_algorithm')
            isolated = grouped.get_group('IsolatedPlanning')
            collaborative = grouped.get_group('CollaborativePlanning')
            # average overall
            record = dict(name="CollaborativePlanning / IsolatedPlanning")
            for x in columns:
                record[x] = collaborative[x].agg('mean') / isolated[x].agg('mean')
            records.append(record)
            # average by different groupings
            for grouper in [
                ['rad'],
                ['n'],
                secondary_parameters,
                ['rad', *secondary_parameters],
                ['rad', 'n'],
                ['rad', 'n', *secondary_parameters]]:
                for iso, coll in zip(isolated.groupby(grouper), collaborative.groupby(grouper)):
                    iso_name, iso_group = iso
                    coll_name, coll_group = coll
                    assert coll_name == iso_name
                    if len(grouper) > 1:
                        filter_name = list("=".join(x) for x in zip(grouper, (str(y) for y in coll_name)))
                    else:
                        filter_name = f'[\'{grouper[0]}={coll_name}\']'
                    record = dict(name=f'{filter_name}: CollaborativePlanning / IsolatedPlanning')
                    # record = dict(
                    #     name=f'CollaborativePlanning{("=".join(x) for x in zip(grouper, (str(y) for y in coll_name)))} / '
                    #          f'IsolatedPlanning{("=".join(x) for x in zip(grouper, (str(y) for y in iso_name)))}')
                    for x in columns:
                        record[x] = coll_group[x].agg('mean') / iso_group[x].agg('mean')
                    records.append(record)

            df_collaboration_gains = pd.DataFrame.from_records(records, index='name')
            print(df_collaboration_gains)

        if 'CollaborativePlanning' in solution_df.index.get_level_values('solution_algorithm'):
            print('=============/ consistency check: collaborative better than isolated? /=============')
            secondary_parameters_categorical = [param for param in secondary_parameters if 'runtime' not in param]
            print(['run', 'rad', 'n', *secondary_parameters_categorical])
            for name, group in solution_df.groupby(['run', 'rad', 'n', *secondary_parameters_categorical]):
                grouped = group.groupby('solution_algorithm')
                if all([x in grouped.groups for x in ['CollaborativePlanning', 'IsolatedPlanning']]):
                    isolated = grouped.get_group('IsolatedPlanning')
                    for _, collaborative in grouped.get_group('CollaborativePlanning').groupby(secondary_parameters):
                        if not isolated.squeeze()['sum_profit'] <= collaborative.squeeze()['sum_profit']:
                            warnings.warn(f'{name}: Collaborative is worse than Isolated!')
                        else:
                            print(f'{name}: consistency check successful')
                else:
                    print(f'{name}: Cannot compare Isolated and Collaborative, '
                          f'since in this group only {grouped.groups.keys()} exist')


if __name__ == '__main__':
    path = "C:/Users/Elting/ucloud/PhD/02_Research/02_Collaborative Routing for Attended Home " \
           "Deliveries/01_Code/data/Output/Gansterer_Hartl/evaluation_agg_solution_#004.csv"
    df = pd.read_csv(path)
    df.fillna(value={col: 0 for col in df.columns if 'runtime' in col}, inplace=True)
    df.fillna(value='None', inplace=True)

    df = df[df.tour_improvement_time_limit_per_carrier == 1]
    # df = df[df.n == 15]

    df.set_index(['rad', 'n', 'run', ] + ut.solver_config, inplace=True)  # add 'carrier_id_' if agg_level==carrier
    secondary_parameter = 'tour_improvement'
    print_top_level_stats(df, [secondary_parameter])
    # bar_chart(df,
    #           title=str(Path(path).name),
    #           values='sum_travel_distance',
    #           color=['tour_improvement'],
    #           category=None,
    #           facet_col=None,
    #           facet_row=None,
    #           show=True,
    #           width=1000*0.85,
    #           height=450*0.85*1.8,
    #           # html_path=ut.unique_path(Path("C:/Users/Elting/Desktop"), 'CAHD_#{:03d}.html').as_posix()
    #           )
