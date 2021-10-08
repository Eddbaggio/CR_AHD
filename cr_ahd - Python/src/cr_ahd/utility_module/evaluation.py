import datetime as dt
import warnings
from pathlib import Path
from typing import Sequence, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px

import utility_module.utils as ut

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
def plot(df: pd.DataFrame,
         values,
         category,
         color,
         facet_row,
         facet_col,
         title: str = '',
         width: float = None,
         height: float = None,
         show: bool = True,
         html_path=None
         ):
    if category == 'run' or df['run'].nunique() == 1:
        bar_chart(df, values, category, color, facet_row, facet_col, title=title, width=width, height=height, show=show,
                  html_path=html_path)
    else:
        bar_chart(df, values, category, color, facet_row, facet_col, title=title, width=width, height=height, show=show,
                  html_path=html_path)


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
    assert 'carrier_id' not in df.index.names
    assert 'tour_id' not in df.index.names

    annotation, df, splitters_dict = plotly_prepare_df(df, category, color, facet_col, facet_row)

    # group by facets, colors and categories
    px_ready = df.groupby([x for x in set(splitters_dict.values()) if x], dropna=False).agg('mean')

    # prepare for use in plotly express
    px_ready: pd.DataFrame = px_ready.reset_index()
    px_ready = px_ready.round(2)

    # bar plot
    fig = px.bar(px_ready,
                 x=splitters_dict['category'],
                 y=values,
                 title=title,
                 color=splitters_dict['color'],
                 color_discrete_sequence=ut.univie_colors_100 + ut.univie_colors_60,
                 facet_row=splitters_dict['facet_row'],
                 facet_col=splitters_dict['facet_col'],
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


def box_plot(df: pd.DataFrame,
             values,
             category,
             color,
             facet_row,
             facet_col,
             title: str,
             height: float,
             width: float,
             points: str = None,
             show: bool = True,
             html_path=None
             ):
    assert 'carrier_id' not in df.index.names
    assert 'tour_id' not in df.index.names

    annotation, df, splitters_dict = plotly_prepare_df(df, category, color, facet_col, facet_row)

    # box plot
    fig = px.box(df,
                 points=points,
                 x=splitters_dict['category'],
                 y=values,
                 title=title,
                 color=splitters_dict['color'],
                 color_discrete_sequence=ut.univie_colors_100 + ut.univie_colors_60,
                 facet_row=splitters_dict['facet_row'],
                 facet_col=splitters_dict['facet_col'],
                 template='plotly_white',
                 hover_data=df.columns.values,
                 boxmode='group',
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
    # fig.update_layout(
    #     legend=dict(
    #         orientation="h",
    #         yanchor="top",
    #         y=-0.1,
    #         xanchor="center",
    #         x=0.5)
    #     title_font_size=12,
    #     template=template
    # )
    if show:
        fig.show(config=config)

    if html_path:
        fig.write_html(html_path, )


def plotly_prepare_df(df, category, color, facet_col, facet_row):
    splitters_dict = dict(category=category,
                          color=color,
                          facet_row=facet_row,
                          facet_col=facet_col, )
    splitter_flat_list = ut.flatten([category, color, facet_row, facet_col])
    # if any of the splitters is a sequence, merge these levels into one
    already_joined = []
    for k, v in splitters_dict.items():
        if isinstance(v, (List, Tuple)):
            if len(v) > 1:
                splitters_dict[k] = '-'.join(v)
                if v not in already_joined:
                    df = merge_index_levels(df, v)
                    already_joined.append(v)
            else:
                splitters_dict[k] = v[0]
    # create annotation
    # svi, _ = single_and_zero_value_indices(multiindex=df.index, ignore=list(splitters_dict.values()))
    # drop indices with single values or only None values
    svi, zvi = single_and_zero_value_indices(df.index, splitter_flat_list)
    annotation = [f"{i}={df.index.unique(i).dropna().difference([np.NaN, float('nan'), None, 'None'])[0]}"
                  for i in svi]
    annotation = '<br>'.join(annotation)
    # annotation = '<br>'.join(('; '.join(x) for x in zip_longest(annotation[::2], annotation[1::2], fillvalue='')))
    df.droplevel([*svi, *zvi], axis=0)
    df.reset_index(inplace=True)
    return annotation, df, splitters_dict


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


def merge_index_levels(df: pd.DataFrame, levels: Sequence, merge_str: str = '-'):
    """merges two or more levels of a pandas Multiindex Dataframe into a single level"""
    df[merge_str.join(levels)] = df.reset_index(df.index.names.difference(levels)).index.to_flat_index()
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
           "Deliveries/01_Code/data/Output/evaluation_agg_solution_#012.csv"
    df = pd.read_csv(path, index_col=list(range(33)))
    df.fillna(value={col: 0 for col in df.columns if 'runtime' in col}, inplace=True)
    df.fillna(value='None', inplace=True)

    secondary_parameter = 'num_int_auctions'
    # print_top_level_stats(df, [secondary_parameter])
    bar_chart(df,
              title=str(Path(path).name),
              values='sum_profit',
              color=['solution_algorithm', secondary_parameter],
              category='rad',
              facet_col=None,
              facet_row='n',
              show=True,
              # width=1000 * 0.85,
              # height=450 * 0.85 * 1.8,
              # html_path=ut.unique_path(Path("C:/Users/Elting/Desktop"), 'CAHD_#{:03d}.html').as_posix()
              )
