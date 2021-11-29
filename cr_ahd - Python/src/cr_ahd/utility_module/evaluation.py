import datetime as dt
import warnings
from pathlib import Path
from typing import Sequence, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px

import utility_module.utils as ut

labels = {'num_carriers': 'Number of carriers',
          'travel_distance': 'Travel distance',
          'travel_duration': 'Travel duration',
          'rad': 'Radius of Service area',
          'n': 'Number of requests per carrier',
          'num_acc_inf_requests': 'Number of accepted but infeasible requests\n(served by pendulum tour)',
          'sum_profit': 'Sum of profit'
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

ppt_layout = dict(
    # width=31.6,  # cm
    width=4106,  # pixels
    # height=10.1,  # cm
    height=1313,  # pixels
    font_size=12,
    title_font_size=18.6,
)

ppt_half_width_layout = dict(
    # width=15.4,  # cm
    width=2001,  # pixels
    # height=10.1,  # cm
    height=1313,  # pixels
    font_size=12,
    title_font_size=18.6,
)


# =================================================================================================
# PLOTLY
# =================================================================================================
def plot(df: pd.DataFrame,
         values,
         category: Tuple,
         color: Tuple,
         facet_row: Tuple,
         facet_col: Tuple,
         title: str = '',
         width: float = None,
         height: float = None,
         show: bool = True,
         html_path=None
         ):
    for x in category, color, facet_row, facet_col:
        assert isinstance(x, Tuple), 'splitters must be given as tuples'
    if category == ('run',) or df['run'].nunique() == 1:
        bar_chart(df, values, category, color, facet_row, facet_col, title=title, width=width, height=height, show=show,
                  html_path=html_path)
    else:
        box_plot(df, values, category, color, facet_row, facet_col, title=title, width=width, height=height, show=show,
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
        # **ppt_half_width_layout,
        template=template,
        margin=dict(r=500)
    )

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
    fig.update_layout(
        # legend=dict(
        #     orientation="h",
        #     yanchor="top",
        #     y=-0.1,
        #     xanchor="center",
        #     x=0.5),
        # **ppt_half_width_layout,
        template=template,
        margin=dict(r=500)
    )
    if show:
        fig.show(config=config)

    if html_path:
        fig.write_html(html_path, )


def plotly_prepare_df(df, category, color, facet_col, facet_row):
    splitters_dict = dict(category=category,
                          color=color,
                          facet_row=facet_row,
                          facet_col=facet_col, )
    splitter_flat_list = ut.flatten([list(x) for x in splitters_dict.values()])

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

    # force the color column to be treated as a categorical column
    df[splitters_dict['color']] = df[splitters_dict['color']].astype('category')

    # create annotation
    single_val_col, zero_val_col = single_and_zero_value_columns(df, splitter_flat_list)
    annotation = [f"{col}={df[col].dropna().unique()[0]}" for col in single_val_col]
    annotation = '<br>'.join(annotation)
    # annotation = '<br>'.join(('; '.join(x) for x in zip_longest(annotation[::2], annotation[1::2], fillvalue='')))

    return annotation, df, splitters_dict


def single_and_zero_value_columns(df: pd.DataFrame, ignore=None):
    """
    identify the columns that contain a *single* value (not counting None, NaN, etc.)
    or *no* value at all (not counting None, NaN, etc.).
    Zero-value-indices can only happen if all values of that column are None, NaN, etc.

    :param ignore: columns to ignore even if they only have a single value or no value
    :return: two lists of levels from the DataFrame's MultiIndex that contain (1) only a single value and (2) no values
    """
    if ignore is None:
        ignore = []

    if len(df) == 1:
        warnings.warn('Dataframe has only a single row, thus all indices will only have a single value!')

    single_value_indices = []
    zero_value_indices = []
    for column in df.columns:
        if column in ignore:
            continue
        num_unique_values = df[column].nunique(dropna=True)

        # df[column].difference([np.NaN, float('nan'), None, 'None']))
        if num_unique_values == 1:
            single_value_indices.append(column)
        elif num_unique_values == 0:
            zero_value_indices.append(column)
    return single_value_indices, zero_value_indices


def merge_index_levels(df: pd.DataFrame, levels: Sequence, merge_str: str = '-'):
    """merges two or more levels of a pandas Multiindex Dataframe into a single level"""
    df_idx = df.set_index(list(levels))
    df_idx.index = df_idx.index.to_flat_index()
    df_idx.index.name = merge_str.join(levels)
    df_idx = df_idx.reset_index()
    return df_idx


def print_top_level_stats(df: pd.DataFrame, secondary_parameters: List[str]):
    df = df.set_index(keys=df.columns[:33].tolist())

    if len(df) > 1:
        single_val_col, zero_val_col = single_and_zero_value_columns(
            df.reset_index(), ['rad', 'n', 'run', 'carrier_id_', 'solution_algorithm', *secondary_parameters])
        drop_idx = [idx for idx in (single_val_col + zero_val_col) if idx in df.index.names]
        df = df.droplevel(drop_idx, axis=0)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 0):
        if 'carrier_id_' in df.index.names:
            # aggregate the 3 carriers
            solution_df = df.groupby(df.columns.difference(['carrier_id_']),
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
            grouped = solution_df.groupby('solution_algorithm', dropna=False)
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
                for iso, coll in zip(isolated.groupby(grouper, dropna=False),
                                     collaborative.groupby(grouper, dropna=False)):
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
            inconsistent = []
            secondary_parameters_categorical = [param for param in secondary_parameters if 'runtime' not in param]
            print(['run', 'rad', 'n', *secondary_parameters_categorical])
            for name, group in solution_df.groupby(['run', 'rad', 'n', *secondary_parameters_categorical],
                                                   dropna=False):
                grouped = group.groupby('solution_algorithm', dropna=False)
                if all([x in grouped.groups for x in ['CollaborativePlanning', 'IsolatedPlanning']]):
                    isolated = grouped.get_group('IsolatedPlanning')
                    for idx, collaborative in grouped.get_group(
                            'CollaborativePlanning').iterrows():  # .groupby(secondary_parameters, dropna=False):
                        if not all([isolated.squeeze()['sum_profit'] <= collaborative.squeeze()['sum_profit']]):
                            print(f'{name}: consistency check failed!')
                            warnings.warn(f'{name}: Collaborative is worse than Isolated!')
                            inconsistent.append(idx)
                        else:
                            print(f'{name}: consistency check successful')
                else:
                    print(f'{name}: Cannot compare Isolated and Collaborative, '
                          f'since in this group only {grouped.groups.keys()} exist')
            print(f'\n\n{len(inconsistent)} inconsistent solutions:')
            for x in inconsistent:
                print(x)


if __name__ == '__main__':
    path = "C:/Users/Elting/ucloud/PhD/02_Research/02_Collaborative Routing for Attended Home Deliveries/01_Code/data/Output/evaluation_agg_solution_#083.csv"
    df = pd.read_csv(path)

    print_top_level_stats(df, [])
    plot(df,
         title=str(Path(path).name),
         values='sum_profit',
         color=('num_acc_inf_requests',),
         category=('rad',),
         facet_col=(None,),
         facet_row=('n',),
         show=True,
         # width=1000 * 0.85,
         # height=450 * 0.85 * 1.8,
         # html_path=ut.unique_path(Path("C:/Users/Elting/Desktop"), 'CAHD_#{:03d}.html').as_posix()
         )
