from pathlib import Path

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
                 title=f"<b>n</b>: Number of requests per carrier<br>"
                       f"<b>rad</b>: Radius of the carriers' operational area around the depot<br>",
                 color=color,
                 color_discrete_sequence=ut.univie_colors_100,
                 facet_row=facet_row,
                 facet_col=facet_col,
                 text=values,
                 template='plotly_white',
                 hover_data=df.columns.values,
                 barmode=barmode,
                 category_orders={'solution_algorithm': [
                     'IsolatedPlanning',
                     'CollaborativePlanning',
                     'CentralizedPlanning'
                 ]}
                 )

    if show:
        fig.show(config=config)

    if html_path:
        fig.write_html(html_path, )


if __name__ == '__main__':
    df = pd.read_csv(
        "C:/Users/Elting/ucloud/PhD/02_Research/02_Collaborative Routing for Attended Home "
        "Deliveries/01_Code/data/Output/Gansterer_Hartl/evaluation_carrier_#004.csv",
        index_col=['rad', 'n', 'run', 'solution_algorithm', 'carrier_id_'])
    bar_chart(df,
              values='sum_profit',
              category='run',
              color='solution_algorithm',
              facet_col='rad',
              facet_row='n',
              show=True,
              # html_path=ut.unique_path(ut.output_dir_GH, 'CAHD_#{:03d}.html').as_posix()
              )
    # boxplot(df,
    #         show=True,
    #         category='n',
    #         facet_col=None,
    #         facet_row='rad'
    #         )
