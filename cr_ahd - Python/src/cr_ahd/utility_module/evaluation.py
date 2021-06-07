from pathlib import Path

import pandas as pd
import plotly.express as px

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
              values: str = 'sum_profit',
              category='solution_algorithm',
              color='solution_algorithm',
              facet_row='n',
              facet_col='rad',
              sum_by=['carrier_id_', 'tour_id_'],
              mean_by=['run'],
              barmode='group',
              show: bool = True,
              html_path=None,
              ):
    """

    :param df: multi-index dataframe
    :return:
    """
    df = df.droplevel(['id_', 'dist', 'num_carriers'])
    multiindex = df.index.names
    # sum up values for sum_by variables (e.g. tours or carriers)
    df = df.groupby(list(set(multiindex) - set(sum_by))).sum()
    # compute mean, e.g. over all the different runs
    if mean_by:
        df = df.groupby(list(set(multiindex) - set(sum_by) - set(mean_by))).mean()
    # prepare for use in plotly express
    df: pd.DataFrame = df.reset_index()
    df = df.round(2)

    # hover text
    hover_text = []
    if facet_row is None:
        hover_text.append('n')
    if facet_col is None:
        hover_text.append('rad')
    if not mean_by:
        hover_text.append('run')

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
                 category_orders={'solution_algorithm': ['IsolatedPlanningNoTW',
                                                         'CollaborativePlanningNoTW',
                                                         'IsolatedPlanning',
                                                         'CollaborativePlanning',
                                                         ]}
                 )

    if show:
        fig.show(config=config)

    if html_path:
        fig.write_html(html_path, )


'''
def plotly_bar_plot(solomon_list: List, attributes: List[str], ):
    df: pd.DataFrame = combine_eval_files(solomon_list)[attributes]

    for attr in attributes:
        attr_df = df[attr].unstack('solution_algorithm').reset_index('rand_copy', drop=True).groupby(level=[0, 1])
        attr_df = attr_df.agg(lambda x: x.mean())  # workaround to aggregate also timedelta values
        if attr == 'duration':
            for col in attr_df:
                attr_df[col] = attr_df[col] + pd.to_datetime('1970/01/01')
        fig = go.Figure()
        colors = itertools.cycle([ut.univie_colors_100[0]] + ut.univie_colors_100[2:])
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
        for v_line_x in np.arange(-0.5, 2 + len(solomon_list), 2):
            fig.add_vline(x=v_line_x, line_width=1, line_color="grey")
        fig.update_layout(title=f'Mean {labels[attr]}',
                          xaxis_title=f'{labels["num_carriers"]} // {labels["solomon_base"]}',
                          yaxis_title=f'{labels[attr]}',
                          # line break with <br>, but then its outside the plot, needed to adjust margins then
                          template='plotly_white',
                          uniformtext_minsize=14, uniformtext_mode='show')

        # multicategory axis not supported by plotly express
        # fig = px.bar(attr_df, x=attr_df.index.names, y=attr_df.columns)
        path = ut.path_output_custom.joinpath(f'plotly_bar_plot_{attr}.html')
        fig.write_html(str(path), auto_open=True)
        # path = ut.path_output_custom.joinpath(f'plotly_bar_plot_{attr}.svg')
        # fig.write_image(str(path))
    pass'''

if __name__ == '__main__':
    df = pd.read_csv(
        "C:/Users/Elting/ucloud/PhD/02_Research/02_Collaborative Routing for Attended Home Deliveries/01_Code/data/Output/Gansterer_Hartl/evaluation_#014.csv",
        index_col=[0, 1, 2, 3, 4, 5, 6, 7, 8])
    bar_chart(df,
              category='run',
              mean_by=None,
              show=True,
              # html_path=ut.unique_path(ut.output_dir_GH, 'CAHD_#{:03d}.html').as_posix()
              )
