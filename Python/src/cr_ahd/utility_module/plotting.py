import webbrowser

import folium
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
from folium.plugins import FeatureGroupSubGroup
from matplotlib.colors import to_hex

import tw_management_module.tw
from core_module import instance as it, solution as slt
from utility_module import utils as ut, io

config = dict({'scrollZoom': True})


def _make_depot_scatter(instance: it.MDVRPTWInstance, solution: slt.CAHDSolution, carrier: int):
    carrier_ = solution.carriers[carrier]
    return go.Scatter(x=[instance.vertex_x_coords[carrier]],
                      y=[instance.vertex_y_coords[carrier]],
                      mode='markers+text',
                      marker=dict(
                          symbol='square',
                          size=15,
                          line=dict(color=ut.univie_colors_100[carrier], width=2),
                          color=ut.univie_colors_60[carrier]),
                      text=carrier,
                      name=f'Carrier {carrier} depot',
                      showlegend=False,
                      # legendgroup=carrier, hoverinfo='x+y'
                      )


def _make_tour_scatter(instance: it.MDVRPTWInstance, solution: slt.CAHDSolution, carrier: int, tour: int):
    tour_ = solution.tours[tour]

    df = pd.DataFrame({
        'id_': tour_.routing_sequence[1:-1],
        'x': [instance.vertex_x_coords[v] for v in tour_.routing_sequence[1:-1]],
        'y': [instance.vertex_y_coords[v] for v in tour_.routing_sequence[1:-1]],
        'request': [instance.request_from_vertex(v) for v in tour_.routing_sequence[1:-1]],
        'type': [instance.vertex_type(v) for v in tour_.routing_sequence[1:-1]],
        'revenue': [instance.vertex_revenue[v] for v in tour_.routing_sequence[1:-1]],
        'tw_open': [instance.tw_open[v] for v in tour_.routing_sequence[1:-1]],
        'tw_close': [instance.tw_close[v] for v in tour_.routing_sequence[1:-1]],
    })
    df['type'] = df['type'].map({'pickup': '+', 'delivery': '-'})
    df['text'] = df['type'] + df['request'].astype(str)
    original_carrier_assignment = [instance.request_to_carrier_assignment[r] for r in
                                   [instance.request_from_vertex(v) for v in tour_.routing_sequence[1:-1]]]

    hover_text = []
    for v in tour_.routing_sequence[1:-1]:
        hover_text.append(
            f'Request {instance.request_from_vertex(v)}</br>{instance.vertex_type(v)}</br>'
            f'Vertex id: {v}</br>'
            f'Revenue: {instance.vertex_revenue[v]}</br>'
            f'TW: {tw_management_module.tw.TimeWindow(instance.tw_open[v], instance.tw_close[v])}</br>'
            f'Tour\'s Travel Distance: {tour_.sum_travel_distance}')

    # colorscale = [(c, ut.univie_colors_100[c]) for c in range(instance.num_carriers)]
    return go.Scatter(
        x=df['x'],
        y=df['y'],
        mode='markers+text',
        marker=dict(
            symbol='circle',
            size=ut.linear_interpolation(df['revenue'].values, 15, 30, min(instance.vertex_revenue),
                                         max(instance.vertex_revenue)),
            line=dict(color=[ut.univie_colors_100[c] for c in original_carrier_assignment], width=2),
            color=ut.univie_colors_60[carrier]),
        text=df['text'],
        name=f'Carrier {carrier}, Tour {tour}',
        hovertext=hover_text
        # legendgroup=carrier, hoverinfo='x+y'
    )


def _make_unrouted_scatter(instance: it.MDVRPTWInstance, solution: slt.CAHDSolution, carrier: int):
    carrier_ = solution.carriers[carrier]
    unrouted_vertices = ut.flatten([list(instance.pickup_delivery_pair(r)) for r in carrier_.unrouted_requests])

    df = pd.DataFrame({
        'id_': unrouted_vertices,
        'x': [instance.vertex_x_coords[v] for v in unrouted_vertices],
        'y': [instance.vertex_y_coords[v] for v in unrouted_vertices],
        'request': [instance.request_from_vertex(v) for v in unrouted_vertices],
        'type': [instance.vertex_type(v) for v in unrouted_vertices],
        'revenue': [instance.vertex_revenue[v] for v in unrouted_vertices],
        'tw_open': [instance.tw_open[v] for v in unrouted_vertices],
        'tw_close': [instance.tw_close[v] for v in unrouted_vertices],
        'original_carrier_assignment': [instance.request_to_carrier_assignment[r] for r in
                                        [instance.request_from_vertex(v) for v in unrouted_vertices]]})
    df['type'] = df['type'].map({'pickup': '+', 'delivery': '-'})
    df['text'] = df['type'] + df['request'].astype(str)

    hover_text = []
    for v in unrouted_vertices:
        hover_text.append(
            f'Request {instance.request_from_vertex(v)}</br>{instance.vertex_type(v)}</br>'
            f'Vertex id: {v}</br>'
            f'Revenue: {instance.vertex_revenue[v]}</br>'
            f'TW: [{instance.tw_open[v]} - {instance.tw_close[v]}]</br>'
        )

    return go.Scatter(
        x=df['x'], y=df['y'], mode='markers+text',
        marker=dict(
            symbol=['circle'] * len(unrouted_vertices),
            size=ut.linear_interpolation(df['revenue'].values, 15, 30, min(instance.vertex_revenue),
                                         max(instance.vertex_revenue)),
            line=dict(color=[ut.univie_colors_100[c] for c in df['original_carrier_assignment']], width=2),
            color=ut.univie_colors_60[carrier]),
        text=df['text'],
        textfont=dict(color='red'),
        name=f'Carrier {carrier}, unrouted',
        hovertext=hover_text
        # legendgroup=carrier, hoverinfo='x+y'
    )


def _make_unassigned_scatter(instance: it.MDVRPTWInstance, solution: slt.CAHDSolution):
    vertex_id = ut.flatten(
        [[p, d] for p, d in [instance.pickup_delivery_pair(r) for r in solution.unassigned_requests]])
    df = pd.DataFrame({
        'id_': vertex_id,
        'x': [instance.vertex_x_coords[v] for v in vertex_id],
        'y': [instance.vertex_y_coords[v] for v in vertex_id],
        'request': [instance.request_from_vertex(v) for v in vertex_id],
        'type': [instance.vertex_type(v) for v in vertex_id],
        'revenue': [instance.vertex_revenue[v] for v in vertex_id],
        'original_carrier_assignment': [instance.request_to_carrier_assignment[r] for r in
                                        [instance.request_from_vertex(v) for v in vertex_id]]
    })
    df['type'] = df['type'].map({'pickup': '+', 'delivery': '-'})
    df['text'] = df['type'] + df['request'].astype(str)

    hover_text = []
    for v in vertex_id:
        hover_text.append(
            f'Request {instance.request_from_vertex(v)}</br>{instance.vertex_type(v)}</br>'
            f'Vertex id: {v}</br>'
            f'Revenue: {instance.vertex_revenue[v]}</br>'
            f'TW: [{instance.tw_open[v]} - {instance.tw_close[v]}]</br>'
        )

    return go.Scatter(
        x=df['x'], y=df['y'], mode='markers+text',
        marker=dict(
            symbol=['circle'] * len(df),
            size=ut.linear_interpolation(df['revenue'].values, 15, 30, min(instance.vertex_revenue),
                                         max(instance.vertex_revenue)),
            line=dict(color=[ut.univie_colors_100[c] for c in df['original_carrier_assignment']],
                      width=2),
            color='white'),
        text=df['text'],
        name=f'Unassigned',
        hovertext=hover_text
    )


def _make_tour_edges(instance: it.MDVRPTWInstance, solution: slt.CAHDSolution, carrier: int, tour: int):
    tour_ = solution.tours[tour]
    # creating arrows as annotations
    directed_edges = []
    for i in range(len(tour_.routing_sequence) - 1):
        from_vertex = tour_.routing_sequence[i]
        to_vertex = tour_.routing_sequence[i + 1]
        arrow_tail = (instance.vertex_x_coords[from_vertex], instance.vertex_y_coords[from_vertex])
        arrow_head = (instance.vertex_x_coords[to_vertex], instance.vertex_y_coords[to_vertex])
        min_rev = min(instance.vertex_revenue)
        max_rev = max(instance.vertex_revenue)

        anno = go.layout.Annotation(
            x=arrow_head[0], y=arrow_head[1], ax=arrow_tail[0], ay=arrow_tail[1],
            xref='x', yref='y', axref='x', ayref='y',
            text="",
            showarrow=True, arrowhead=2, arrowsize=2, arrowwidth=1,
            arrowcolor=ut.univie_colors_100[carrier],
            standoff=ut.linear_interpolation([instance.vertex_revenue[to_vertex]], 15, 30, min_rev, max_rev)[0] / 2,
            startstandoff=ut.linear_interpolation([instance.vertex_revenue[from_vertex]], 15, 30, min_rev, max_rev)[
                              0] / 2,
            name=f'Carrier {carrier}, Tour {tour_}')
        directed_edges.append(anno)
    return directed_edges


def _add_carrier_solution(fig: go.Figure, instance, solution: slt.CAHDSolution, carrier_id: int):
    carrier = solution.carriers[carrier_id]
    scatter_traces = []
    edge_traces = []

    depot_scatter = _make_depot_scatter(instance, solution, carrier_id)
    fig.add_trace(depot_scatter)
    scatter_traces.append(depot_scatter)

    for tour_id in [tour.id_ for tour in carrier.tours]:
        tour_scatter = _make_tour_scatter(instance, solution, carrier_id, tour_id)
        fig.add_trace(tour_scatter)
        scatter_traces.append(tour_scatter)

        edges = _make_tour_edges(instance, solution, carrier_id, tour_id)
        for edge in edges:
            fig.add_annotation(edge)
        edge_traces.append(edges)

    if carrier.unrouted_requests:
        unrouted_scatter = _make_unrouted_scatter(instance, solution, carrier_id)
        fig.add_trace(unrouted_scatter)
        scatter_traces.append(unrouted_scatter)

    return scatter_traces, edge_traces


def plot_solution_2(instance: it.MDVRPTWInstance, solution: slt.CAHDSolution, title='', tours: bool = True,
                    time_windows: bool = True, arrival_times: bool = True, service_times: bool = True,
                    show: bool = False):
    fig = go.Figure()
    # [[scatter tour 0, scatter tour 1], [scatter tour 0, scatter tour 1, scatter tour 2], ...]
    scatters = []
    # [carrier 0:
    #   tour 0: [[edge_0, edge_1, edge_2, ...],
    #   tour 1: [edge_0, edge_1, edge_2, ...], ],
    # carrier 1:
    #   tour 0: [[edge_0, edge_1, edge_2, ...],
    #   tour 1: [edge_0, edge_1, edge_2, ...], ]
    # ]
    edges = []

    for c in range(len(solution.carriers)):
        scatter_traces, edge_traces = _add_carrier_solution(fig, instance, solution, c)
        scatters.append(scatter_traces)
        edges.append(edge_traces)

    if solution.unassigned_requests:
        unassigned_scatter = _make_unassigned_scatter(instance, solution)
        fig.add_trace(unassigned_scatter)
        scatters.append(unassigned_scatter)

    # custom buttons to hide edges
    button_dicts = []
    for c in range(len(solution.carriers)):
        ahd_solution = solution.carriers[c]
        for t in range(len(ahd_solution.tours)):
            button_dicts.append(
                dict(label=f'Carrier {c}, Tour {t}',
                     method='update',
                     args=[dict(visible=[True]),
                           dict(annotations=edges[c][t])
                           ],
                     # toggle on/off buttons -> not working as intended
                     # args2=[dict(visible=[True]),
                     #        dict(annotations=edges)
                     #        ],
                     )
            )
    button_dicts.append(
        dict(label=f'All',
             method='update',
             args=[dict(visible=[True]),
                   dict(annotations=ut.flatten(edges))
                   ],

             )
    )

    fig.update_layout(updatemenus=
                      [dict(type='buttons',
                            y=c / len(solution.carriers),
                            yanchor='auto',
                            active=-1,
                            buttons=button_dicts
                            )
                       ],
                      title=title,
                      # height=1200,
                      # width=1600
                      )
    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1
    )

    if show:
        fig.show(config=config)


def plot_vienna_vrp_solution(instance: it.MDVRPTWInstance, solution: slt.CAHDSolution):
    num_carriers = instance.num_carriers
    # plot
    vienna_lat, vienna_long = 48.210033, 16.363449
    m = folium.Map((vienna_lat, vienna_long), zoom_start=12, crs='EPSG3857', tiles='Stamen Toner')
    folium.TileLayer('openstreetmap').add_to(m)
    folium.TileLayer()
    cmap1 = plt.get_cmap('jet', num_carriers)

    for carrier in solution.carriers:
        carrier_group = folium.FeatureGroup(f'Carrier {carrier.id_}')
        m.add_child(carrier_group)
        color = to_hex(cmap1(carrier.id_ / num_carriers))
        # depots
        d = folium.features.RegularPolygonMarker(location=instance.coords(carrier.id_),
                                                 number_of_sides=4,
                                                 popup=f'Depot of carrier {carrier.id_}<br>'
                                                       f'dur={carrier.sum_travel_duration()}<br>'
                                                       f'dist={round(carrier.sum_travel_distance())}',
                                                 tooltip=f'Depot {carrier.id_}',
                                                 rotation=45,
                                                 radius=10,
                                                 color='black',
                                                 fill_color=color,
                                                 fill_opacity=1,
                                                 )
        d.add_to(carrier_group)
        # tours
        for tour in carrier.tours:
            tour_group = FeatureGroupSubGroup(carrier_group, f'Tour {tour.id_}')
            m.add_child(tour_group)
            folium.PolyLine(locations=[instance.coords(i) for i in tour.routing_sequence],
                            popup=f'Tour {tour.id_}<br>'
                                  f'dur={tour.sum_travel_duration}<br>'
                                  f'dist={round(tour.sum_travel_distance)})',
                            tooltip=f'Tour {tour.id_}',
                            color=color,
                            weight=8
                            ).add_to(tour_group)

            # routed requests
            for index, vertex in enumerate(tour.routing_sequence[1:-1], start=1):
                request = instance.request_from_vertex(vertex)
                folium.CircleMarker(
                    location=instance.coords(vertex),
                    popup=f'Request {request}<br>'
                          f'arrival={tour.arrival_time_sequence[index]}'
                    # f'x,y={instance.vertex_x_coords[vertex], instance.vertex_y_coords[vertex]}<br>'
                          f'carrier={carrier.id_}<br>'
                          f'tw={instance.time_window(vertex)}',
                    tooltip=f'Stop {index}',
                    radius=5,
                    color=color,
                    fill_color=color,
                    fill_opacity=1,
                ).add_to(tour_group)

        # unrouted requests
        for request in carrier.unrouted_requests:
            delivery_vertex = instance.vertex_from_request(request)
            r = folium.CircleMarker(location=instance.coords(delivery_vertex),
                                    tooltip=f'{request}',
                                    popup=f'Request {request}(xy={instance.coords(delivery_vertex)}, carrier={carrier.id_}',
                                    radius=5,
                                    color=color,
                                    fill_color=color,
                                    fill_opacity=0.4
                                    )
            r.add_to(carrier_group)

    # unassigned requests
    for request in solution.unassigned_requests:
        delivery_vertex = instance.vertex_from_request(request)
        orig_carrier = instance.request_to_carrier_assignment[request]
        r = folium.CircleMarker(location=instance.coords(delivery_vertex),
                                tooltip=f'{request}, unassigned',
                                popup=f'Request {request}(xy={instance.coords(delivery_vertex)}, '
                                      f'original carrier={orig_carrier}',
                                radius=5,
                                color=to_hex(cmap1(orig_carrier / num_carriers)),
                                weight=2,
                                fill_color='grey',
                                fill_opacity=0.4
                                )
        r.add_to(m)

    # totals
    folium.features.RegularPolygonMarker(location=(48.261738, 16.280746),
                                         number_of_sides=5,
                                         popup=f'Total Duration: {solution.sum_travel_duration()}<br>'
                                               f'Total Distance: {round(solution.sum_travel_distance(), 2)}<br>'
                                               f'Number of Tours: {solution.num_tours()}<br>'
                                               f'Number of Pendulum Tours: {solution.num_pendulum_tours()}<br>'
                                               f'Avg acceptance rate: {solution.avg_acceptance_rate()}',
                                         tooltip='Totals',
                                         fill_color='blue',
                                         ).add_to(m)

    path = io.output_dir.joinpath('folium_map.html')
    layer_control = folium.LayerControl(collapsed=False).add_to(m)
    m.save(path.as_posix())
    webbrowser.open(path)
    return m
