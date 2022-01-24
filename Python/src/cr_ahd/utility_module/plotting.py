import webbrowser

import folium
import matplotlib.pyplot as plt
from folium.plugins import FeatureGroupSubGroup
from matplotlib.colors import to_hex

from core_module import instance as it, solution as slt
from utility_module import io

config = dict({'scrollZoom': True})


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
