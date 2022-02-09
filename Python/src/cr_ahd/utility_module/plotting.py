import webbrowser

import folium
import geopandas as gp
import matplotlib.pyplot as plt
import numpy as np
from folium.plugins import FeatureGroupSubGroup
from geopandas import GeoSeries
from matplotlib.colors import to_hex
from shapely.geometry import Point

from core_module import instance as it, solution as slt
from utility_module import io, utils as ut, geometry as geo

config = dict({'scrollZoom': True})


def read_vienna_districts_shapefile():
    districts = gp.read_file(io.input_dir.joinpath('BEZIRKSGRENZEOGD/BEZIRKSGRENZEOGDPolygon.shp'))
    districts.rename(columns={'SHAPE': 'geometry'}, inplace=True)
    districts['BEZNR'] = districts['BEZNR'].astype(int)
    districts.set_index(['BEZNR', 'NAMEG'], inplace=True)
    districts.sort_index(inplace=True)
    districts = districts.geometry
    districts = districts.apply(geo.flip_coords)
    return districts


def plot_vienna_vrp_solution(instance: it.MDVRPTWInstance, solution: slt.CAHDSolution, service_areas=True):
    num_carriers = instance.num_carriers
    # prepare districts
    carrier_competition = instance.meta['o'] / 100
    vienna_depots = [(instance.vertex_x_coords[x], instance.vertex_y_coords[x]) for x in range(instance.num_carriers)]
    vienna_depots = [Point(*x) for x in vienna_depots]
    vienna_depots = GeoSeries(vienna_depots, crs="EPSG:4326")
    districts = read_vienna_districts_shapefile()
    districts_centroids = districts.geometry.to_crs(epsg=3035).centroid.to_crs(epsg=4326)
    # assign service districts based on (euclidean?) distance d(depot, district_centroid)
    centroid_depot_dist = np.zeros((len(districts_centroids), num_carriers))
    for (idx, name), centroid in districts_centroids.to_crs(epsg=3035).items():
        for jdx, (_, depot) in enumerate(vienna_depots.to_crs(epsg=3035).geometry.items()):
            centroid_depot_dist[idx - 1, jdx] = centroid.distance(depot)
    carrier_district_assignment = centroid_depot_dist.min(axis=1).repeat(num_carriers).reshape(
        centroid_depot_dist.shape)
    carrier_district_assignment = carrier_district_assignment / centroid_depot_dist
    carrier_district_assignment = np.argwhere(carrier_district_assignment >= (1 - carrier_competition))
    carrier_district_assignment = ut.indices_to_nested_lists(carrier_district_assignment[:, 1],
                                                             carrier_district_assignment[:, 0])

    # plot
    vienna_lat, vienna_long = 48.210033, 16.363449
    m = folium.Map((vienna_lat, vienna_long), zoom_start=12, crs='EPSG3857', tiles='Stamen Toner')
    folium.TileLayer('openstreetmap').add_to(m)
    cmap1 = plt.get_cmap('jet', num_carriers)

    # carriers
    keep_in_front_feautres = []
    for carrier in solution.carriers:
        carrier_group = folium.FeatureGroup(f'Carrier {carrier.id_}')
        m.add_child(carrier_group)
        color = to_hex(cmap1(carrier.id_ / num_carriers))

        # depots
        d = depot_marker(instance, carrier, color)
        d.add_to(carrier_group)
        keep_in_front_feautres.append(d)

        # service_areas
        service_area_group = FeatureGroupSubGroup(carrier_group, f'  Service Area {carrier.id_}')
        m.add_child(service_area_group)
        for district_idx in carrier_district_assignment[carrier.id_]:
            poly, name = districts.iloc[district_idx], districts.index[district_idx]
            poly = folium.Polygon(
                locations=poly.exterior.coords, popup=name, color=color, fill_color=color, fill_opacity=0.1
            )
            poly.add_to(service_area_group)

        # tours
        for tour in carrier.tours:
            tour_group = FeatureGroupSubGroup(carrier_group, f'  Tour {tour.id_}')
            m.add_child(tour_group)
            tour_polyline(instance, tour, color).add_to(tour_group)

            # routed requests
            for index, vertex in enumerate(tour.routing_sequence[1:-1], start=1):
                routed_request_marker(instance, carrier, color, index, tour, vertex).add_to(tour_group)

        # unrouted requests
        for request in carrier.unrouted_requests:
            unrouted_request_marker(instance, carrier, color, request).add_to(carrier_group)

    # unassigned requests
    for request in solution.unassigned_requests:
        color = to_hex(cmap1(instance.request_to_carrier_assignment[request] / num_carriers))
        unassigned_request_marker(instance, color, request).add_to(m)

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

    m.keep_in_front(*keep_in_front_feautres)
    path = io.output_dir.joinpath('folium_map.html')
    folium.LayerControl(collapsed=False).add_to(m)
    m.save(path.as_posix())
    webbrowser.open(path)
    return m


def unassigned_request_marker(instance, color, request):
    delivery_vertex = instance.vertex_from_request(request)
    orig_carrier = instance.request_to_carrier_assignment[request]
    return folium.CircleMarker(location=instance.coords(delivery_vertex),
                               tooltip=f'{request}, unassigned',
                               popup=f'Request {request}<br>'
                                     f'UNASSIGNED<br>'
                                     # f'(xy={instance.vertex_x_coords(delivery_vertex)}, {instance.vertex_y_coords(delivery_vertex)} <br>'
                                     f'original carrier={orig_carrier}',
                               radius=5,
                               color=color,
                               weight=2,
                               fill_color='grey',
                               fill_opacity=0.4
                               )


def unrouted_request_marker(instance, carrier, color, request):
    delivery_vertex = instance.vertex_from_request(request)
    return folium.CircleMarker(location=instance.coords(delivery_vertex),
                               tooltip=f'{request}, unrouted',
                               popup=f'Request {request}<br>'
                                     f'UNROUTED<br>'
                                     # f'(xy={instance.vertex_x_coords(delivery_vertex)}, {instance.vertex_y_coords(delivery_vertex)} <br>'
                                     f' carrier={carrier.id_}',
                               radius=5,
                               color=color,
                               fill_color=color,
                               fill_opacity=0.4
                               )


def routed_request_marker(instance, carrier, color, index, tour, vertex):
    request = instance.request_from_vertex(vertex)
    return folium.CircleMarker(
        location=instance.coords(vertex),
        popup=f'Request {request}<br>'
              f'arrival={tour.arrival_time_sequence[index]}<br>'
              f'wait={tour.wait_duration_sequence[index]}<br>'
              f'service={tour.service_time_sequence[index]}<br>'
              f'carrier={carrier.id_}<br>'
              f'tw={instance.time_window(vertex)}',
        tooltip=f'Stop {index}',
        radius=5,
        color=color,
        fill_color=color,
        fill_opacity=1,
    )


def tour_polyline(instance, tour, color):
    total_dur = tour.arrival_time_sequence[-1] - tour.arrival_time_sequence[1] - instance.travel_duration(
        [tour.routing_sequence[0]], [tour.routing_sequence[1]])
    return folium.PolyLine(locations=[instance.coords(i) for i in tour.routing_sequence],
                           popup=f'Tour {tour.id_}<br>'
                                 f'travel_dur={tour.sum_travel_duration}<br>'
                                 f'total_dur={total_dur}<br>'
                                 f'travel_dist={round(tour.sum_travel_distance)}',
                           tooltip=f'Tour {tour.id_}',
                           color=color,
                           weight=4
                           )


def depot_marker(instance, carrier, color):
    return folium.features.RegularPolygonMarker(location=instance.coords(carrier.id_),
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
