import datetime as dt
import random
import webbrowser
from typing import List, Union, Dict

import folium
import geopandas as gp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_hex
from shapely.geometry import Point

import instance_module.geometry as geo
from core_module import instance as it
from utility_module import io, utils as ut
from vienna_data_handling import read_vienna_addresses


# CONVENTION
# when using coordinates, use the order LATITUDE FIRST, LONGITUDE SECOND! All inputs will be rearranged to follow this
# this order. https://en.wikipedia.org/wiki/ISO_6709#Items

def generate_generic_cr_ahd_instance(
        area_km: tuple = (20, 20),
        num_carriers: int = 3,
        dist_center_to_carrier_km=7,
        carrier_competition: float = 0.2,
        num_request: int = 10,
        max_num_tours_per_carrier: int = 3,
        max_vehicle_load: int = 10,
        max_tour_length: int = 1000,
        requests_service_duration: List[dt.timedelta] = (dt.timedelta(minutes=4),) * (3 * 10)
):
    center_x, center_y = area_km[0] / 2, area_km[1] / 2
    # generate evenly positioned depots around the city center
    depots = geo.circle(center_x, center_y, dist_center_to_carrier_km, resolution=num_carriers)
    depots = gp.GeoSeries([Point(lat, long) for lat, long in list(depots.exterior.coords)[:-1]], crs="EPSG:4326")
    service_circles = gp.GeoSeries([geo.circle(p.x, p.y, 7, 24) for p in depots], crs="EPSG:4326")

    raise NotImplementedError
    pass


def generate_vienna_cr_ahd_instance(
        num_carriers: int = 3,
        dist_center_to_carrier_km=7,
        carrier_competition: float = 0,
        num_requests: int = 10,
        max_num_tours_per_carrier: int = 3,
        max_vehicle_load: int = 10,
        max_tour_length: int = 1000,  # TODO set appropriately
        # max_tour_duration: dt.timedelta = dt.timedelta(hours=8),
        requests_service_duration: Union[dt.timedelta, List[dt.timedelta]] = dt.timedelta(minutes=4),
        requests_revenue: List[int] = (99,) * (3 * 10),  # TODO should this depend on some distance?
        requests_load: List[int] = (1,) * (3 * 10),  # TODO should this be randomly distribute?
        # time_horizon_start: dt.datetime = dt.datetime(0, 0, 0, 9),
        # time_horizon_end: dt.datetime = dt.datetime(0, 0, 0, 17),
):
    """

    :param num_carriers:
    :param dist_center_to_carrier_km:
    :param carrier_competition: value between 0 and 1 indicating the degree of service area overlap. 0 means no overlap,
        1 means all carriers serve all districts
    :param num_requests:
    :param max_num_tours_per_carrier:
    :param max_vehicle_load:
    :param max_tour_length:
    :param requests_service_duration:
    :param requests_revenue:
    :param requests_load:
    :return:
    """

    if num_carriers < 2:
        return False

    if isinstance(requests_service_duration, dt.timedelta):
        requests_service_duration = (requests_service_duration,) * (num_carriers * num_requests),

    vienna_lat, vienna_long = 48.210033, 16.363449

    # generate evenly positioned depots around the city center
    depots = geo.circle(vienna_lat, vienna_long, dist_center_to_carrier_km, resolution=num_carriers)
    depots = gp.GeoSeries([Point(lat, long) for lat, long in list(depots.exterior.coords)[:-1]], crs="EPSG:4326")

    districts = read_vienna_districts_shapefile()
    districts_centroids = districts.geometry.to_crs(epsg=3035).centroid.to_crs(epsg=4326)

    # assign service districts based on distance d(depot, district_centroid)
    centroid_depot_dist = np.zeros((len(districts_centroids), num_carriers))
    for (idx, name), centroid in districts_centroids.to_crs(epsg=3035).items():
        for jdx, depot in depots.to_crs(epsg=3035).items():
            centroid_depot_dist[idx - 1, jdx] = centroid.distance(depot)
    district_carrier_assignment = centroid_depot_dist.min(axis=1).repeat(num_carriers).reshape(
        centroid_depot_dist.shape)
    district_carrier_assignment = district_carrier_assignment / centroid_depot_dist
    district_carrier_assignment = np.argwhere(district_carrier_assignment >= (1 - carrier_competition))
    district_carrier_assignment = ut.indices_to_nested_lists(*district_carrier_assignment.T)
    district_carrier_assignment = {idx: district_carrier_assignment[idx - 1] for idx, name in districts.index}

    m = plot_service_areas(district_carrier_assignment, depots, districts, vienna_lat, vienna_long)

    # assign carriers to requests
    durations = pd.read_csv(io.input_dir.joinpath('vienna_durations_1000_#001.csv'), index_col=0)
    vienna_addresses = read_vienna_addresses()  # fixme
    vienna_addresses = vienna_addresses.loc[list(durations.index)]
    vienna_addresses.to_csv(io.input_dir.joinpath('vienna_addresses_1000_#001.csv'))  # fixme remove once saved, then read this subset only
    vienna_addresses['carrier'] = [random.choice(district_carrier_assignment[x])
                                   for x in vienna_addresses['GEB_BEZIRK']]
    # plotting customers
    layer = folium.map.FeatureGroup(f'customers').add_to(m)
    cmap = plt.get_cmap('jet', num_carriers)
    for idx, srs in vienna_addresses.iterrows():
        district = srs['GEB_BEZIRK']
        c = srs['carrier']
        cm = folium.CircleMarker(
            location=(srs.geometry.x, srs.geometry.y),
            radius=3,
            color=to_hex(cmap(c / num_carriers)),
        )
        cm.add_to(layer)

    # write and display
    folium.LayerControl(collapsed=False).add_to(m)  # must be added last!
    path = io.output_dir.joinpath('folium_map.html')
    m.save(path.as_posix())
    webbrowser.open(path)

    # sampling the customers
    data: pd.DataFrame = vienna_addresses
    selected = []
    for name, group in data.groupby(['carrier']):
        s = group.sample(num_requests, replace=False, random_state=42)
        selected.extend(s.index)
    data = data.loc[selected]

    rng = np.random.default_rng()

    # instance = it.MDVRPTWInstance(
    #     id_=f'vienna_test',
    #     max_num_tours_per_carrier=3,
    #     max_vehicle_load=None,
    #     max_tour_length=None,
    #     requests=list(range(len(data))),
    #     requests_initial_carrier_assignment=data['carrier'],
    #     requests_disclosure_time=None,
    #     requests_x=data.geometry.x,
    #     requests_y=data.geometry.y,
    #     requests_revenue=rng.integers(10, 40, len(data)),
    #     requests_service_duration=dt.timedelta(minutes=4),
    #     requests_load=1,
    #     request_time_window_open=None,
    #     request_time_window_close=None,
    #     carrier_depots_x=None,
    #     carrier_depots_y=None,
    #     carrier_depots_tw_open=None,
    #     carrier_depots_tw_close=None,
    #     duration_matrix=None,
    # )

    return


def plot_service_areas(assignment: Dict[int, List], depots, districts, vienna_lat, vienna_long):
    num_carriers = len(depots)
    # plot
    m = folium.Map((vienna_lat, vienna_long), zoom_start=12, crs='EPSG3857', tiles='Stamen Toner')
    folium.TileLayer('openstreetmap').add_to(m)
    # plot service areas
    cmap = plt.get_cmap('jet', num_carriers)
    carrier_layers = [folium.map.FeatureGroup(f'carrier {carrier} service areas', show=True) for carrier in
                      range(num_carriers)]
    for district_idx, carriers in assignment.items():
        for carrier in carriers:
            poly, name = districts.loc[district_idx].squeeze(), districts.index[district_idx - 1]
            poly = folium.Polygon(
                locations=poly.exterior.coords,
                popup=name,
                color=to_hex(cmap(carrier / num_carriers)),
                fill_color=to_hex(cmap(carrier / num_carriers)),
                fill_opacity=0.4
            )
            poly.add_to(carrier_layers[carrier])
    for cl in carrier_layers:
        cl.add_to(m)

    # for carrier in range(num_carriers):
    #     layer = folium.map.FeatureGroup(f'carrier {carrier} service area')
    #     layer.add_to(m)
    #     for district_idx in assignment[assignment[:, 1] == carrier, 0]:
    #         poly = folium.Polygon(
    #             locations=districts[district_idx + 1][0].exterior.coords,
    #             popup=districts[district_idx + 1]['NAMEG'],
    #             color=to_hex(cmap(carrier / num_carriers)),
    #             fill_color=to_hex(cmap(carrier / num_carriers)),
    #             fill_opacity=0.4
    #         )
    #         poly.add_to(layer)
    # plot depots
    depot_markers = []
    for idx, depot in depots.items():
        cm = folium.CircleMarker(location=(depot.x, depot.y),
                                 popup=f'Depot {idx}',
                                 radius=5,
                                 color='black',
                                 fill_color=to_hex(cmap(idx / num_carriers)),
                                 fill_opacity=1,
                                 )
        cm.add_to(m)
        depot_markers.append(cm)

    m.keep_in_front(*depot_markers)

    return m


def read_vienna_districts_shapefile():
    districts = gp.read_file(io.input_dir.joinpath('BEZIRKSGRENZEOGD/BEZIRKSGRENZEOGDPolygon.shp'))
    districts.rename(columns={'SHAPE': 'geometry'}, inplace=True)
    districts['BEZNR'] = districts['BEZNR'].astype(int)
    districts.set_index(['BEZNR', 'NAMEG'], inplace=True)
    districts.sort_index(inplace=True)
    districts = districts.geometry
    districts = districts.apply(geo.flip_coords)
    return districts


if __name__ == '__main__':
    generate_vienna_cr_ahd_instance(num_carriers=3, carrier_competition=0.3)

    # generate a 3d plot to visualize the ratios between radius, center distance and intersection_area of two circles
    """
    r = np.linspace(0, 10, 50, endpoint=False)
    d = np.linspace(0, 10, 50, endpoint=False)
    R, D = np.meshgrid(r, d)
    vfunc = np.vectorize(two_circle_overlap_area)
    I = vfunc(R, D)

    fig = go.Figure(data=[go.Surface(x=R, y=D, z=I, colorscale='Viridis', connectgaps=False)], )
    fig.update_layout(scene=dict(
        xaxis_title='x=radius',
        yaxis_title='y=center distance',
        zaxis_title='z=intersection area',
    ),

        # width=700,
        # margin=dict(r=20, b=10, l=10, t=10)
        )
    fig.show()
    """
    # r = np.linspace(0, 10, 100, endpoint=False)
    # d = 7
    # vfunc = np.vectorize(two_circle_overlap_area)
    # i = vfunc(r, d) / (np.pi * r ** 2)
    # fig = go.Figure(data=[go.Scatter(x=r, y=i, mode='lines+markers')])
    # fig.update_layout(xaxis_title='radius', yaxis_title='intersection_area / circle_area')
    # fig.show()
