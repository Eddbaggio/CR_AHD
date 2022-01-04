import datetime as dt
import random
import webbrowser
from typing import List, Union

import folium
import geopandas as gp
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import to_hex
from shapely.geometry import Point
from shapely.ops import nearest_points
from tqdm import tqdm

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
        vienna_addresses: gp.GeoDataFrame,
        vienna_durations: pd.DataFrame,
        num_carriers: int = 3,
        dist_center_to_carrier_km=7,
        carrier_competition: float = 0,
        num_requests_per_carrier: int = 10,
        max_num_tours_per_carrier: int = 3,
        max_vehicle_load: int = 10,
        max_tour_length: int = 1000,
        requests_service_duration: Union[dt.timedelta, List[dt.timedelta]] = dt.timedelta(minutes=4),
        requests_revenue: Union[float, int, List[float], List[int]] = 1,
        requests_load: Union[float, int, List[float], List[int]] = 1,
        plot=False):
    """

    :param plot:
    :param num_carriers:
    :param dist_center_to_carrier_km:
    :param carrier_competition: value between 0 and 1 indicating the degree of service area overlap. 0 means no overlap,
        1 means all carriers serve all districts
    :param num_requests_per_carrier:
    :param max_num_tours_per_carrier:
    :param max_vehicle_load:
    :param max_tour_length:
    :param requests_service_duration:
    :param requests_revenue:
    :param requests_load:
    :return:
    """

    if num_carriers < 2:
        raise ValueError('Must have at least 2 carriers for a CR_AHD instance')

    assert all(
        vienna_durations.index == vienna_addresses.index), f'Duration matrix and address matrix must share the same index'
    vienna_addresses = vienna_addresses.copy()
    vienna_durations = vienna_durations.copy()

    if isinstance(requests_load, (float, int)):
        requests_load = [requests_load] * (num_carriers * num_requests_per_carrier)

    if isinstance(requests_revenue, (float, int)):
        requests_revenue = [requests_revenue] * (num_carriers * num_requests_per_carrier)

    if isinstance(requests_service_duration, dt.timedelta):
        requests_service_duration = [requests_service_duration] * (num_carriers * num_requests_per_carrier)

    vienna_lat, vienna_long = 48.210033, 16.363449

    # generate evenly positioned depots around the city center
    vienna_depots = geo.circle(vienna_lat, vienna_long, dist_center_to_carrier_km, resolution=num_carriers)
    vienna_depots = gp.GeoSeries([Point(lat, long) for lat, long in list(vienna_depots.exterior.coords)[:-1]],
                                 crs="EPSG:4326")
    # snap the depot positions to the closest true address
    vienna_depots = vienna_depots.apply(
        lambda x: np.where(vienna_addresses.geometry == nearest_points(x, vienna_addresses.unary_union)[1])[0][0])
    vienna_depots = vienna_addresses.iloc[vienna_depots].geometry

    districts = read_vienna_districts_shapefile()
    districts_centroids = districts.geometry.to_crs(epsg=3035).centroid.to_crs(epsg=4326)

    # assign service districts based on (euclidean?) distance d(depot, district_centroid)
    centroid_depot_dist = np.zeros((len(districts_centroids), num_carriers))
    for (idx, name), centroid in districts_centroids.to_crs(epsg=3035).items():
        for jdx, (_, depot) in enumerate(vienna_depots.to_crs(epsg=3035).geometry.items()):
            centroid_depot_dist[idx - 1, jdx] = centroid.distance(depot)
    district_carrier_assignment = centroid_depot_dist.min(axis=1).repeat(num_carriers).reshape(
        centroid_depot_dist.shape)
    district_carrier_assignment = district_carrier_assignment / centroid_depot_dist
    district_carrier_assignment = np.argwhere(district_carrier_assignment >= (1 - carrier_competition))
    district_carrier_assignment = ut.indices_to_nested_lists(*district_carrier_assignment.T)
    district_carrier_assignment = {idx: district_carrier_assignment[idx - 1] for idx, name in districts.index}

    # assign carriers to requests. If more than one carrier serves a district, one of them is chosen randomly
    vienna_requests = vienna_addresses.drop(index=vienna_depots.index)
    vienna_requests['carrier'] = [random.choice(district_carrier_assignment[x])
                                  for x in vienna_requests['GEB_BEZIRK']]

    # sampling the customers
    selected = []
    for name, group in vienna_requests.groupby(['carrier']):
        s = group.sample(num_requests_per_carrier, replace=False)
        selected.extend(s.index)
    vienna_requests = vienna_requests.loc[selected]
    loc_idx = list(vienna_depots.index) + list(vienna_requests.index)
    vienna_durations = vienna_durations.loc[loc_idx, loc_idx]

    # plotting
    if plot:
        plot_service_areas_and_requests(vienna_depots, district_carrier_assignment, districts,
                                        vienna_requests, vienna_lat, vienna_long)

    # generate disclosure times
    vienna_requests['disclosure_time'] = None
    for name, group in vienna_requests.groupby(['carrier']):
        vienna_requests.loc[group.index, 'disclosure_time'] = list(ut.datetime_range(start=ut.ACCEPTANCE_START_TIME,
                                                                                     stop=ut.EXECUTION_START_TIME,
                                                                                     num=len(group),
                                                                                     endpoint=False))

    run = len(list(io.input_dir.glob(f't=vienna+d={dist_center_to_carrier_km}+c={num_carriers}'
                                     f'+n={num_requests_per_carrier:02d}+o={int(carrier_competition * 100):03d}+r=*')))

    return it.MDVRPTWInstance(
        id_=f't=vienna+d={dist_center_to_carrier_km}+c={num_carriers}'
            f'+n={num_requests_per_carrier:02d}+o={int(carrier_competition * 100):03d}+r={run:02d}',
        max_num_tours_per_carrier=max_num_tours_per_carrier,
        max_vehicle_load=max_vehicle_load,
        max_tour_length=max_tour_length,
        requests=list(range(len(vienna_requests))),
        requests_initial_carrier_assignment=vienna_requests['carrier'],
        requests_disclosure_time=vienna_requests['disclosure_time'],
        requests_x=vienna_requests.geometry.x,
        requests_y=vienna_requests.geometry.y,
        requests_revenue=requests_revenue,
        requests_service_duration=requests_service_duration,
        requests_load=requests_load,
        request_time_window_open=[ut.EXECUTION_START_TIME] * len(vienna_requests),
        request_time_window_close=[ut.END_TIME] * len(vienna_requests),
        carrier_depots_x=vienna_depots.geometry.x,
        carrier_depots_y=vienna_depots.geometry.y,
        carrier_depots_tw_open=[ut.EXECUTION_START_TIME] * len(vienna_requests),
        carrier_depots_tw_close=[ut.END_TIME] * len(vienna_requests),
        duration_matrix=np.array(vienna_durations),
    )


def plot_service_areas_and_requests(depots, district_carrier_assignment, districts, vienna_addresses, vienna_lat,
                                    vienna_long):
    num_carriers = len(depots)
    # plot
    m = folium.Map((vienna_lat, vienna_long), zoom_start=12, crs='EPSG3857', tiles='Stamen Toner')
    folium.TileLayer('openstreetmap').add_to(m)
    # plot service areas
    cmap1 = plt.get_cmap('jet', num_carriers)
    carrier_layers = [folium.map.FeatureGroup(f'carrier {carrier} service areas', show=True) for carrier in
                      range(num_carriers)]
    for district_idx, carriers in district_carrier_assignment.items():
        for carrier in carriers:
            poly, name1 = districts.loc[district_idx].squeeze(), districts.index[district_idx - 1]
            poly = folium.Polygon(
                locations=poly.exterior.coords,
                popup=name1,
                color=to_hex(cmap1(carrier / num_carriers)),
                fill_color=to_hex(cmap1(carrier / num_carriers)),
                fill_opacity=0.4
            )
            poly.add_to(carrier_layers[carrier])
    for cl in carrier_layers:
        cl.add_to(m)
    depot_markers = []
    for idx1, (_, depot1) in enumerate(depots.items()):
        cm1 = folium.CircleMarker(location=(depot1.x, depot1.y),
                                  popup=f'Depot {idx1}',
                                  radius=5,
                                  color='black',
                                  fill_color=to_hex(cmap1(idx1 / num_carriers)),
                                  fill_opacity=1,
                                  )
        cm1.add_to(m)
        depot_markers.append(cm1)

    m.keep_in_front(*depot_markers)

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
    m = 1000
    n = 2
    durations = pd.read_csv(io.input_dir.joinpath(f'vienna_{m}_durations_#{n:03d}.csv'), index_col=0)
    addresses = read_vienna_addresses(io.input_dir.joinpath(f'vienna_{m}_addresses_#{n:03d}.csv'))
    for _ in tqdm(range(10)):
        instance = generate_vienna_cr_ahd_instance(
            vienna_addresses=addresses,
            vienna_durations=durations,
            num_carriers=3,
            carrier_competition=0.3,
            num_requests_per_carrier=10,
            plot=True
        )
        instance.write(io.input_dir.joinpath(instance.id_ + '.dat'))
