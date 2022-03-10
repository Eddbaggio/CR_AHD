import cProfile
import datetime as dt
import pstats
import webbrowser
from typing import List, Union, Dict, Sequence
import multiprocessing
from itertools import product

import folium
import geopandas as gp
import numpy as np
import pandas as pd
import tqdm
from matplotlib import pyplot as plt
from matplotlib.colors import to_hex
from shapely.geometry import Point
from shapely.ops import nearest_points

import utility_module.geometry as geo
from core_module import instance as it
from utility_module import io, utils as ut
from vienna_data_handling import read_vienna_addresses


# CONVENTION
# when using coordinates, use the order LATITUDE FIRST, LONGITUDE SECOND! All inputs will be rearranged to follow this
# this order. https://en.wikipedia.org/wiki/ISO_6709#Items

def generate_vienna_cr_ahd_instance(dist_center_to_carrier_km: float,
                                    num_carriers: int,
                                    num_requests_per_carrier: int,
                                    carriers_max_num_tours: int,
                                    carrier_competition: float,
                                    run: int,
                                    max_vehicle_load: int,
                                    max_tour_length: int,
                                    max_tour_duration,
                                    requests_revenue: Union[float, int, List[float], List[int]],
                                    requests_service_duration: Union[dt.timedelta, List[dt.timedelta]],
                                    requests_load: Union[float, int, List[float], List[int]],
                                    plot=False,
                                    save=True):
    rng = np.random.default_rng(
        int(num_carriers + num_requests_per_carrier + carrier_competition * 100 + run + carriers_max_num_tours))

    # choose one of the (address + durations + distances) data sets randomly
    # vienna_durations, vienna_distances, vienna_addresses = matrices[rng.choice(list(range(len(matrices))))]

    m = 1000
    n = rng.choice([1, 2, 3])
    # travel duration in seconds *as floats*
    vienna_durations = pd.read_csv(io.input_dir.joinpath(f'vienna_{m}_durations_#{n:03d}.csv'), index_col=0)
    # distance in meters
    vienna_distances = pd.read_csv(io.input_dir.joinpath(f'vienna_{m}_distances_#{n:03d}.csv'), index_col=0)
    # addresses
    vienna_addresses = read_vienna_addresses(io.input_dir.joinpath(f'vienna_{m}_addresses_#{n:03d}.csv'))

    if num_carriers < 2:
        raise ValueError('Must have at least 2 carriers for a CR_AHD instance')

    assert all(vienna_durations.index == vienna_addresses.index) and all(
        vienna_distances.index == vienna_addresses.index), \
        f'Duration, distance and address matrices must share the same index'

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

    districts = read_vienna_districts_shapefile()  # is in EPSG:4326

    # NOTE previously, I computed the centroids on-the-fly which was computationally expensive. Particularly,
    #  transforming between different EPSG is slow
    # districts_centroids: gp.GeoSeries = districts.geometry.to_crs(epsg=3035).centroid
    # districts_centroids.to_file(io.input_dir.joinpath('districts_centroids_epsg3035.shp'))
    # districts_centroids = districts.geometry.to_crs(epsg=3035).centroid.to_crs(epsg=4326)

    districts_centroids = gp.read_file(io.input_dir.joinpath('districts_centroids_epsg3035.shp'))  # is in EPSG:3035
    districts_centroids.set_index(['BEZNR', 'NAMEG'], inplace=True)
    districts_centroids = districts_centroids.geometry

    vienna_depots_3035 = vienna_depots.to_crs(epsg=3035).geometry

    # assign service districts based on (euclidean?) distance depot -> district_centroid
    centroid_depot_dist = np.zeros((len(districts_centroids), num_carriers))
    for (idx, name), centroid in districts_centroids.items():
        for jdx, (_, depot) in enumerate(vienna_depots_3035.items()):
            centroid_depot_dist[idx - 1, jdx] = centroid.distance(depot)
    district_carrier_assignment = centroid_depot_dist.min(axis=1).repeat(num_carriers).reshape(
        centroid_depot_dist.shape)
    district_carrier_assignment = district_carrier_assignment / centroid_depot_dist
    district_carrier_assignment = np.argwhere(district_carrier_assignment >= (1 - carrier_competition))
    district_carrier_assignment = ut.indices_to_nested_lists(*district_carrier_assignment.T)
    district_carrier_assignment = {idx: district_carrier_assignment[idx - 1] for idx, name in districts.index}

    # assign carriers to requests. If more than one carrier serves a district, one of them is chosen randomly
    vienna_requests = vienna_addresses.drop(index=vienna_depots.index)
    vienna_requests['carrier'] = [rng.choice(district_carrier_assignment[x]) for x in vienna_requests['GEB_BEZIRK']]

    # sampling the customers
    selected = []
    for name, group in vienna_requests.groupby(['carrier']):
        s = group.sample(num_requests_per_carrier, replace=False, random_state=rng.bit_generator)
        selected.extend(s.index)
    vienna_requests = vienna_requests.loc[selected]

    # filter addresses, durations and distances
    loc_idx = list(vienna_depots.index) + list(vienna_requests.index)
    vienna_durations = np.array(vienna_durations.loc[loc_idx, loc_idx])
    vienna_durations = np.array([[dt.timedelta(seconds=j) for j in i] for i in vienna_durations])
    vienna_distances = np.array(vienna_distances.loc[loc_idx, loc_idx])

    # check_triangle_inequality(vienna_durations, True)
    # check_triangle_inequality(vienna_distances, True)

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

    instance = it.CAHDInstance(id_=f't=vienna'
                                      f'+d={dist_center_to_carrier_km}'
                                      f'+c={num_carriers}'
                                      f'+n={num_requests_per_carrier:02d}'
                                      f'+v={carriers_max_num_tours}'
                                      f'+o={int(carrier_competition * 100):03d}'
                                      f'+r={run:02d}',
                               carriers_max_num_tours=carriers_max_num_tours,
                               max_vehicle_load=max_vehicle_load,
                               max_tour_length=max_tour_length,
                               max_tour_duration=max_tour_duration,
                               requests=list(range(len(vienna_requests))),
                               requests_initial_carrier_assignment=list(vienna_requests['carrier']),
                               requests_disclosure_time=list(vienna_requests['disclosure_time']),
                               requests_x=vienna_requests.geometry.x,
                               requests_y=vienna_requests.geometry.y,
                               requests_revenue=requests_revenue,
                               requests_service_duration=requests_service_duration,
                               requests_load=requests_load,
                               request_time_window_open=[ut.EXECUTION_START_TIME] * len(vienna_requests),
                               request_time_window_close=[ut.END_TIME] * len(vienna_requests),
                               carrier_depots_x=vienna_depots.geometry.x,
                               carrier_depots_y=vienna_depots.geometry.y,
                               carrier_depots_tw_open=[ut.EXECUTION_START_TIME] * len(vienna_depots),
                               carrier_depots_tw_close=[ut.END_TIME] * len(vienna_depots),
                               duration_matrix=np.array(vienna_durations),
                               distance_matrix=np.array(vienna_distances))

    if save:
        instance.write_json(io.input_dir.joinpath(instance.id_ + '.json'))

    return instance


def _generate_vienna_cr_ahd_instance_star(kwargs):
    return generate_vienna_cr_ahd_instance(**kwargs),


def mp_instance_gen(list_of_kwargs: Sequence[Dict], num_threads: int):
    """
    Generating instances using multiprocessing.

    :param list_of_kwargs: A Sequence of dicts that can be used as kwargs to the function generate_vienna_cr_ahd_instance
    :param num_threads: number of threads to be used for instance generation
    :return:
    """
    if num_threads == 1:
        instances = []
        for kwargs in tqdm.tqdm(list_of_kwargs):
            instances.append(_generate_vienna_cr_ahd_instance_star(kwargs))
    else:
        n_jobs = len(list_of_kwargs)
        with multiprocessing.Pool(num_threads) as pool:
            instances = list(
                tqdm.tqdm(pool.imap(_generate_vienna_cr_ahd_instance_star, list_of_kwargs), total=n_jobs))
    return instances


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
                fill_opacity=0.2
            )
            poly.add_to(carrier_layers[carrier])
    for cl in carrier_layers:
        cl.add_to(m)
    depot_markers = []
    for idx1, (_, depot1) in enumerate(depots.items()):
        cm1 = folium.features.RegularPolygonMarker(location=(depot1.x, depot1.y),
                                                   number_of_sides=4,
                                                   popup=f'Depot {idx1}',
                                                   radius=10,
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
            radius=5,
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
    """
    # read the OSRM and address data, each of the random data sets contains m locations
    m = 1000
    rng = np.random.default_rng(42)
    matrices = []
    for n in [1, 2, 3]:
        # travel duration in seconds *as floats*
        durations = pd.read_csv(io.input_dir.joinpath(f'vienna_{m}_durations_#{n:03d}.csv'), index_col=0)
        # distance in meters
        distances = pd.read_csv(io.input_dir.joinpath(f'vienna_{m}_distances_#{n:03d}.csv'), index_col=0)
        # addresses
        addresses = read_vienna_addresses(io.input_dir.joinpath(f'vienna_{m}_addresses_#{n:03d}.csv'))
        matrices.append([durations, distances, addresses])"""


    def foo():

        s_dist_center_to_carrier_km = [7]
        s_num_carriers = [3]
        s_num_requests_per_carrier = [10, 25, 50, 100]
        s_carriers_max_num_tours = [1, 3]
        s_carrier_competition = [0, 0.25, 0.5, 1, 0]
        s_run = list(range(20))
        s_max_vehicle_load = [999_999_999]
        s_max_tour_length = [999_999_999]
        s_max_tour_duration = [ut.EXECUTION_TIME_HORIZON.duration]
        s_requests_revenue = [1]
        s_requests_service_duration = [dt.timedelta(minutes=4)]
        s_requests_load = [1]
        s_plot = [False]
        s_save = [True]  # NOTE set to false for testing

        list_of_kwargs = []
        for dist_center_to_carrier_km, num_carriers, num_requests_per_carrier, carriers_max_num_tours, carrier_competition, \
            run, max_vehicle_load, max_tour_length, max_tour_duration, requests_revenue, requests_service_duration, \
            requests_load, plot, save in product(s_dist_center_to_carrier_km,
                                                 s_num_carriers,
                                                 s_num_requests_per_carrier,
                                                 s_carriers_max_num_tours,
                                                 s_carrier_competition,
                                                 s_run,
                                                 s_max_vehicle_load,
                                                 s_max_tour_length,
                                                 s_max_tour_duration,
                                                 s_requests_revenue,
                                                 s_requests_service_duration,
                                                 s_requests_load,
                                                 s_plot,
                                                 s_save
                                                 ):
            list_of_kwargs.append(dict(dist_center_to_carrier_km=dist_center_to_carrier_km,
                                       num_carriers=num_carriers,
                                       num_requests_per_carrier=num_requests_per_carrier,
                                       carriers_max_num_tours=carriers_max_num_tours,
                                       carrier_competition=carrier_competition,
                                       run=run,
                                       max_vehicle_load=max_vehicle_load,
                                       max_tour_length=max_tour_length,
                                       max_tour_duration=max_tour_duration,
                                       requests_revenue=requests_revenue,
                                       requests_service_duration=requests_service_duration,
                                       requests_load=requests_load,
                                       plot=plot,
                                       save=save,
                                       ))
        mp_instance_gen(list_of_kwargs, 6)


    cProfile.run('foo()', io.output_dir.joinpath('instance_gen_stats'))
    # STATS
    p = pstats.Stats(io.output_dir.joinpath('instance_gen_stats').as_posix())
    # remove the extraneous path from all the module names:
    p.strip_dirs()
    # sorts the profile by cumulative time in a function, and then only prints the n most significant lines:
    p.sort_stats('cumtime').print_stats(50)
    # see what functions were looping a lot, and taking a lot of time:
    p.sort_stats('tottime').print_stats(50)  # time spent in a function (excluding time made in calls to sub-functions)
    p.sort_stats('ncalls').print_stats(20)
    # p.print_callers(20)

    """# define the parameters desired for the set of instances that shall be created
    s_num_carriers = [3]
    s_num_requests_per_carrier = [100]
    s_carrier_max_num_tours = [3]
    s_carrier_competition = [0.0, 0.25, 0.5, 0.75, 1.0]
    s_run = list(range(20))
    num_instances = len(s_num_carriers) * \
                    len(s_num_requests_per_carrier) * \
                    len(s_carrier_max_num_tours) * \
                    len(s_carrier_competition) * \
                    len(s_run)

    with tqdm.tqdm(total=num_instances) as pbar:
        for num_carriers in s_num_carriers:
            for num_requests_per_carrier in s_num_requests_per_carrier:
                for carrier_max_num_tours in s_carrier_max_num_tours:
                    for carrier_competition in s_carrier_competition:
                        for run in s_run:
                            durations, distances, addresses = matrices[rng.choice([0, 1, 2])]
                            instance = generate_vienna_cr_ahd_instance(dist_center_to_carrier_km=7,
                                                                       num_carriers=num_carriers,
                                                                       num_requests_per_carrier=num_requests_per_carrier,
                                                                       carriers_max_num_tours=carrier_max_num_tours,
                                                                       carrier_competition=carrier_competition, run=run,
                                                                       max_vehicle_load=1000, max_tour_length=1_000_000,
                                                                       max_tour_duration=ut.EXECUTION_TIME_HORIZON.duration,
                                                                       requests_revenue=1,
                                                                       requests_service_duration=dt.timedelta(
                                                                           minutes=4), requests_load=1, plot=False,
                                                                       save=True)
                            pbar.update()
"""
