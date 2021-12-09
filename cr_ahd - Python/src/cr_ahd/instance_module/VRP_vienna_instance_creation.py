import datetime as dt
import math
import webbrowser
from typing import List, Union

import folium
import geopandas as gp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_hex
from shapely.geometry import Point, Polygon

import instance_module.geometry as geo
from utility_module import io
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
    assignment = centroid_depot_dist.min(axis=1).repeat(num_carriers).reshape(centroid_depot_dist.shape)
    assignment = assignment / centroid_depot_dist
    assignment = np.argwhere(assignment >= (1 - carrier_competition))

    plot_service_areas(assignment, depots, districts, num_carriers, vienna_lat, vienna_long)

    vienna_addresses = read_vienna_addresses(nrows=100)
    # for carrier in range(num_carriers):
    #     vienna_addresses[f'carrier_{carrier}'] = vienna_addresses['geometry'].apply(
    #         lambda x: x.within(service_circles[carrier]))

    return


def plot_service_areas(assignment, depots, districts, num_carriers, vienna_lat, vienna_long):
    # plot
    m = folium.Map((vienna_lat, vienna_long), zoom_start=12, crs='EPSG3857', tiles='Stamen Toner')
    # plot service areas
    cmap = plt.get_cmap('jet', num_carriers)
    for carrier in range(num_carriers):
        layer = folium.map.FeatureGroup(f'carrier {carrier} service area')
        layer.add_to(m)
        for district_idx in assignment[assignment[:, 1] == carrier, 0]:
            poly = folium.Polygon(
                locations=districts[district_idx + 1][0].exterior.coords,
                popup=districts[district_idx + 1],
                color=to_hex(cmap(carrier / num_carriers)),
                fill_color=to_hex(cmap(carrier / num_carriers)),
                fill_opacity=0.4
            )
            poly.add_to(layer)
    # plot depots
    for idx, depot in depots.items():
        folium.CircleMarker(location=(depot.x, depot.y),
                            popup=f'Depot {idx}',
                            radius=5,
                            color='black',
                            fill_color=to_hex(cmap(idx / num_carriers)),
                            fill_opacity=1
                            ).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    # write and display
    path = io.output_dir.joinpath('folium_map.html')
    m.save(path.as_posix())
    webbrowser.open(path)


def read_vienna_districts_shapefile():
    districts = gp.read_file(io.input_dir.joinpath('BEZIRKSGRENZEOGD/BEZIRKSGRENZEOGDPolygon.shp'))
    districts.rename(columns={'SHAPE': 'geometry'}, inplace=True)
    districts['BEZNR'] = districts['BEZNR'].astype(int)
    districts.set_index(['BEZNR', 'NAMEG'], inplace=True)
    districts.sort_index(inplace=True)
    districts = districts.geometry
    districts = districts.apply(geo.flip_coords)
    return districts


def plot_coords(objs):
    ax: plt.Axes
    fig, ax = plt.subplots()

    for obj in objs:
        if isinstance(obj, Point):
            coords = list(obj.coords)
            color = 'blue'
        elif isinstance(obj, Polygon):
            coords = list(obj.exterior.coords)
            color = 'red'
            # ax.plot(zip(coords))
        for x, y in coords:
            ax.scatter(x, y, c=color)
    plt.show()


if __name__ == '__main__':
    generate_vienna_cr_ahd_instance(num_carriers=5, carrier_competition=0)

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
