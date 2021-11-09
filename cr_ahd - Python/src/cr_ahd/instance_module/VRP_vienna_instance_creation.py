import webbrowser
from typing import List

import folium
from shapely.geometry import Point
from shapely.affinity import translate

from address_retrieval import read_vienna_addresses
import datetime as dt
import math
import geopandas as gp
import pandas as pd
from folium_plot import plot_points_in_vienna
import matplotlib.pyplot as plt

from utility_module import io


def lat_lon_on_circle(center_lat: float, center_lon: float, radius_km: float, angle_degree: float):
    """
    https://stackoverflow.com/questions/7477003/calculating-new-longitude-latitude-from-old-n-meters
    """
    radius_earth_km = 6378
    x_offset_km = math.cos(math.radians(angle_degree)) * radius_km
    y_offset_km = math.sin(math.radians(angle_degree)) * radius_km
    new_latitude = center_lat + (y_offset_km / radius_earth_km) * (180 / math.pi)
    new_longitude = center_lon + (x_offset_km / radius_earth_km) * (180 / math.pi) / math.cos(
        center_lat * 180 / math.pi)
    return new_latitude, new_longitude


def generate_vienna_CR_AHD_instance(
        num_carriers: int = 3,
        num_requests: int = 10,
        max_num_tours_per_carrier: int = 3,
        max_vehicle_load: int = 10,
        max_tour_length: int = 1000,  # TODO set appropriately
        # max_tour_duration: dt.timedelta = dt.timedelta(hours=8),
        requests_service_duration: List[dt.timedelta] = (dt.timedelta(minutes=4),) * (3 * 10),
        requests_revenue: List[int] = (99,) * (3 * 10),  # TODO should this depend on some distance?
        requests_load: List[int] = (1,) * (3 * 10),  # TODO should this be randomly distribute?
        # time_horizon_start: dt.datetime = dt.datetime(0, 0, 0, 9),
        # time_horizon_end: dt.datetime = dt.datetime(0, 0, 0, 17),
        carrier_competition_level: int = 2,  # TODO make this a value that represent properly the service area overlap
):
    if num_carriers < 2:
        return

    degree = 360 / num_carriers
    circle_radius_km = 7  # radius of the circle on which the depots lie, in km
    vienna_center = Point(16.363449, 48.210033)
    names = []
    geometries = []

    for carrier_id in range(num_carriers):
        carrier_degree = degree * carrier_id
        carrier_lat, carrier_lon = lat_lon_on_circle(vienna_center.y, vienna_center.x, circle_radius_km, carrier_degree)
        names.append(f'Depot {carrier_id}')
        geometries.append(Point(carrier_lon, carrier_lat))

    depots = gp.GeoDataFrame({'geometry': geometries, 'NAME': names}, crs='EPSG:4326')
    m = folium.Map(
        (vienna_center.y, vienna_center.x),
        zoom_start=12,
    )
    for x, y, label in zip(depots.geometry.x, depots.geometry.y, depots.NAME):
        folium.CircleMarker(location=(y, x), popup=label, radius=8, fill=True, fill_color='blue').add_to(m)

    # service_areas = gp.GeoDataFrame({'geometry': depots.buffer(0.05, resolution=16), 'depot': names})
    folium.Marker((vienna_center.y, vienna_center.x), 'Center').add_to(m)
    folium.Circle(location=(vienna_center.y, vienna_center.x), radius=7000, color='red').add_to(m)
    buffer = vienna_center.buffer(1)
    path = io.output_dir.joinpath('folium_map.html')
    m.save(path.as_posix())
    webbrowser.open(path)

    # vienna_addresses = read_vienna_addresses()
    # vienna_addresses_with_depot = vienna_addresses.sjoin(service_areas, how='left', predicate='contains')

    return


if __name__ == '__main__':
    generate_vienna_CR_AHD_instance(num_carriers=12)
