import folium
from shapely.geometry import Point

from address_retrieval import read_vienna_addresses
import geopandas as gp
import webbrowser
from utility_module import io


def plot_points_in_vienna(point_gdf: gp.GeoDataFrame,
                          path: str = io.output_dir.joinpath('folium_map.html'),
                          show=True):
    """

    """
    assert str(point_gdf.crs) == 'EPSG:4326', f'The GeoDataFrame is in {point_gdf.crs} but must be in EPSG:4326'
    vienna_center = Point(16.363449, 48.210033)
    m = folium.Map(
        (vienna_center.y, vienna_center.x),
        zoom_start=12,
        # tiles='Stamen Toner',
        # crs='EPSG4326',  # NOTE apparently, forcing the crs to (what should be the correct) 4326 messes with the map
    )

    # icon = folium.CustomIcon(icon_image='C:/Users/Elting/Desktop/map-marker-alt-solid.png', icon_size=(50, 50), icon_anchor=(0, 0))
    for x, y, label in zip(point_gdf.geometry.x, point_gdf.geometry.y, point_gdf.NAME):
        folium.CircleMarker(location=(y, x), popup=label).add_to(m)

    # saving and showing
    if path:
        m.save(path.as_posix())
        if show:
            webbrowser.open(path)
    else:
        assert not show, 'If show=True, a path must be given to store the map!'

    return m


population = {
    1010: 16047,
    1020: 105848,
    1030: 91680,
    1040: 33212,
    1050: 55123,
    1060: 31651,
    1070: 31961,
    1080: 25021,
    1090: 41884,
    1100: 207193,
    1110: 104434,
    1120: 97078,
    1130: 54040,
    1140: 93634,
    1150: 76813,
    1160: 103117,
    1170: 57027,
    1180: 51497,
    1190: 73901,
    1200: 86368,
    1210: 167968,
    1220: 195230,
    1230: 110464,
}

if __name__ == '__main__':
    vienna_addresses = read_vienna_addresses(io.input_dir.joinpath('vienna_addresses.csv'))
    weights = [population[plz] for plz in vienna_addresses['PLZ']]
    plot_points_in_vienna(vienna_addresses.sample(100, replace=False, random_state=42, weights=weights))
