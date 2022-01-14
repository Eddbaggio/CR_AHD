import re
import time
import warnings;
from pathlib import Path

import geopandas as gp
import numpy as np
import pandas as pd
import requests
import tqdm
from shapely.geometry import Point
from tqdm import trange

from utility_module import io

# CONVENTION
# when using coordinates, use the order LATITUDE FIRST, LONGITUDE SECOND! All inputs will be rearranged to follow this
# this order. https://en.wikipedia.org/wiki/ISO_6709#Items

warnings.filterwarnings('ignore', message='.*initial implementation of Parquet.*')

district_zip_codes = [1010, 1020, 1030, 1040, 1050, 1060, 1070, 1080, 1090, 1100, 1110, 1120, 1130, 1140, 1150, 1160,
                      1170, 1180, 1190, 1200, 1210, 1220, 1230, ]

vienna_pop_per_district = {
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


def read_vienna_addresses(path=io.input_dir.joinpath('vienna_addresses.csv'),
                          EPSG=4326,
                          **kwargs
                          ):
    """

    :param path:
    :param EPSG:
    :param kwargs: anything accepted by pd.read_csv/read_feather
    :return:
    """
    t = time.time()
    if path.suffix == '.csv':
        dtype = {'FID': 'string',
                 'ABSCHNITT_ZUGANG_STR': 'int64',
                 'ACD': 'int64',
                 'AP_TIMESTAMP_MODIFY': 'string',
                 'BST_BEZEICHNUNG': 'float64',
                 'BST_NUMMER': 'string',
                 'BST_NUMMER_ALPHA': 'string',
                 'BST_NUMMER_NUM': 'float64',
                 'BST_TRENNZEICHEN': 'float64',
                 'BST_TRENNZ_KURZTEXT': 'string',
                 'EDGEOBJECTID_ZUGANG_STR': 'int64',
                 'EXTERNALORGCODE': 'int64',
                 'GEBADR_ANZEIGEN_J_N': 'int64',
                 'GEBADR_REGIONALCODE': 'string',
                 'GEB_BAUBLOCK': 'int64',
                 'GEB_BAUBLOCK_ID': 'int64',
                 'GEB_BEZEICHNUNG': 'string',
                 'GEB_BEZIRK': 'int64',
                 'GEB_NAMECAT': 'int64',
                 'GEB_NUMMER': 'string',
                 'GEB_NUMMER_ALPHA': 'string',
                 'GEB_NUMMER_NUM': 'float64',
                 'GEB_OBJEKTID': 'int64',
                 'GEB_STATUS': 'string',
                 'GEB_STATUS_KURZTEXT': 'string',
                 'GEB_TIMESTAMP_MODIFY': 'string',
                 'GEB_TRENNZEICHEN': 'int64',
                 'GEB_TRENNZ_KURZTEXT': 'string',
                 'GEB_WINKEL': 'float64',
                 'GEB_X': 'float64',
                 'GEB_Y': 'float64',
                 'HAUPT_IDENT_PSEUDO': 'string',
                 'LAGE_ZU_ADR': 'string',
                 'LAGE_ZU_ADR_TXT': 'string',
                 'NAME': 'string',
                 'NAME_BST': 'string',
                 'NAME_BST_GEB': 'string',
                 'NAME_GEB': 'string',
                 'NAME_ONR': 'string',
                 'NAME_ONR_BST_GEB': 'string',
                 'NAME_STR': 'string',
                 'OBJECTID': 'int64',
                 'OBJEKT_ID': 'int64',
                 'OBJEKTID_NAME_STR': 'int64',
                 'ON_BIS_ALPHA': 'string',
                 'ON_BIS_NUM': 'float64',
                 'ONR_BIS': 'string',
                 'ONR_VON': 'string',
                 'ON_VON_ALPHA': 'string',
                 'ON_VON_NUM': 'float64',
                 'PLATZKENNZEICHEN': 'float64',
                 'PLZ': 'int64',
                 'SCD_NAME_STR': 'int64',
                 'SCD_ZUGANG_STR': 'int64',
                 'SE_ANNO_CAD_DATA': 'float64',
                 'geometry': 'string',  # geometry column
                 'STRABS_ZUGANG_STR': 'int64',
                 'STRASSENSEITE': 'float64',
                 'TOPOGR_NAME_STR': 'string',
                 'UEBERACD': 'float64',
                 'ZUGADR_ONR_KGNUM': 'float64',
                 'ZUGADR_ONR_STAMMNR': 'float64',
                 'ZUGADR_ONR_TEILNR': 'float64',
                 'ZUGADR_REGIONALCODE': 'string',
                 'ZUGADR_TYP': 'int64',
                 'ZUGANG_MOEGLICH_J_N': 'int64',
                 'ZUG_BEZIRK': 'int64',
                 'ZUG_OBJEKTID': 'int64',
                 'ZUG_TIMESTAMP_MODIFY': 'string',
                 'ZUG_TRENNZEICHEN': 'int64',
                 'ZUG_TRENNZ_KURZTEXT': 'string',
                 'ZUG_WINKEL': 'float64',
                 'ZUG_X': 'float64',
                 'ZUG_Y': 'float64',
                 'ABLAUFDATUM': 'string'}
        df = pd.read_csv(path, index_col='FID', dtype=dtype, **kwargs)
        # parse datetime columns
        for date_col in df.columns:
            if 'TIMESTAMP' in date_col:
                df[date_col] = pd.to_datetime(df[date_col])
        # transform access points to proper shapely Point objects
        df['geometry'] = df['geometry'].apply(lambda x: Point(float(s) for s in re.findall(r'-?\d+\.?\d*', x)))
        # turn into geodataframe
        gdf = gp.GeoDataFrame(df, crs=f'EPSG:4326')
        print(f'reading and preparing {path.name} took {time.time() - t} seconds')

    elif path.suffix == '.feather':
        warnings.warn('Writing and Reading in feather has caused issues before: values are not preserved!')
        gdf = gp.read_feather(path)
        print(f'reading {path.name} took {time.time() - t} seconds')

    # filter out Train Stations, Museums, public toilets, Schrebergärten, ...
    gdf = gdf[gdf['ZUGADR_TYP'] == 0]
    return gdf


def _query_vienna_addresses_online(write_path: Path = io.input_dir.joinpath('vienna_addresses.csv'),
                                   districts=range(1, 24),
                                   EPSG=4326
                                   ):
    """
    Obtain detailed address information for the city of Vienna provided by https://www.data.gv.at. Data must be queried
    for each district individually but can obviously be stitched together afterwards. Downloading the data may take some
    time.
    """
    vienna_addresses = gp.GeoDataFrame()
    for district in districts:
        print(f'Querying vienna address data for district {district}/23')
        # query the raw data
        d = dict(
            service='WFS',
            request='GetFeature',
            version='1.1.0',
            typeName='ogdwien:ADRESSENOGD',
            srsName=f'EPSG:{EPSG}',
            outputFormat='csv',
            cql_filter=f"GEB_BEZIRK='{district:02d}'",
        )
        url = f"https://data.wien.gv.at/daten/geo?{'&'.join([f'{k}={v}' for k, v in d.items()])}"
        district_df = pd.read_csv(
            url,
            index_col='FID',
            dtype={'FID': 'string',
                   'ABSCHNITT_ZUGANG_STR': 'int64',
                   'ACD': 'int64',
                   'AP_TIMESTAMP_MODIFY': 'string',
                   'BST_BEZEICHNUNG': 'float64',
                   'BST_NUMMER': 'string',
                   'BST_NUMMER_ALPHA': 'string',
                   'BST_NUMMER_NUM': 'float64',
                   'BST_TRENNZEICHEN': 'float64',
                   'BST_TRENNZ_KURZTEXT': 'string',
                   'EDGEOBJECTID_ZUGANG_STR': 'int64',
                   'EXTERNALORGCODE': 'int64',
                   'GEBADR_ANZEIGEN_J_N': 'int64',
                   'GEBADR_REGIONALCODE': 'string',
                   'GEB_BAUBLOCK': 'int64',
                   'GEB_BAUBLOCK_ID': 'int64',
                   'GEB_BEZEICHNUNG': 'string',
                   'GEB_BEZIRK': 'int64',
                   'GEB_NAMECAT': 'int64',
                   'GEB_NUMMER': 'string',
                   'GEB_NUMMER_ALPHA': 'string',
                   'GEB_NUMMER_NUM': 'float64',
                   'GEB_OBJEKTID': 'int64',
                   'GEB_STATUS': 'string',
                   'GEB_STATUS_KURZTEXT': 'string',
                   'GEB_TIMESTAMP_MODIFY': 'string',
                   'GEB_TRENNZEICHEN': 'int64',
                   'GEB_TRENNZ_KURZTEXT': 'string',
                   'GEB_WINKEL': 'float64',
                   'GEB_X': 'float64',
                   'GEB_Y': 'float64',
                   'HAUPT_IDENT_PSEUDO': 'string',
                   'LAGE_ZU_ADR': 'string',
                   'LAGE_ZU_ADR_TXT': 'string',
                   'NAME': 'string',
                   'NAME_BST': 'string',
                   'NAME_BST_GEB': 'string',
                   'NAME_GEB': 'string',
                   'NAME_ONR': 'string',
                   'NAME_ONR_BST_GEB': 'string',
                   'NAME_STR': 'string',
                   'OBJECTID': 'int64',
                   'OBJEKT_ID': 'int64',
                   'OBJEKTID_NAME_STR': 'int64',
                   'ON_BIS_ALPHA': 'string',
                   'ON_BIS_NUM': 'float64',
                   'ONR_BIS': 'string',
                   'ONR_VON': 'string',
                   'ON_VON_ALPHA': 'string',
                   'ON_VON_NUM': 'float64',
                   'PLATZKENNZEICHEN': 'float64',
                   'PLZ': 'int64',
                   'SCD_NAME_STR': 'int64',
                   'SCD_ZUGANG_STR': 'int64',
                   'SE_ANNO_CAD_DATA': 'float64',
                   'SHAPE': 'string',
                   'STRABS_ZUGANG_STR': 'int64',
                   'STRASSENSEITE': 'float64',
                   'TOPOGR_NAME_STR': 'string',
                   'UEBERACD': 'float64',
                   'ZUGADR_ONR_KGNUM': 'float64',
                   'ZUGADR_ONR_STAMMNR': 'float64',
                   'ZUGADR_ONR_TEILNR': 'float64',
                   'ZUGADR_REGIONALCODE': 'string',
                   'ZUGADR_TYP': 'int64',
                   'ZUGANG_MOEGLICH_J_N': 'int64',
                   'ZUG_BEZIRK': 'int64',
                   'ZUG_OBJEKTID': 'int64',
                   'ZUG_TIMESTAMP_MODIFY': 'string',
                   'ZUG_TRENNZEICHEN': 'int64',
                   'ZUG_TRENNZ_KURZTEXT': 'string',
                   'ZUG_WINKEL': 'float64',
                   'ZUG_X': 'float64',
                   'ZUG_Y': 'float64',
                   'ABLAUFDATUM': 'string'},
        )
        # some cleaning since there was at least one record with a transposition of digits

        district_df = district_df[district_df['PLZ'].isin(district_zip_codes)]
        # convert to datetime
        for date_col in district_df.columns:
            if 'TIMESTAMP' in date_col:
                district_df[date_col] = pd.to_datetime(district_df[date_col])

        district_df.rename(columns={'SHAPE': 'geometry'}, inplace=True)

        # transform access points to proper shapely Point objects
        district_df['geometry'] = district_df['geometry'].apply(
            lambda x: Point(float(s) for s in re.findall(r'-?\d+\.?\d*', x)))
        district_df['geometry'] = district_df['geometry'].apply(lambda p: Point(p.y, p.x))
        # turn into geodataframe and append
        district_gdf = gp.GeoDataFrame(district_df, crs=f'EPSG:{EPSG}')
        vienna_addresses = vienna_addresses.append(district_gdf)

    if write_path.suffix == '.geojson':
        vienna_addresses.to_file(write_path, index='FID', driver='GeoJSON')
    elif write_path.suffix == '.csv':
        vienna_addresses.to_csv(write_path, encoding='utf-8-sig')
    elif write_path.suffix == '.feather':
        warnings.warn('Writing and Reading in feather has caused issues before: values are not preserved!')
        vienna_addresses.to_feather(write_path, index=True)  # issues *may* be due to index=True, better stick to csv
    else:
        vienna_addresses.to_file(write_path, index='FID')
    print(f'Collated vienna address data stored at {write_path.as_posix()}')
    return vienna_addresses


def query_osrm(gs: gp.GeoSeries):
    """
    OSRM uses long/lat instead of lat/lon
    The demo server only provides the car profile

    :return: car-based distances and durations of the pairwise fastest routes between points in gs. The distance
    is not the shortest distance between points but the distance of the fastest route between points. durations are
    in seconds, distances are in meters
    """
    api_limit = 100
    half_api_limit = api_limit // 2
    durations = dict()
    distances = dict()
    remainder = 0 if len(gs) % half_api_limit == 0 else 1

    for i in tqdm.tqdm(range((len(gs) // half_api_limit) + remainder), desc='OSRM API QUERY'):
        sources_chunk = gs.iloc[half_api_limit * i: half_api_limit * (i + 1)]
        sources = [list(p.coords)[0] for p in sources_chunk]
        sources = [",".join([str(p[1]), str(p[0])]) for p in sources]  # OSRM uses lon/lat instead of lat/lon
        sources = ";".join(sources)

        for j in tqdm.tqdm(range((len(gs) // half_api_limit) + remainder)):
            destinations_chunk = gs.iloc[half_api_limit * j: half_api_limit * (j + 1)]
            destinations = [list(p.coords)[0] for p in destinations_chunk]
            destinations = [",".join([str(p[1]), str(p[0])]) for p in
                            destinations]  # OSRM uses lon/lat instead of lat/lon
            destinations = ";".join(destinations)

            # make the API request
            locations = ";".join([sources, destinations])
            sources_ids = ";".join(str(x) for x in (range(len(sources_chunk))))
            destinations_ids = ";".join(
                str(x) for x in (range(len(sources_chunk), len(sources_chunk) + len(destinations_chunk))))
            annotations = 'duration,distance'
            # querying the OSRM demo server
            # req = f"http://router.project-osrm.org/table/v1/car/{locations}?sources={sources_ids}&destinations={destinations_ids}&annotations={annotations}"
            # querying the docker image local server (much faster!)
            req = f"http://127.0.0.1:5000/table/v1/car/{locations}?sources={sources_ids}&destinations={destinations_ids}&annotations={annotations}"
            r = requests.get(req)
            assert r.status_code == 200, f'Status code: {r.status_code} is invalid. Check your request.\n{r.content}'

            # r.json()['durations']: array of arrays that stores the matrix in row-major order. durations[i][j] gives
            # the travel time from the i-th source to the j-th destination

            for k in range(len(sources_chunk)):
                key = sources_chunk.index[k]
                if key in durations:
                    durations[key] = durations[key] + r.json()['durations'][k]
                else:
                    durations[key] = r.json()['durations'][k]
                if key in distances:
                    distances[key] = distances[key] + r.json()['distances'][k]
                else:
                    distances[key] = r.json()['distances'][k]

    duration_matrix = pd.DataFrame.from_dict(durations,
                                             columns=gs.index,
                                             orient='index')  # index orientation since the dicts store values row-wise
    distance_matrix = pd.DataFrame.from_dict(distances,
                                             columns=gs.index,
                                             orient='index')  # index orientation since the dicts store values row-wise

    return distance_matrix, duration_matrix


def check_triangle_inequality(data, verbose=False, disable_progress_bar=True):
    data = np.array(data)
    n = len(data)
    num_violations = 0
    for i in trange(n, disable=disable_progress_bar):
        for j in range(n):
            if i == j:
                continue
            for k in range(n):
                if i == k or j == k:
                    continue

                d_ik = data[i][k]
                d_ij = data[i][j]
                d_jk = data[j][k]
                if not d_ik <= d_ij + d_jk:
                    num_violations += 1
                    if verbose:
                        print(f'violation: {i:02d}-{k:02d} > {i:02d}-{j:02d} + {j:02d}-{k:02d} =>'
                              f' {d_ik} > {d_ij} + {d_jk} => [{d_ik - d_ij - d_jk}]')
    return num_violations


if __name__ == '__main__':
    # re-query the data
    # _query_vienna_addresses_online(write_path=io.input_dir.joinpath('vienna_addresses.csv'))

    # read addresses from disk and preprocess
    gdf_full = read_vienna_addresses(io.input_dir.joinpath('vienna_addresses.csv'),
                                     # nrows=200,  # for testing
                                     )

    # sample some points for easier and more efficient handling
    n = 1000  # sample size
    for i in range(3):
        gdf = gdf_full.sample(n=n, replace=False)

        # write the sample to disk for faster reading later
        write_path = io.unique_path(io.input_dir, f'vienna_{n}_addresses' + '_#{:03d}' + '.csv')

        # query OSRM durations
        distance_matrix, duration_matrix = query_osrm(gdf.geometry)
        gdf.to_csv(io.input_dir.joinpath(write_path), encoding='utf-8-sig', index=True, header=True)
        dur_write_path = write_path.as_posix().replace('addresses', 'durations')
        duration_matrix.to_csv(dur_write_path, encoding='utf-8-sig', index=True, header=True)
        dist_write_path = write_path.as_posix().replace('addresses', 'distances')
        distance_matrix.to_csv(dist_write_path, encoding='utf-8-sig', index=True, header=True)

    pass
