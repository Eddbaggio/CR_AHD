import re
from pathlib import Path

import geopandas as gp
import pandas as pd
from shapely.geometry import Point
from utility_module import io


def read_vienna_addresses(url=io.input_dir.joinpath('vienna_addresses.csv'),
                          EPSG=4326,):
    df = pd.read_csv(
        url,
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
               'ABLAUFDATUM': 'string'},
    )

    for date_col in df.columns:
        if 'TIMESTAMP' in date_col:
            df[date_col] = pd.to_datetime(df[date_col])

    # transform access points to proper shapely Point objects
    df['geometry'] = df['geometry'].apply(lambda x: Point(float(s) for s in re.findall(r'-?\d+\.?\d*', x)))

    # turn into geodataframe
    gdf = gp.GeoDataFrame(df, crs=f'EPSG:4326')
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
        url = f"https://data.wien.gv.at/daten/geo?{'&'.join([f'{k}={v}' for k,v in d.items()])}"
        district_df = pd.read_csv(
            url,
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
        # some cleaning
        district_df = district_df[district_df['PLZ'].isin([
            1010, 1020, 1030, 1040, 1050, 1060, 1070, 1080, 1090, 1100, 1110, 1120, 1130, 1140, 1150, 1160, 1170, 1180,
            1190, 1200, 1210, 1220, 1230,
        ])]
        # convert to datetime
        for date_col in district_df.columns:
            if 'TIMESTAMP' in date_col:
                district_df[date_col] = pd.to_datetime(district_df[date_col])

        district_df.rename(columns={'SHAPE': 'geometry'}, inplace=True)

        # transform access points to proper shapely Point objects
        district_df['geometry'] = district_df['geometry'].apply(lambda x: Point(float(s) for s in re.findall(r'-?\d+\.?\d*', x)))

        # turn into geodataframe and append
        district_gdf = gp.GeoDataFrame(district_df, crs=f'EPSG:{EPSG}')
        vienna_addresses = vienna_addresses.append(district_gdf)

    if write_path.suffix == '.geojson':
        vienna_addresses.to_file(write_path, index='FID', driver='GeoJSON')
    elif write_path.suffix == '.csv':
        vienna_addresses.to_csv(write_path)
    else:
        vienna_addresses.to_file(write_path, index='FID')
    print(f'Collated vienna address data stored at {write_path.as_posix()}')
    return vienna_addresses


if __name__ == '__main__':
    # re-query the data
    _query_vienna_addresses_online()

    # read from disk
    # gdf = read_vienna_addresses()
