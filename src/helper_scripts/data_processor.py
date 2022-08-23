import pandas as pd
import numpy as np
from datetime import timedelta
import json, urllib.request
from math import atan, cos, radians, sin, tan, asin, sqrt
import warnings
from shapely.errors import ShapelyDeprecationWarning
from holidays import Switzerland
import osmnx as ox

def get_litter_columns(df):
    """     Get litter columns from dataframe.
    Returns a list with numeric values corresponding to the litter columns in the dataframe.
    Used to easily call the litter columns, without having to manually search which ones
            are present in the dataframe.
    Only works when the litter columns names are numerical values

    Parameters
    ----------
    df : pandas.dataframe
        Dataframe containing litter columns with numeric values

    Returns
    -------
    litter_columns : list
        List with the column names that where umerical values
    """
    litter_columns = []
    for item in df.columns.to_list():
        if item.isdigit(): 
            litter_columns.append(item) 
    return litter_columns

def clean_df(df):
    """    Prepare dataframe for scripts
    Removes columns that are not needed for the prediction.
    Drops rows that don't have an 'edge_id' value.
    Cleans data types.
    """
    not_needed_columns = ['Unnamed: 0', 'place_id', 'value.Vehicle_Mode', 'speed', '_id',]
    df.drop(not_needed_columns, axis=1, inplace=True, errors='ignore')
    renaming_dictionary = {'suitcase.id':'suitcase_id', 'date.utc':'date_utc', 
                           'edge.id':'edge_id', 'edge.osmid':'edge_osmid', 
                           'place.id':'place_id', 'osm.highway':'osm_highway'}
    df.rename(columns = renaming_dictionary, inplace=True, errors='ignore')
    litter_columns = get_litter_columns(df)
    df = df.dropna(subset=['edge_id']).copy() #--> Drops rows without an edge_id specified
    df['date_utc'] = pd.to_datetime(df['date_utc']).dt.date
    df[litter_columns] = df[litter_columns].fillna(0)
    df[litter_columns] = df[litter_columns].astype(np.int64) 
    df['total_litter'] = df[litter_columns].sum(axis=1)
    df['edge_osmid'] = df['edge_osmid'].astype(int)
    return df

def aggregate_df(df, aggregation_method='sum'):
    """     Aggregate for model training
    Aggregates the dataframe by day and edge_id.
    Returns datframe where each unique day has no repeated edge_id
    """
    litter_columns = get_litter_columns(df)
    to_agg = {'edge_osmid':'first', 'osm_highway':'first', 'total_litter':aggregation_method}
    for litter in litter_columns:
        to_agg[litter] = aggregation_method
    df = df.groupby(['date_utc', 'edge_id'], as_index=False).agg(to_agg)
    return df

def create_date_features(df):
    """     Adds date features for model training
    Separates the date into day, month and year columns.
    Add days of the week as a categorical feature.
    Marks the day and following day after a holiday as a boolean.
    """
    df['Year'] = pd.DatetimeIndex(df['date_utc']).year.astype(object)
    df['month'] = pd.DatetimeIndex(df['date_utc']).month.astype(object)
    df['day'] = pd.DatetimeIndex(df['date_utc']).day.astype(object)
    weekdays = ('Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday')
    df['weekday'] = pd.to_datetime(df['date_utc']).dt.date.apply(lambda x: x.weekday())
    df['weekday'] = df['weekday'].apply(lambda x: weekdays[x])
    holiday = [holiday for holiday in Switzerland(years=[2021, 2022]).items()]
    for day in set(holiday):
        holiday.append(((day[0] + timedelta(days=1)), day[1]))
    holidays_df = pd.DataFrame(holiday, columns=["date", "holiday"])
    holidays_df['holiday'] = holidays_df['holiday'].astype(str)
    df['holiday'] = df['date_utc'].apply(lambda x: 1 if x in holidays_df['date'].values else 0)
    return df

def create_coordinates_features(df, file_path='data/edges.geojson'):
    """     Adds columns with bounding box coordinates for the corresponding edge_id
    Using the edges.geojson file, adds the coordinates that create a bounding box for each edge_id.
    Also adds the length of the edge as a feature
    """
    with open(file_path) as file:
        data = json.load(file)
    df_edges = pd.DataFrame(data['features'])  # <-- The only column needed from geojson file
    df_edges = pd.concat([df_edges.drop(['properties'], axis=1), df_edges['properties'].apply(pd.Series)], axis=1)  # <--Explode properties dictionary inside cells
    df_edges.rename(columns = {'id':'edge_id'}, inplace = True)
    def sort_bbox(x):  #<-- Rearrange BBOX (lat_north, lat_south, lon_east, lon_west)
        lat = sorted([x[1], x[3]], key=float, reverse=True)
        lon = sorted([x[0], x[2]], key=float, reverse=True)
        return lat + lon
    df_edges['bbox'] = df_edges['bbox'].apply(sort_bbox)
    df_edges = df_edges[['edge_id', 'bbox', 'length']].copy()
    bbox_exploded = pd.DataFrame(df_edges["bbox"].to_list(), columns=['lat_north', 'lat_south', 'lon_east', 'lon_west'])
    df_edges = pd.concat([df_edges, bbox_exploded], axis=1)
    df_edges.drop(['bbox'], axis=1, inplace=True)
    df_edges.rename(columns = {'length':'edge_length'}, inplace = True)
    df = pd.merge(df, df_edges, how="left", on="edge_id")
    return df

def create_weather_features(df, file_path='data/weather_basel_2021-2022.csv'):
    """     Creates weather data as a feature
    Data collected from "https://www.meteoblue.com/en/weather/archive/export/basel_switzerland_2661604"
    """
    df_weather = pd.read_csv(file_path)
    df['date_utc'] = pd.to_datetime(df['date_utc']).dt.date  #<-- Make sure date_utc is date format
    weather_columns = {'location':'date_utc', 'Basel':'temperature_max', 'Basel.1':'temperature_min', 'Basel.2':'temperature_mean', 'Basel.3':'precipitation',
                    'Basel.4':'snowfall', 'Basel.5':'humidity_max', 'Basel.6':'humidity_min', 'Basel.7':'humidity_mean', 'Basel.8':'cloud_coverage',
                    'Basel.9':'wind_speed_max', 'Basel.10':'wind_speed_min', 'Basel.11':'wind_speed_mean'}
    df_weather.rename(columns = weather_columns, inplace = True)
    df_weather.drop(['Basel.12'], axis=1, inplace=True)
    df_weather = df_weather.iloc[9:].copy()  #<-- Delete rows with metadata
    df_weather['date_utc'] = pd.to_datetime(df_weather['date_utc']).dt.date
    weather_date = df_weather.pop('date_utc')  #<-- Pop date so it doesn't get converted
    df_weather = df_weather.apply(pd.to_numeric)
    df_weather = df_weather.round(decimals = 1)
    df_weather.insert(0, 'date_utc', weather_date) #<-- Reassign unmodified date
    df = pd.merge(df, df_weather, how="left", on="date_utc")
    return df

def create_osm_columns():
    """     Creates a list with the amenities from OSM to extract
    Amenities dictionary at https://wiki.openstreetmap.org/wiki/Map_features
    """
    osm_tags = ['vending_machine', 'bench', 'bar', 'fast_food', 'ice_cream', 'kindergarten', 'school', 'hospital', 'cinema', 
                'fountain', 'dog_toilet', 'recycling', 'waste_basket', 'waste_disposal', 'childcare', 'marketplace',
                'bus_station', 'fuel', 'taxi', 'parking', 'atm', 'clinic', 'nightclub', 'toilets'
                ]
    osm_columns = sorted(osm_tags)
    return osm_columns

def create_osm_features(df):
    """     Adds number of amenities located at the edge
    Based on the tags dictionary, adds a count of the amount of amenities present in each edge
    Obtains data from Open Street Maps using the OSMNX library
    """
    warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning) 
    basel = 'Basel, Basel, Switzerland'
    tags = {'amenity': sorted(create_osm_columns())}
    amenity = ox.geometries_from_place(basel, tags=tags)
    df_osm = pd.DataFrame(amenity)
    df_osm = df_osm[['amenity', 'geometry']].copy()  #<-- Select the only needed columns
    df_osm['osm_id'] = df_osm.index.to_numpy()  #<-- Detach the index and assign it to a normal column
    df_osm.reset_index(drop=True, inplace=True)  #<-- Drop index
    osm_id_exploded = pd.DataFrame(df_osm["osm_id"].to_list(), columns=['type', 'osm'])  #<-- Explode index since it contains two indices
    df_osm = pd.concat([df_osm, osm_id_exploded], axis=1) 
    df_osm.drop(['osm_id', 'osm'], axis=1, inplace=True)
    df_osm = df_osm[df_osm['type'] == 'node']  #<-- Drop Multipoligon points
    #### Clean the coordinates from the GeoPandas geometry format to latitude and longitude columns
    df_osm['lon'] = df_osm[df_osm['type'] == "node"]['geometry'].apply(lambda p: p.x)
    df_osm['lat'] = df_osm[df_osm['type'] == "node"]['geometry'].apply(lambda p: p.y)
    df_osm.drop(['geometry', 'type'], axis=1, inplace=True)
    df_edges_coordinates = df[['edge_id', 'lat_north', 'lat_south', 'lon_east', 'lon_west']].copy()
    df_edges_coordinates = df_edges_coordinates.drop_duplicates(subset='edge_id', keep='first')

    # Make a list of the edges that have an amenity to them based on lat,lon conditional
    def is_between(a, x, b):
        return min(a, b) < x < max(a, b)
    edges_dict = []
    for edges_row in df_edges_coordinates.itertuples():
        for osm_row in df_osm.itertuples():
            if is_between(edges_row.lat_south, osm_row.lat, edges_row.lat_north) and is_between(edges_row.lon_west, osm_row.lon, edges_row.lon_east):
                edges_dict.append([edges_row.edge_id, osm_row.amenity])

    # Group by edge_id and get the value counts per amenity
    df_edges_dict = pd.DataFrame(edges_dict, columns = ['edge_id', 'amenity'])
    df_edges_dict = df_edges_dict.groupby('edge_id')['amenity'].value_counts().unstack(fill_value=0).reset_index()
    df = pd.merge(df, df_edges_dict, how="left", on="edge_id")
    osm_columns = list(df_edges_dict.columns[1:])
    df[osm_columns] = df[osm_columns].fillna(value=0)  #<-- Fill missing values, since not all edges have amenities
    df[osm_columns] = df[osm_columns].astype(int)
    return df