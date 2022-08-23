import helper_scripts.data_processor as data_processor
import helper_scripts.train as train
import helper_scripts.dz_creator as dz_creator
import numpy as np

def train_models(df, litters, aggregation_method='sum'):
    """     Train ML prediciton models

    Parameters
    ----------
    df : pandas.dataframe
        Cleaned and aggegated Dataframe with added features all from data_processor script
    litters : list
        List containig the numerical values of the desired litters to train the ML models
    aggregation_method : string
        The type of aggregation when grouping by edge_id and date

    Returns
    -------
    models : dictionary
    Each key correspond to a type of litter, which in turn conaints the following:
        'model' : The ML algorithm
        'y_test' : Dataframe with y_test
        'score' : The deviance squared score for the Poisson Model
        'time2train' : Simply the time it took in minutes to fit the model
    """
    df = data_processor.clean_df(df)
    df = data_processor.aggregate_df(df, aggregation_method)
    df = data_processor.create_date_features(df)
    df = data_processor.create_coordinates_features(df)
    df = data_processor.create_weather_features(df)
    df = data_processor.create_osm_features(df)
    models = train.make_models(df, litters)
    return models

def predict_darkzones(df, models):
    """     Predict the litter counts for the darkzones
    The main purpose of this package, to predict the litter counts for the edges that have no data
        on a given day from the image recognition

    Parameters
    ----------
    df : pandas.dataframe
        Cleaned and aggegated Dataframe with added features all from data_processor script
    models : dictionary
        Dictionary returned using the train_models() function

    Returns
    -------
    df : pandas.dataframe
        Dataframe with the same dates as dataframe passed as argument, but with 
            predicted litters for the missing edges
    """
    df = dz_creator.create_darkzones(df)
    df = data_processor.create_date_features(df)
    df = data_processor.create_coordinates_features(df)
    df = data_processor.create_weather_features(df)
    df = data_processor.create_osm_features(df)
    osm_columns = data_processor.create_osm_columns()
    for key, model in models.items():
        predictions = model[0].predict(df)
        predictions = np.rint(predictions).astype(int)
        df[f"{key}"] = predictions
    columns_to_drop = ['Year', 'month', 'day', 'weekday', 'holiday', 'lat_north', 'lat_south', 'lon_east', 'lon_west', 'edge_length', 
            'temperature_max', 'temperature_min', 'temperature_mean', 'precipitation', 'snowfall', 'humidity_max', 'humidity_min', 
            'humidity_mean', 'cloud_coverage', 'wind_speed_max', 'wind_speed_min', 'wind_speed_mean']
    columns_to_drop.extend(osm_columns)
    df.drop(columns_to_drop, axis=1, inplace=True, errors='ignore')
    return df
