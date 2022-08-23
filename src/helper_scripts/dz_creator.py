import helper_scripts.data_processor as data_processor
import pandas as pd

def create_darkzones(df):
    """     Creates a dataframe with the edges not present during each day
    Considers only the dates present on the dataframe passed as an argument.
    Based on the dataframe passed as an argument, it will create a dictionary of the unique edge_ids present.
    Using this dictionary, it will add edge_ids not present on each individual day to the returned dataframe
    """
    litter_columns = data_processor.get_litter_columns(df)
    df = data_processor.clean_df(df)
    df = data_processor.aggregate_df(df)
    df.drop('total_litter', axis=1, inplace=True, errors='ignore')
    df.drop(litter_columns, axis=1, inplace=True, errors='ignore')
    df['date_utc'] = df['date_utc'].astype(str)

    edges_dictionary = df['edge_id'].unique()
    # Make a dictionary with the date as key and the edges existing within that day in a list as their value
    today = ''
    existing_edges = {}
    for index, row in df.iterrows():
        if row['date_utc'] != today:
            today = row['date_utc']
            existing_edges[today] = []
        if row['edge_id'] not in existing_edges[today]:
            existing_edges[today].append(row['edge_id'])

    # Compare the existing edges in a day with the list of all the unique edges, 
    #   add the edges not in a day in a dictionary, 
    #   where the key is the date and the value is a list of the missing edges in that dataframe
    missing_edges = {}
    for key, value in existing_edges.items():
        missing_edges[key] = list(set(edges_dictionary).difference(value))
    missing_edges_list = []
    for key, value in missing_edges.items():
        for v in value:
            missing_edges_list.append([key, v])

    df_darkzones = pd.DataFrame(list(missing_edges_list), columns = ['date_utc', 'edge_id']) 
    edges_dictionary_extended = df.groupby(['edge_id'], as_index=False).agg({'edge_osmid':'first', 'osm_highway':'first'})
    df_darkzones = pd.merge(df_darkzones, edges_dictionary_extended, how='outer', on='edge_id')
    #df_darkzones['date_utc'] = pd.to_datetime(df_darkzones['date_utc'], format='%Y-%m-%d') Maybe delete
    df_darkzones['date_utc'] = pd.to_datetime(df_darkzones['date_utc']).dt.date
    df_darkzones = df_darkzones.sort_values(by="date_utc")
    df_darkzones = df_darkzones.reset_index(drop=True)
    df_darkzones['row_type'] = "darkzone"
    df_darkzones['edge_osmid'] = df_darkzones['edge_osmid'].astype(int)
    return df_darkzones