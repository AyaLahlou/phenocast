import os
import glob
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
import pandas.api.types as ptypes
import matplotlib.pyplot as plt
from sklearn.preprocessing import QuantileTransformer, LabelEncoder, StandardScaler, MinMaxScaler

def sort_loc_time(data_path: str, output_path: str):
    """
    Sort the data by location and time and save to new file.    
    Args:
        data_path: str, the path to the data file
        
    Returns:
        None
    """
    file_name=data_path.split('/')[-1]
    output_path=output_path+'/sorted_'+file_name

    data_df= pd.read_parquet(data_path)

    sorting_order = {
        'location': 'asc',  # descending order
        'time': 'asc',   # ascending order
    }

    # Convert the sorting order to ascending/descending values
    sorting_values = {col: (True if order == 'asc' else False) for col, order in sorting_order.items()}

    # Sort the DataFrame based on the specified columns and order
    data_df = data_df.sort_values(by=list(sorting_order.keys()), ascending=list(sorting_values.values()))
    data_df.to_parquet(output_path)



def tft_process(path, hist_len, fut_len, output_path):
    """
    Process the data for TFT model training and save to new file.
    
    Args:
    path: str, the path to the data file
    hist_len: int, the length of the historical time series
    fut_len: int, the length of the future time series
    output_path: str, the path to the output file
    output_filename: str, the name of the output file
    
    Returns:
        None
    """
    
    output_filename = path.split('/')[-1].split('.')[0] + '.pkl'
    data_df= pd.read_parquet(path)
    #validation columns before dropping na 
    data_df = data_df[['time', 'location', 'latitude', 'longitude', 'tmin', 'tmax', 'precipitation', 'radiation', 'photoperiod', 'sm', 'sif_clear_inst']]
    #print size of
    print("length of sorted parquet file",len(data_df))
    data_df = data_df.dropna()
    
    #data_df=data_df[data_df['latitude']>0]# northern hemisphere only 
    data_df['time'] = pd.to_datetime(data_df['time'])
    # No records will be considered outside these bounds
    start_date = datetime(1982, 1, 14)
    end_date = datetime(2022, 1, 1)
    
    print(data_df['time'].min())
    print(data_df['time'].max()) 
    
    # these will not be included as part of the input data which will end up feeding the model
    meta_attrs = ['time', 'location','soil', "soil_x", "soil_y"]
    
    # These are the variables that are known in advance, and will compose the futuristic time-series
    known_attrs = ['tmin', 'tmax',  'radiation','precipitation', 'sm','photoperiod']
    # The following set of variables will be considered as static, i.e. containing non-temporal information
    # every attribute which is not listed here will be considered as temporal.
    static_attrs = ['latitude', 'longitude']
    # The following set of variables will be considered as categorical.
    # The rest of the variables (which are not listed below) will be considered as numeric.
    categorical_attrs = [] #soil used to be here 
    target_signal = 'sif_clear_inst'

    unique_locations = data_df['location'].unique()
    print(str(unique_locations)+" unique locations.")
    
    
    all_cols = list(data_df.columns)
    feature_cols = [col for col in all_cols if col not in meta_attrs]
    
    feature_map = {
        'static_feats_numeric': [col for col in feature_cols if col in static_attrs and col not in categorical_attrs],
        'static_feats_categorical': [col for col in feature_cols if col in static_attrs and col in categorical_attrs],
        'historical_ts_numeric': [col for col in feature_cols if col not in static_attrs and col not in categorical_attrs],
        'historical_ts_categorical': [col for col in feature_cols if col not in static_attrs and col in categorical_attrs],
        'future_ts_numeric': [col for col in feature_cols if col in known_attrs and col not in categorical_attrs],
        'future_ts_categorical': [col for col in feature_cols if col in known_attrs and col in categorical_attrs]
    }
    # allocate a dictionary to contain the scaler and encoder objects after fitting them
    scalers = {'numeric': dict(), 'categorical': dict()}
    # for the categorical variables we would like to keep the cardinalities (how many categories for each variable)
    categorical_cardinalities = dict()
    
    only_train= data_df
    
    for col in tqdm(feature_cols):
        if col in categorical_attrs:
            scalers['categorical'][col] = LabelEncoder().fit(only_train[col].values)
            categorical_cardinalities[col] = only_train[col].nunique()
        else:
            if col in ['sif_clear_inst']:
                scalers['numeric'][col] = StandardScaler().fit(only_train[col].values.astype(float).reshape(-1, 1))
            elif col in ['day_of_year']:
                scalers['numeric'][col] = MinMaxScaler().fit(only_train[col].values.astype(float).reshape(-1, 1))
            else:
                scalers['numeric'][col] = QuantileTransformer(n_quantiles=256).fit(
                    only_train[col].values.astype(float).reshape(-1, 1))
                
    for col in tqdm(feature_cols):
        if col in categorical_attrs:
            #le = scalers['categorical'][col]
            # handle cases with unseen keys
            #le_dict = dict(zip(le.classes_, le.transform(le.classes_)))
            #data_df[col] = data_df[col].apply(lambda x: le_dict.get(x, max(le.transform(le.classes_)) + 1))
            data_df[col] = data_df[col].astype(np.int32)
        else:
            data_df[col] = scalers['numeric'][col].transform(data_df[col].values.reshape(-1, 1)).squeeze()
            data_df[col] = data_df[col].astype(np.float32)
            
    # split data also by location: keep some locations for training, some for validation, and some for testing
    # Split locations into train, validation, and test sets (60%, 20%, 20%)
    np.random.seed(42)  # for reproducibility
    unique_locations = np.random.permutation(unique_locations)
    num_locations = len(unique_locations)
    train_end = int(0.7 * num_locations)
    validation_end = int(0.9 * num_locations)

    train_locations = unique_locations[:train_end]
    validation_locations = unique_locations[train_end:validation_end]
    test_locations = unique_locations[validation_end:]


    print(data_df['time'].min())
    print(data_df['time'].max())    
    
    train_subset_start = data_df.loc[data_df['time'] < datetime(1992, 1, 1)]
    train_subset_end = data_df.loc[data_df['time'] >= datetime(2011, 1, 1)]
    validation_subset = data_df[(data_df['time']  >= datetime(1992, 1, 1)) & (data_df['time'] < datetime(2001, 1, 1))]
    test_subset = data_df[(data_df['time'] >= datetime(2001, 1, 1))& (data_df['time'] < datetime(2011, 1, 1))]
    train_subset = pd.concat([train_subset_start, train_subset_end], axis=0)


    print(str(len(train_subset['location'].unique()))+" train unique locations.")
    print(str(len(validation_subset['location'].unique()))+" validation unique locations.")
    print(str(len(test_subset['location'].unique()))+" test unique locations.")
    
    # Split data by location
    train_subset = train_subset[train_subset['location'].isin(train_locations)]
    validation_subset = validation_subset[validation_subset['location'].isin(validation_locations)]
    test_subset = test_subset[test_subset['location'].isin(test_locations)]
    
    subsets_dict = {'train': train_subset,
                    'validation': validation_subset,
                    'test': test_subset}

    print(str(len(train_subset['location'].unique()))+" train unique locations.")
    print(str(len(validation_subset['location'].unique()))+" validation unique locations.")
    print(str(len(test_subset['location'].unique()))+" test unique locations.")
    
    data_sets = {'train': dict(),'validation': dict(),'test': dict()}
    
    for subset_name, subset_data in subsets_dict.items():
        subset_data['id'] = subset_data['location'].astype(str) + '_' + subset_data['time'].astype(str)

    for subset_key, subset_data in subsets_dict.items():
        print(subset_key)
        samp_interval =hist_len+fut_len
        history_len = hist_len
        future_len = fut_len 
        # sliding window, according to samp_interval skips between adjacent windows
        for i in range(0, len(subset_data), samp_interval):              
            slc = subset_data.iloc[i: i + history_len + future_len]
            #print(i,i + history_len + future_len)
            if len(slc) < (history_len + future_len) or slc.iloc[0]['location']!=slc.iloc[-1]['location'] or (slc.iloc[-1]['time'] - slc.iloc[0]['time'])>timedelta(days=samp_interval):
                # skip edge cases, where not enough steps are included
                if (slc.iloc[-1]['time'] - slc.iloc[0]['time'])>timedelta(days=samp_interval):
                    print('SKIP starts at:', slc.iloc[0]['time'], 'ends at ',slc.iloc[-1]['time'])
                #print('switching time series: ', slc.iloc[0]['location'], slc.iloc[-1]['location'])
                continue
            # meta
            data_sets[subset_key].setdefault('time_index', []).append(slc.iloc[history_len - 1]['location'])
            #print(slc.iloc[:history_len]['location'])
            #print(slc.iloc[history_len:]['sif_clear_inst'])
    
            # static attributes
            data_sets[subset_key].setdefault('static_feats_numeric', []).append(
                slc.iloc[0][feature_map['static_feats_numeric']].values.astype(np.float32))
            data_sets[subset_key].setdefault('static_feats_categorical', []).append(
                slc.iloc[0][feature_map['static_feats_categorical']].values.astype(np.int32))
    
            # historical
            data_sets[subset_key].setdefault('historical_ts_numeric', []).append(
                slc.iloc[:history_len][feature_map['historical_ts_numeric']].values.astype(np.float32).reshape(
                    history_len, -1))
            data_sets[subset_key].setdefault('historical_ts_categorical', []).append(
                slc.iloc[:history_len][feature_map['historical_ts_categorical']].values.astype(np.int32).reshape(
                    history_len, -1))
    
            # futuristic (known)
            data_sets[subset_key].setdefault('future_ts_numeric', []).append(
                slc.iloc[history_len:][feature_map['future_ts_numeric']].values.astype(np.float32).reshape(future_len,
                                                                                                           -1))
            data_sets[subset_key].setdefault('future_ts_categorical', []).append(
                slc.iloc[history_len:][feature_map['future_ts_categorical']].values.astype(np.int32).reshape(future_len,
                                                                                                             -1))
    
            # target
            data_sets[subset_key].setdefault('target', []).append(
                slc.iloc[history_len:]['sif_clear_inst'].values.astype(np.float32))
            data_sets[subset_key].setdefault('id', []).append(
                slc.iloc[history_len:]['id'].values.astype(str))
            #break
        #break
    # for each set
    print('Saving...')
    for set_key in list(data_sets.keys()):
        # for each component in the set
        for arr_key in list(data_sets[set_key].keys()):
            # list of arrays will be concatenated
            if isinstance(data_sets[set_key][arr_key], np.ndarray):
                data_sets[set_key][arr_key] = np.stack(data_sets[set_key][arr_key], axis=0)
            # lists will be transformed into arrays
            else:
                data_sets[set_key][arr_key] = np.array(data_sets[set_key][arr_key])

    output_path = os.path.abspath(output_path)
    file_name= output_filename
    
    #testing the output
    #check if length of train subset time index >0 and print it 
    print('length of train subset time index:',len(data_sets['train']['time_index']))
    #check if length of validation subset time index >0 and print it
    print('length of validation subset time index:',len(data_sets['validation']['time_index']))
    #check if length of test subset time index >0 and print it
    print('length of test subset time index:',len(data_sets['test']['time_index']))
    #if len test subset time index <0 pritn error message 
    if len(data_sets['test']['time_index'])<0:
        print('error len test subset time index <0')
    if len(data_sets['train']['time_index'])<0:
        print('error len train subset time index <0')
    if len(data_sets['validation']['time_index'])<0:
        print('error len validation subset time index <0')
        

    with open(os.path.join(output_path, file_name), 'wb') as f:
            pickle.dump({
                'data_sets': data_sets,
                'feature_map': feature_map,
                'scalers': scalers,
                'categorical_cardinalities': categorical_cardinalities
            }, f, pickle.HIGHEST_PROTOCOL)
    print('Done!')
    
    
if __name__ == "__main__":
    print('start')

    data_path= "/burg/glab/users/al4385/data/BorealClass"
    sorted_path="/burg/glab/users/al4385/data/sorted_BorealClass"
    output_path="/burg/glab/users/al4385/data/TFT_boreal"

    hist_len=365
    fut_len=30    
    
    
    print('Checking data path:', data_path)
    print('Checking sorted path:', sorted_path)
    print('Checking output path:', output_path)
    
    print(glob.glob(data_path+'/*.parquet'))
    
    #for file in glob.glob(data_path+'/*.parquet'):
        #print('sorting'+file) 
        #sort_loc_time(file, sorted_path)
    
    for sorted_file in glob.glob(sorted_path+'/*.parquet'):
        if sorted_file == "/burg/glab/users/al4385/data/sorted_BorealClass/sorted_BoNE_1982_2021.parquet":
            continue
        print('processing'+sorted_file)
        tft_process(sorted_file, hist_len, fut_len, output_path)
    