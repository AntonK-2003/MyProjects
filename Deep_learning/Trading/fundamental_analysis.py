import OpenBlender
import pandas as pd
import json

# It only contains the id, we'll add more parameters later.
parameters = { 
 'id_dataset':'5d4c3af79516290b01c83f51'
}

parameters = {     
   'token':'your_token',
   'id_dataset':'5d4c3af79516290b01c83f51'
}

# This function pulls the data and orders by timestamp
def pullObservationsToDF(parameters):
    action = 'API_getObservationsFromDataset'
    df = pd.read_json(json.dumps(OpenBlender.call(action,parameters)['sample']), convert_dates=False,convert_axes=False) .sort_values('timestamp', ascending=False)
    df.reset_index(drop=True, inplace=True)
    return df
df = pullObservationsToDF(parameters)

parameters = { 
   'token':'your_token',
   'id_dataset':'5d4c3af79516290b01c83f51',
   'target_threshold' : {'feature':'change','success_thr_over':0},
   'lag_target_feature' : {'feature':'change_over_0', 'periods':1},
   'blends':[{'id_blend':'5dc1a404951629331f6359dd',
               'blend_type' : 'text_ts',
               'restriction' : 'predictive',
               'blend_class' : 'closest_observation', 
               'specifications':{'time_interval_size' : 3600*12 }}],
   'date_filter':{'start_date':'2019-08-20T16:59:35.825Z',
                   'end_date':'2019-11-04T17:59:35.825Z'},
   'drop_non_numeric' : 1
}

df = pullObservationsToDF(parameters)
df.head() 

df = pullObservationsToDF(parameters)
df.head()

target_variable = 'TARGET_change_over_0'
df = df.dropna()
df.corr()[target_variable].sort_values()

action = 'API_createTextVectorizer'
vectorizer_parameters = {
    'token' : 'your_token',
    'name' : 'Fox Business TextVectorizer',
    'sources':[{'id_dataset' : '5d571f9e9516293a12ad4f6d',
                'features' : ['title']}],
    'ngram_range' : {'min' : 1, 'max' : 2},
    'language' : 'en',
    'remove_stop_words' : 'on',
    'min_count_limit' : 2
}

X = df.loc[:, df.columns != target_variable].values
y = df.loc[:,[target_variable]].values
div = int(round(len(X) * 0.29))
# We take the first observations as test and the last as train because the dataset is ordered by timestamp descending.
X_test = X[:div]
y_test = y[:div]
print(X_test.shape)
print(y_test.shape)
X_train = X[div:]
y_train = y[div:]
print(X_train.shape)
print(y_train.shape)
