import os
import logging
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utils import AttributeDict, DATASETS_PATH

#logging.basicConfig(level=logging.DEBUG)

ALL_FEATURES = ['data_channel_is_bus', 'global_subjectivity',
                  'data_channel_is_entertainment', 'avg_positive_polarity',
                  'max_negative_polarity', 'data_channel_is_tech',
                  'n_non_stop_unique_tokens', 'data_channel_is_world',
                  'data_channel_is_lifestyle', 'abs_title_subjectivity', 'LDA_00',
                  'rate_positive_words', 'data_channel_is_socmed',
                  'abs_title_sentiment_polarity', 'LDA_02', 'max_positive_polarity',
                  'LDA_01', 'LDA_04', 'weekday_is_monday',
                  'global_sentiment_polarity', 'is_weekend', 'title_subjectivity',
                  'average_token_length', 'n_tokens_title', 'weekday_is_thursday',
                  'weekday_is_saturday', 'weekday_is_friday', 'num_self_hrefs',
                  'weekday_is_tuesday', 'min_negative_polarity', 'num_imgs',
                  'num_hrefs', 'title_sentiment_polarity', 'num_keywords',
                  'num_videos', 'weekday_is_wednesday', 'kw_min_min', 'kw_avg_avg',
                  'kw_avg_min', 'kw_min_avg', 'n_tokens_content', 'kw_max_avg',
                  'kw_max_min', 'self_reference_avg_sharess',
                  'self_reference_min_shares', 'self_reference_max_shares',
                  'kw_min_max', 'kw_avg_max', 'kw_max_max', 'rate_negative_words',
                  'min_positive_polarity', 'avg_negative_polarity',
                  'global_rate_negative_words', 'global_rate_positive_words',
                  'n_non_stop_words', 'n_unique_tokens', 'weekday_is_sunday',
                  'LDA_03']

CATEGORICAL_FEATURES = ["data_channel_is_lifestyle",
                        "data_channel_is_entertainment",
                        "data_channel_is_bus",
                        "data_channel_is_socmed",
                        "data_channel_is_tech",
                        "data_channel_is_world",
                        "weekday_is_monday",
                        "weekday_is_tuesday",
                        "weekday_is_wednesday",
                        "weekday_is_thursday",
                        "weekday_is_friday",
                        "weekday_is_saturday",
                        "weekday_is_sunday",
                        "is_weekend"]

DEFAULT_HPARAMS = AttributeDict(
    kbest=10,

)


def get_dataset(root:str=DATASETS_PATH, train_size:float=.8, shuffle=True, seed=None) -> AttributeDict:
    dtype_dict = {x:np.float_ if x in CATEGORICAL_FEATURES else np.float_ for x in ALL_FEATURES}
    dataset = AttributeDict(
        train=AttributeDict(
            X=pd.read_csv(os.path.join(root, 'X1.csv'), dtype=dtype_dict),
            y=pd.read_csv(os.path.join(root, 'Y1.csv'), header=None, dtype={"shares":np.float_}, names=["shares"])
        ),
        validation=AttributeDict(
            X=None,
            y=None
        ),
        test=AttributeDict(
            X=pd.read_csv(os.path.join(root, 'X2.csv'), dtype=dtype_dict),
            y=None
        )

    )

    dataset.train.X, dataset.validation.X, dataset.train.y, dataset.validation.y = train_test_split(dataset.train.X, dataset.train.y, train_size=train_size, shuffle=shuffle, random_state=seed)

    return dataset

def dataset_to_X_y(dataset, keys, datatype="numpy"):
    if isinstance(keys, str):
        if keys == "all":
            keys = ("train", "validation")
    out = tuple()
    for key in keys:
        X, y = dataset[key].X, dataset[key].y
        if datatype.lower() in ("np", "numpy"):
            X, y = X.values, y.values.reshape(-1)
        
        out += (X,y)

    return out


def cutoff_outliers(X, y):
    from sklearn.ensemble import IsolationForest

    if_clf = IsolationForest() # contamination in auto mode
    if_clf.fit(X)

    _out = if_clf.predict(X)
    mask = _out == -1

    _out_idx = np.argwhere(mask).reshape(-1)
    out_idx = X.index[_out_idx]

    X.drop(out_idx, inplace=True)
    y.drop(out_idx, inplace=True)

    logging.info(f"Dropped {100*len(out_idx)/len(y):.2f}% of outliers ")

    return X, y

def cutoff_outliers_hard(X, y):
    greater = np.argwhere(y.values  > 2e4).reshape(-1)
    greater_idx = y.index[greater]


    X.drop(greater_idx, inplace=True)
    y.drop(greater_idx, inplace=True)

    return X, y

def preprocess_all(
    ds,
    subset=None, 
    scale=True,
    remove_outliers=False
    ):
    dataset = ds.copy()
    for key in ("train", "validation", "test"):
        if subset is not None:
            dataset[key].X = dataset[key].X[subset]
        #dataset[key].X = preprocess_pipe(dataset[key].X)

    # --------------------------------------------------
    # --------- APPLY ONLY ON TRAIN DATASET ------------   
    # --------------------------------------------------

    if remove_outliers:
        dataset.train.X, dataset.train.y = cutoff_outliers(dataset.train.X, dataset.train.y)
        dataset.validation.X, dataset.validation.y = cutoff_outliers(dataset.validation.X, dataset.validation.y)

    # --------------------------------------------------
    # ------------ APPLY ON ALL DATASETS ---------------
    # --------------------------------------------------

    # Scale the values
    if scale:
        scaler = StandardScaler()
        scaler.fit(dataset.train.X)
        for key in ("train", "validation", "test"):
            dataset[key].X.iloc[:,:] = scaler.transform(dataset[key].X.values)

    return dataset



def preprocess_pipe(df, hparams):
    # Set to categorical 
    return df   


SUBSET_FEATURES = [
    "self_reference_avg_sharess",
    "kw_max_avg",
    "self_reference_max_shares",
    "kw_min_avg",
    "self_reference_min_shares",
    "kw_avg_avg",
    "LDA_02",
    "LDA_04",
    "LDA_01",
    "LDA_03",
]

LASSO_FEATURES = ['data_channel_is_bus', 'global_subjectivity',
       'data_channel_is_entertainment', 'avg_positive_polarity',
       'max_negative_polarity', 'data_channel_is_tech',
       'n_non_stop_unique_tokens', 'data_channel_is_world',
       'data_channel_is_lifestyle', 'abs_title_subjectivity', 'LDA_00',
       'rate_positive_words', 'data_channel_is_socmed',
       'abs_title_sentiment_polarity', 'LDA_02', 'max_positive_polarity',
       'LDA_01', 'LDA_04', 'weekday_is_monday',
       'global_sentiment_polarity', 'is_weekend', 'title_subjectivity',
       'average_token_length', 'n_tokens_title', 'weekday_is_thursday',
       'weekday_is_saturday', 'weekday_is_friday', 'num_self_hrefs',
       'weekday_is_tuesday', 'min_negative_polarity', 'num_imgs',
       'num_hrefs', 'title_sentiment_polarity', 'num_keywords',
       'num_videos', 'weekday_is_wednesday', 'kw_min_min', 'kw_avg_avg',
       'kw_avg_min', 'kw_min_avg', 'n_tokens_content', 'kw_max_avg',
       'kw_max_min', 'self_reference_avg_sharess',
       'self_reference_min_shares', 'self_reference_max_shares',
       'kw_min_max', 'kw_avg_max', 'kw_max_max', 'rate_negative_words',
       'min_positive_polarity', 'avg_negative_polarity',
       'global_rate_negative_words', 'global_rate_positive_words',
       'n_non_stop_words', 'n_unique_tokens', 'weekday_is_sunday',
       'LDA_03']

MI_FEATURES = ['self_reference_avg_sharess', 'self_reference_min_shares',
       'self_reference_max_shares', 'kw_avg_avg', 'kw_max_avg',
       'kw_min_avg', 'LDA_02', 'num_self_hrefs', 'LDA_01',
       'data_channel_is_entertainment', 'LDA_03', 'LDA_04', 'is_weekend',
       'num_imgs', 'LDA_00', 'avg_negative_polarity', 'num_keywords',
       'kw_min_max', 'n_non_stop_words', 'kw_max_min', 'num_videos',
       'data_channel_is_tech', 'kw_avg_max', 'weekday_is_saturday',
       'n_tokens_content', 'max_positive_polarity', 'kw_avg_min',
       'weekday_is_friday', 'weekday_is_sunday', 'rate_negative_words',
       'kw_min_min', 'rate_positive_words', 'n_tokens_title',
       'data_channel_is_world', 'global_subjectivity',
       'min_positive_polarity', 'n_non_stop_unique_tokens', 'num_hrefs',
       'min_negative_polarity', 'global_sentiment_polarity',
       'n_unique_tokens', 'data_channel_is_socmed',
       'global_rate_positive_words', 'max_negative_polarity',
       'avg_positive_polarity', 'data_channel_is_bus',
       'average_token_length', 'abs_title_subjectivity', 'kw_max_max',
       'weekday_is_monday', 'title_sentiment_polarity',
       'global_rate_negative_words', 'title_subjectivity',
       'data_channel_is_lifestyle', 'weekday_is_thursday',
       'weekday_is_wednesday', 'weekday_is_tuesday',
       'abs_title_sentiment_polarity']

#MRMR_10_FEATURES = ['kw_avg_avg', 'max_positive_polarity', 'weekday_is_wednesday', 'max_negative_polarity', 'abs_title_subjectivity', 'data_channel_is_socmed', 'self_reference_min_shares', 'rate_negative_words', 'data_channel_is_lifestyle', 'weekday_is_saturday']


MRMR_5_FEATURES  = ['kw_avg_avg', 'max_positive_polarity', 'weekday_is_wednesday', 'max_negative_polarity', 'abs_title_subjectivity']
MRMR_10_FEATURES = ['kw_avg_avg', 'max_positive_polarity', 'weekday_is_wednesday', 'max_negative_polarity', 'abs_title_subjectivity', 'data_channel_is_socmed', 'self_reference_min_shares', 'rate_negative_words', 'data_channel_is_lifestyle', 'weekday_is_saturday']
MRMR_15_FEATURES = ['kw_avg_avg', 'max_positive_polarity', 'weekday_is_wednesday', 'max_negative_polarity', 'abs_title_subjectivity', 'data_channel_is_socmed', 'self_reference_min_shares', 'rate_negative_words', 'data_channel_is_lifestyle', 'weekday_is_saturday', 'LDA_01', 'num_imgs', 'weekday_is_sunday', 'num_videos', 'kw_min_max']
MRMR_20_FEATURES = ['kw_avg_avg', 'max_positive_polarity', 'weekday_is_wednesday', 'max_negative_polarity', 'abs_title_subjectivity', 'data_channel_is_socmed', 'self_reference_min_shares', 'rate_negative_words', 'data_channel_is_lifestyle', 'weekday_is_saturday', 'LDA_01', 'num_imgs', 'weekday_is_sunday', 'num_videos', 'kw_min_max', 'kw_min_min', 'n_tokens_title', 'weekday_is_friday', 'data_channel_is_bus', 'min_positive_polarity']

RUN_FEATURES = {"all_features":None, "5_features":MRMR_5_FEATURES, "10_features":MRMR_10_FEATURES, "15_features":MRMR_15_FEATURES, "20_features":MRMR_20_FEATURES}
