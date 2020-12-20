import os
import json
import json2latex
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Callable
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict

DATASETS_PATH = os.path.join(os.getcwd(), "datasets")
RESSOURCES_PATH = os.path.join(os.getcwd(), "ressources")
TEX_PATH = os.path.join(RESSOURCES_PATH, "tex")
JSON_PATH = os.path.join(RESSOURCES_PATH, "json")
HTML_PATH = os.path.join(RESSOURCES_PATH, "html")
CSV_PATH = os.path.join(RESSOURCES_PATH, "csv")
IMAGES_PATH = os.path.join(RESSOURCES_PATH, "images")
PLOTS_PATH = os.path.join(RESSOURCES_PATH, "plots")

import tikzplotlib
from send2trash import send2trash
from contextlib import contextmanager

class AttributeDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    def copy(self):
        d = AttributeDict()
        for k, v in self.items():
            if hasattr(v, "copy"):
                d[k] = v.copy()
        return d

class SnsPalette:
    def __init__(self, color):
        self._color = color
    def __call__(self, n):
        return np.array(sns.color_palette(self._color, n_colors=n).as_hex())

@contextmanager
def output_dir(path:str, erase=False):
    if not os.path.exists(path):
        os.mkdir(path)
    elif os.path.isdir(path):
        only_fnames = [fname for fname in os.listdir(path) if not os.path.isdir(os.path.join(path, fname))]
        if len(only_fnames) > 0 and erase:
            print(f"Erase from {path} all these files : \n--> ", "\n--> ".join(only_fnames))
            for fname in only_fnames:
                all_path = os.path.join(path, fname)
                send2trash(all_path)

    yield path

def plot_result_over_truth(y_true, y_pred):
    plt.plot(y_true, y_pred)


def sort_df_features(df):
    from preprocess import ALL_FEATURES

    testdesc = df.copy()
    sorter = dict(zip(ALL_FEATURES, range(len(ALL_FEATURES))))
    testdesc["sorter"] = testdesc.index.map(sorter)
    testdesc.sort_values("sorter", inplace=True)
    testdesc = testdesc.drop("sorter", axis=1).T

    return testdesc

def plotsave(filename):
    with output_dir(PLOTS_PATH) as path:
        plt.savefig(os.path.join(path, filename), transparent=True)

def tiksave(filename):
    with output_dir(TEX_PATH) as path:
        tikzplotlib.save(os.path.join(path, filename))

def jsonsave(data, filename):
    with output_dir(JSON_PATH) as path:
        with open(os.path.join(path, filename), 'w') as fd:
            fd.write(json.dumps(data, indent=4))

def dumptex(data, filename):
    if not filename.split(".")[-1] == "tex":
        filename = filename+".tex"
    with output_dir(TEX_PATH) as path:
        with open(os.path.join(path, filename), 'w') as fd:
            json2latex.dump(filename, data, fd)

def run_test(model, fit_kwargs=None, predict_kwargs=None, seed=None, save_json=False, save_tex=False, df=False, verbose=False, position=0):
    from preprocess import get_dataset, preprocess_all, dataset_to_X_y, RUN_FEATURES
    from metrics import score_regression
    if not callable(model):
        modelfn = lambda:model
    else:
        modelfn = model
    _model = modelfn()
    assert hasattr(_model, "fit") and hasattr(_model, "predict")
    fit_kwargs = fit_kwargs or dict(); predict_kwargs = predict_kwargs or dict()
    dataset = get_dataset(seed=seed)
    d = dict()
    dic_feat = RUN_FEATURES.items()
    if verbose:
        dic_feat = tqdm(dic_feat, desc="run", position=position)
    for k, v in dic_feat:
        _model = modelfn()
        run_data = preprocess_all(dataset, subset=v)
        X_train, y_train, X_val, y_val = dataset_to_X_y(run_data, keys=["train", "validation"], datatype="numpy")
        _model.fit(X_train, y_train, **fit_kwargs)
        y_train_hat = _model.predict(X_train, **predict_kwargs)
        train_loss = score_regression(y_train, y_train_hat)
        y_val_hat = _model.predict(X_val, **predict_kwargs)
        val_loss = score_regression(y_val, y_val_hat)
        d[k] = dict(train_loss=train_loss, val_loss=val_loss)
    if save_json:
        jsonsave(d, _model.__class__.__name__+".json")
    if save_tex:
        dumptex(d, _model.__class__.__name__+".tex")
    if not df:
        return d
    return pd.DataFrame(d)


def run_search_test(model_class:Callable, arg, values, predict_kwargs=None, fit_kwargs=None, seed=None, save_json=False, df=False, verbose=False, position=0):
    d = dict()
    if verbose:
        values = tqdm(values, desc="search run", position=position)
    for v in values:
        kwargs = {arg:v}
        v_model = model_class(**kwargs)
        v_score = run_test(v_model, fit_kwargs, predict_kwargs, seed=seed, save_json=False, save_tex=False, df=False, verbose=verbose, position=position+1)
        d[str(v)] = v_score
    if save_json:
        name = f"{v_model.__class__.__name__}_{arg}.json"
        jsonsave(d, name)
    if not df:
        return d
    return pd.DataFrame(d)



def plot_search(name, arg, run_dict, allinfos=False, semilogy=False, save_png=False, save_tex=False):
    if isinstance(run_dict, pd.DataFrame): run_dict = run_dict.to_dict()
    plt.figure(figsize=(12, 9))
    plt.grid(True)
    if semilogy:
        plt.semilogy()
    plt.title(name)
    plt.xlabel(f"{arg}")
    plt.ylabel("Regression score")

    x_values = []
    train_scores = defaultdict(list)
    val_scores = defaultdict(list)
    for k, v in run_dict.items():
        x_values.append(k)
        for krun, vrun in v.items():
            train_scores[krun].append(vrun["train_loss"])
            val_scores[krun].append(vrun["val_loss"])

    train_palette = SnsPalette("flare")(len(train_scores.values()))
    val_palette = SnsPalette("husl")(len(val_scores.values()))


    if allinfos:
        for idx, feat in enumerate(train_scores.keys()):
            plt.plot(x_values, train_scores[feat], color=train_palette[idx], ls="--", alpha=.75, label=f"train_{feat}")

    for idx, feat in enumerate(val_scores.keys()):
        plt.plot(x_values, val_scores[feat], color=val_palette[idx], ls="-", alpha=.75, label=f"val_{feat}" if allinfos else feat)

    plt.legend()

    name = f"{name}_{arg}"
    if save_png:
        plotsave(name+".png")
    if save_tex:
        tiksave(name+".tex")


def get_best_score_from_run(run_dict):
    best_score = None
    d = dict()
    for k, value in run_dict.items():
        for feat, sc in value.items():
            if best_score is None or sc["val_loss"] > best_score:
                best_score = sc["val_loss"]
                d["score"] = best_score
                d["n_feature"] = feat
                d["key"] = k
    return d


def get_best_score_from_run_all_features(run_dict):
    best_score = defaultdict(lambda:-1)
    d = defaultdict(dict)
    for k, value in run_dict.items():
        for feat, sc in value.items():
            if sc["val_loss"] > best_score[feat]:
                best_score[feat] = sc["val_loss"]
                d[feat]["score"] = best_score[feat]
                #d[feat]["n_feature"] = feat
                d[feat]["key"] = k
    return d

