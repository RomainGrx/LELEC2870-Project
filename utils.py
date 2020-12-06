import os
from send2trash import send2trash
from contextlib import contextmanager

class AttributeDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

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

DATASETS_PATH = os.path.join(os.getcwd(), "datasets")
RESSOURCES_PATH = os.path.join(os.getcwd(), "ressources")
HTML_PATH = os.path.join(RESSOURCES_PATH, "html")
CSV_PATH = os.path.join(RESSOURCES_PATH, "csv")
IMAGES_PATH = os.path.join(RESSOURCES_PATH, "images")
PLOTS_PATH = os.path.join(RESSOURCES_PATH, "plots")