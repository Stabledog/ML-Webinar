# %% boston_loader.py
''' Provides a patched-up loader for boston housing data which was removed from sklearn in 2018 '''
import csv
from sklearn.utils import Bunch
import numpy as np

# %%

DATA_ROOT = "../data"  # Caller can set this at runtime
import os, sys
print(f"stub pwd={os.getcwd()}",file=sys.stderr)

def set_data_root(r:str):
    global DATA_ROOT
    DATA_ROOT=r


# %%
def load_descr(descr_file_name, *, descr_module=''):
    #fdescr = resources.read_text(descr_module, descr_file_name)
    return open(descr_file_name,"r").read()

# %%
def load_boston(*, return_X_y=False):
    r"""Load and return the Boston house-prices dataset (regression).

    ==============   ==============
    Samples total               506
    Dimensionality               13
    Features         real, positive
    Targets           real 5. - 50.
    ==============   ==============

    Read more in the :ref:`User Guide <boston_dataset>`.


    Parameters
    ----------
    return_X_y : bool, default=False
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

        .. versionadded:: 0.18

    Returns
    -------
    data : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.

        data : ndarray of shape (506, 13)
            The data matrix.
        target : ndarray of shape (506,)
            The regression target.
        filename : str
            The physical location of boston csv dataset.

            .. versionadded:: 0.20

        DESCR : str
            The full description of the dataset.
        feature_names : ndarray
            The names of features

    (data, target) : tuple if ``return_X_y`` is True
        A tuple of two ndarrays. The first contains a 2D array of shape (506, 13)
        with each row representing one sample and each column representing the features.
        The second array of shape (506,) contains the target samples.

        .. versionadded:: 0.18

    Notes
    -----
        .. versionchanged:: 0.20
            Fixed a wrong data point at [445, 0].


    Examples
    --------
    >>> import warnings
    >>> from sklearn.datasets import load_boston
    >>> with warnings.catch_warnings():
    ...     # You should probably not use this dataset.
    ...     warnings.filterwarnings("ignore")
    ...     X, y = load_boston(return_X_y=True)
    >>> print(X.shape)
    (506, 13)
    """
    data_file_name = f"{DATA_ROOT}/boston_house_prices.csv"
    rst_url = f"{DATA_ROOT}/boston_house_prices.rst"

    descr_text = load_descr(rst_url)

    #with resources.open_text(DATA_MODULE, data_file_name) as f:
    with open(data_file_name,"r") as f:
        data_file = csv.reader(f)
        temp = next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,))
        temp = next(data_file)  # names of features
        feature_names = np.array(temp)

        for i, d in enumerate(data_file):
            data[i] = np.asarray(d[:-1], dtype=np.float64)
            target[i] = np.asarray(d[-1], dtype=np.float64)

    if return_X_y:
        return data, target

    return Bunch(
        data=data,
        target=target,
        # last column is target value
        feature_names=feature_names[:-1],
        DESCR=descr_text,
        filename=data_file_name,
        data_module="data"
    )

# %%

def tryit():
    boston_bunch=load_boston()
    print(boston_bunch.data)
    print(boston_bunch.feature_names)
    print(boston_bunch.DESCR)


