# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/3/7 16:44'

import numpy as np
import os

# to make output stable across runs
np.random.seed(42)

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rc("axes",labelsize=14)
mpl.rc("xtick",labelsize=12)
mpl.rc("ytick",labelsize=12)

# where to save the figure
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "end_to_end_project"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR,"images",CHAPTER_ID)

def save_fig(fig_id,tight_layout=True,fig_extension="png",resolution=300):
    path = os.path.join(IMAGES_PATH,fig_id+"."+fig_extension)
    print("Saving figure",fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path,format=fig_extension,dpi=resolution)

import warnings
warnings.filterwarnings(action="ignore",message="internal gelsd")

import pandas as pd
ABSOULTE_PATH = os.path.abspath(__file__)
HOUSING_PATH = os.path.join("datasets", "housing")
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path,"housing.csv")
    return pd.read_csv(csv_path)
