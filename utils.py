# @ARTICLE{10296014,
#          author={Tobin, Joshua and Zhang, Mimi},
# journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
# title={A Theoretical Analysis of Density Peaks Clustering and the Component-Wise Peak-Finding Algorithm},
# year={2024},
# volume={46},
# number={2},
# pages={1109-1120},
# doi={10.1109/TPAMI.2023.3327471}}
# https://github.com/tobinjo96/CPFcluster


import numpy as np
import sys
import os



from sklearn.metrics.pairwise import euclidean_distances

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def density_broad_search_star(a_b):
    try:
        return euclidean_distances(a_b[1],a_b[0])
    except Exception as e:
        raise Exception(e)