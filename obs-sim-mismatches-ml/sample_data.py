#!/usr/bin/env python
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, make_scorer, roc_curve
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.inspection import permutation_importance

livetime = 90072247.26
n_events = 100000
path = '/net/big-tank/POOL/users/mboerner/multiyear_unfolding/level5_more_features.hdf5'
obs_keys = ['/data_IC86III_IC86II_L5', '/data_IC86II_IC86II_L5', '/data_IC86IV_IC86II_L5']
sim_key = '/numu_11374_clsimbase4_0_3_0_99_eff_L5'

random_state = np.random.RandomState(0)

def main():
    with pd.HDFStore(path, mode='r') as f:

        obs = pd.concat([f[k] for k in obs_keys])
        obs_fraction = n_events / len(obs)
        obs = obs.sample(n_events, random_state=random_state)

        sim = f[sim_key]
        sim_fraction = n_events / len(sim)
        sim = sim.sample(n_events, random_state=random_state)


    sim = sim.astype(np.float32)
    obs = obs.astype(np.float32)

    # remove rows with invalid values
    with pd.option_context("mode.use_inf_as_na", True):
        sim.dropna(axis='index', how='any', inplace=True)
        obs.dropna(axis='index', how='any', inplace=True)


    sim['weight'] = sim['weight_full_aachen'] * livetime * obs_fraction / sim_fraction

    # normalization is a bit off, let's force the same
    sim['weight'] *= len(obs) / sim['weight'].sum()
    obs['weight'] = 1.0

    sim['obs_sim_label'] = 1
    obs['obs_sim_label'] = 0

    cols = set(sim.columns).intersection(set(obs.columns))
    df = pd.concat([sim.loc[:, cols], obs.loc[:, cols]])

    # drop constant columns
    df.drop(df.columns[df.std(axis=0) == 1], axis=1, inplace=True)

    # shuffle
    df = df.sample(frac=1).reset_index(drop=True)
    df.to_hdf('./icecube_obs_sim_data.hdf5', 'events', complevel=5, complib='blosc:zstd')


if __name__ == '__main__':
    main()
