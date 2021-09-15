#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, make_scorer, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.inspection import permutation_importance
from pathlib import Path

outdir = Path('./build/obs-sim-mismatches-ml')
outdir.mkdir(exist_ok=True, parents=True)

random_state = np.random.RandomState(0)

df = pd.read_hdf('./obs-sim-mismatches-ml/icecube_obs_sim_data.hdf5', 'events')



# plot some features
print('Plotting features')

labels = ['Simulations', 'Observations']

col = 'HitStatisticsValuesIC.q_tot_pulses'
low, high = df[col].quantile([1e-3, 1 - 1e-3])
bins = np.geomspace(low, high, 101)

hist_opts = dict(bins=bins, histtype='step')
fig, ax = plt.subplots(constrained_layout=True)
for label, group in df.groupby('obs_sim_label'):
    ax.hist(group[col], weights=group['weight'], **hist_opts, label=labels[label])

ax.legend()
ax.set(xscale='log', xlabel=col)
fig.savefig(outdir / (col + '.pdf'))


col = 'SplineMPEDirectHitsICE.n_dir_pulses'
low, high = df[col].quantile([1e-3, 1 - 1e-3])
bins = np.arange(low, high)
hist_opts = dict(bins=bins, histtype='step')
fig, ax = plt.subplots(constrained_layout=True)
for label, group in df.groupby('obs_sim_label'):
    ax.hist(group[col], weights=group['weight'], **hist_opts, label=labels[label])

ax.legend()
ax.set(xscale='log', xlabel=col)
fig.savefig(outdir / (col + '.pdf'))



print('Learning Obs/Sim mismatches')
clf = RandomForestClassifier(n_estimators=50, n_jobs=-1, max_depth=15, random_state=random_state)


X = df.drop(['obs_sim_label', 'weight'], axis=1)
y = df['obs_sim_label']
sample_weight = df['weight']


most_important = []
iterations = []
to_remove = 4

for i in range(4):
    print('Iteration: ', i)
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)

    X.drop(most_important, axis=1, inplace=True)

    feature_importances = []
    rocs = []
    roc_aucs = []
    predictions = []

    for train, test in cv.split(X, y):

        X_test, y_test, weight_test = X.loc[test], y.loc[test], sample_weight.loc[test]
        X_train, y_train, weight_train = X.loc[train], y.loc[train], sample_weight.loc[train]

        clf.fit(X_train, y_train, sample_weight=weight_train)

        prediction = clf.predict_proba(X_test)[:, 1]
        predictions.append(prediction)

        rocs.append(roc_curve(y_test, prediction, sample_weight=weight_test))
        roc_aucs.append(roc_auc_score(y_test, prediction, sample_weight=weight_test))

        feature_importances.append(clf.feature_importances_)


    importance = pd.Series(np.mean(feature_importances, axis=0), index=X.columns)
    most_important = importance.sort_values(ascending=False).head(to_remove).index

    iterations.append({
        'importances': feature_importances,
        'roc_aucs': roc_aucs,
        'rocs': rocs,
        'features': X.columns.tolist(),
        'predictions': predictions,
    })


print('Plotting feature importance')
fig, ax = plt.subplots(constrained_layout=True, figsize=(9, 6))
it = iterations[0]
importance = pd.Series(np.mean(it['importances'], axis=0), index=it['features'])
importance.sort_values().tail(20).plot.barh(ax=ax)
ax.set_xlabel('Random Forest Feature Importance')
fig.savefig(outdir / 'feature_importance.pdf')



print('Plotting roc curves')
fig, ax = plt.subplots(constrained_layout=True)
ax.set_aspect(1)
ax.axline([0, 0], [1, 1], color='k', alpha=0.2)

for i, iteration in enumerate(iterations):
    for cv, (fpr, tpr, threshold) in enumerate(iteration['rocs']):
        n_removed = len(iterations[0]['features']) - len(iteration['features'])
        roc_auc = np.mean(iteration['roc_aucs'])
        roc_auc_std = np.std(iteration['roc_aucs'])

        label = rf'{n_removed}: $A_\mathrm{{ROC}} = {roc_auc:.3f}\pm{roc_auc_std:.3f}$' if cv == 0 else None
        ax.plot(fpr, tpr, color=f'C{i}', alpha=0.2, label=label)

ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.legend(title='Features removed:')
ax.set(xlim=(0, 1), ylim=(0, 1))
fig.savefig(outdir / 'roc_curves.pdf')


print('Plotting score distributions')
fig, ax = plt.subplots(constrained_layout=True)
hist_opts = dict(bins=100, range=[0, 1], histtype='step')

for i, iteration in enumerate(iterations):
    n_removed = len(iterations[0]['features']) - len(iteration['features'])
    label = rf'{n_removed}'
    ax.hist(np.concatenate(iteration['predictions']), **hist_opts, label=label, color=f'C{i}')

ax.legend(title='Features removed:', ncol=4, bbox_to_anchor=(0.5, 1.01), loc='lower center')
ax.set_xlabel('Prediction Score')
ax.xaxis.set_major_locator(plt.MultipleLocator(0.25))

ax.annotate(
    xy=(0.01, 0.9),
    xytext=(0.1, 0.9),
    text='Likely Observations',
    textcoords='axes fraction',
    xycoords='axes fraction',
    verticalalignment='center',
    horizontalalignment='left',
    arrowprops=dict(facecolor='black', shrink=0.05),
)

ax.annotate(
    xy=(0.99, 0.9),
    xytext=(0.9, 0.9),
    text='Likely Simulations',
    textcoords='axes fraction',
    xycoords='axes fraction',
    arrowprops=dict(facecolor='black', shrink=0.05),
    verticalalignment='center',
    horizontalalignment='right',
)
ax.set_xlim(0, 1.0)

fig.savefig(outdir / 'score_distributions.pdf')
