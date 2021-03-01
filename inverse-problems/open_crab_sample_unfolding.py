from fact.io import read_h5py
from fact.analysis.statistics import li_ma_significance

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from functools import lru_cache, partial
from scipy.optimize import minimize
from numdifftools import Hessian

INVALID = np.finfo(np.float32).max
EPS = 1e-10

# mc information
scatter_radius = 270   # Maximaler simulierter Abstand der Schauer zum Teleskope
sample_fraction = 0.7  # Anteil des Testdatensatzer an der Gesamtzahl simulierten Schauer
area = np.pi * scatter_radius**2

# binning
n_bins_true = 5
n_bins_est = 10

e_min_est = 700
e_min_true = 700
e_max_est = 15e3
e_max_true = 15e3


@lru_cache()
def C_matrix(n):
    I = np.eye(n)
    C = 2.0 * I - np.roll(I, 1) - np.roll(I, -1)
    return C


def llh_poisson(f_est, A, g, b):
    if np.any(f_est < 0):
        return INVALID

    lambda_ = A @ f_est + b

    return np.sum(lambda_ - g * np.log(lambda_ + EPS))


def tikhonov_reg(f_est, tau, effective_area):
    # we ignore under and overflow for the regularization
    C = C_matrix(len(f_est) - 2)

    # we regularize on the log of f with acceptance correction,
    # since only that is expected to be flat
    return tau * np.sum((C @ np.log(f_est[1:-1] / effective_area[1:-1] + EPS)) ** 2)

def llh_poisson_tikhonov(f_est, A, g, b, tau, effective_area):
    if np.any(f_est < 0):
        return INVALID

    return llh_poisson(f_est, A, g, b) + tikhonov_reg(f_est, tau, effective_area)


def mean_correlation(cov):
    cov_inv = np.linalg.inv(cov)
    return np.mean(
        np.sqrt(1 - 1 / (np.diag(cov) * np.diag(cov_inv)))
    )


def unfold(A, g, b, tau, a_eff):
    # allow only positive values
    bounds = [[1e-15, None] for _ in range(len(a_eff))]

    # uniform initial guess
    initial_guess = np.full(len(a_eff), 50)

    nllh = partial(
        llh_poisson_tikhonov,
        A=A, g=g, b=b,
        tau=tau, effective_area=a_eff
    )

    result = minimize(nllh, x0=initial_guess, bounds=bounds)
    hesse = Hessian(nllh)

    cov = np.linalg.inv(hesse(result.x))

    assert result.success
    return result.x, cov



if __name__ == '__main__':
    bins_e_true = np.logspace(np.log10(e_min_true), np.log10(e_max_true), n_bins_true + 1)
    bins_e_est = np.logspace(np.log10(e_min_est), np.log10(e_max_est), n_bins_est + 1)
    bins_e_true = np.concatenate([[-np.inf], bins_e_true, [np.inf]])
    bins_e_est = np.concatenate([[-np.inf], bins_e_est, [np.inf]])

    bin_centers = 0.5 * (bins_e_true[1:-2] + bins_e_true[2:-1])
    bin_width = np.diff(bins_e_true)[1:-1]


    print('Reading in data')
    gammas = read_h5py('build/inverse-problems/gamma_test_dl3.hdf5', key='events', columns=[
        'gamma_energy_prediction',
        'gamma_prediction',
        'theta_deg',
        'corsika_event_header_event_number',
        'corsika_event_header_total_energy',
    ])


    gammas_corsika = read_h5py(
        'build/inverse-problems/gamma_corsika_headers.hdf5',
        key='corsika_events',
        columns=['total_energy'],
    )


    crab_events = read_h5py('build/inverse-problems/open_crab_sample_dl3.hdf5', key='events', columns=[
        'gamma_prediction',
        'gamma_energy_prediction',
        'theta_deg',
        'theta_deg_off_1',
        'theta_deg_off_2',
        'theta_deg_off_3',
        'theta_deg_off_4',
        'theta_deg_off_5',
    ])

    crab_runs = read_h5py('build/inverse-problems/open_crab_sample_dl3.hdf5', key='runs')


    print('Applying event selection')
    on_time = crab_runs['ontime'].sum()
    prediction_threshold = 0.8
    theta_cut = np.sqrt(0.025)

    on_query = f'gamma_prediction > {prediction_threshold} and theta_deg <= {theta_cut}'
    gammas = gammas.query(on_query).copy()
    crab_on = crab_events.query(on_query).copy()

    # concancenate each of the off regions
    crab_off = []
    for i in range(1, 6):
        off_query = f'gamma_prediction > {prediction_threshold} and theta_deg_off_{i} <= {theta_cut}'
        crab_off.append(crab_events.query(off_query))

    crab_off = pd.concat(crab_off)
    off_weights = np.full(len(crab_off), 0.2)

    n_on = len(crab_on)
    n_off = len(crab_off)

    print(f"n_on={n_on}, n_off={n_off}, σ={li_ma_significance(n_on, n_off, 0.2):.1f}")


    print('Calculating response')
    M, _, _ = np.histogram2d(
        gammas['gamma_energy_prediction'],
        gammas['corsika_event_header_total_energy'],
        bins=[bins_e_est, bins_e_true],
    )

    M = M / M.sum(axis=0)

    print('Calculating effective area')
    n_detected, _ = np.histogram(gammas['corsika_event_header_total_energy'], bins=bins_e_true)
    n_simulated, _ = np.histogram(gammas_corsika['total_energy'], bins=bins_e_true)
    a_eff = (n_detected / sample_fraction) / n_simulated * area


    print('Plotting response')
    fig, (ax1, ax2) = plt.subplots(1, 2)
    img = ax1.matshow(M, cmap='inferno')
    ax1.set_xlabel(r'$E$-bin')
    ax1.xaxis.set_label_position('top')
    ax1.set_ylabel(r'$\hat{E}$-bin')
    fig.colorbar(img, ax=ax1)

    ax2.errorbar(bin_centers, a_eff[1:-1], xerr=bin_width / 2, linestyle='')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_ylabel(r'$A_\text{eff} \mathbin{\si{\meter\squared}}')
    ax2.set_xlabel(r'$E \mathbin{/} \si{\GeV}$')
    ax2.set_ylim(1e3, 1e5)

    fig.savefig('build/inverse-problems/fact_response.pdf')
    plt.close('all')

    g, _ = np.histogram(crab_on['gamma_energy_prediction'], bins=bins_e_est)
    b, _ = np.histogram(crab_off['gamma_energy_prediction'], bins=bins_e_est, weights=np.full(n_off, 0.2))

    print('Unfolding for many taus to find best')
    taus = np.logspace(-1.5, 1.5, 100)
    correlations = []
    results = []
    covs = []

    for tau in taus:
        f, cov = unfold(M, g, b, tau, a_eff)

        results.append(f)
        covs.append(cov)
        correlations.append(mean_correlation(cov))


    # best_index = np.argmin(np.abs(taus - 0.1))
    best_index = np.argmin(correlations)
    f = results[best_index]
    cov = covs[best_index]



    print('plotting best result')
    fig, ax =  plt.subplots()
    ax.plot(taus, correlations, '.')
    ax.axvline(taus[best_index], color='C1')
    ax.set_xlabel(r'$\tau$')
    ax.set_ylabel('Mean Correlation')
    ax.set_xscale('log')
    fig.savefig('build/inverse-problems/tau_vs_correlation.pdf')
    plt.close('all')



    norm = 1 / (a_eff[1:-1] * 1e4) / on_time / (bin_width / 1000)
    e_plot = np.logspace(2.7, 4.2, 100)

    fig, ax =  plt.subplots()

    ax.plot(
        e_plot,
        3.23e-11 * (e_plot / 1000)**(-2.47 - 0.24 * np.log10(e_plot / 1000)),
        label='MAGIC, JHEAP 2015',
        color='k'
    )

    ax.errorbar(
        bin_centers,
        f[1:-1] * norm,
        xerr=bin_width / 2,
        yerr=np.sqrt(np.diag(cov))[1:-1] * norm,
        ls='none',
        label='Unfolding',
        zorder=10,
    )

    ax.legend()
    ax.set_xlabel('E / GeV')
    ax.set_ylabel('Flux / (cm⁻² s⁻¹ GeV⁻¹)')
    ax.set_yscale('log')
    ax.set_xscale('log')
    fig.savefig('build/inverse-problems/fact_unfolding.pdf')
