"""Physical kinetic energy and shell-integrated spectra on periodic cubic grids."""
from functools import lru_cache
from pathlib import Path

import numpy as np


@lru_cache(maxsize=4)
def spectral_geometry(side):
    modes = np.fft.fftfreq(side) * side
    last = np.fft.rfftfreq(side) * side
    radius = np.sqrt(modes[:, None, None]**2 + modes[None, :, None]**2 + last[None, None, :]**2)
    shells = np.floor(radius + 0.5).astype(np.int64)
    weights = np.full(len(last), 2.0)
    weights[0] = 1.0
    if side % 2 == 0:
        weights[-1] = 1.0
    return shells, weights[None, None, :], int(shells.max()) + 1


def energy_spectrum(velocity):
    """Return K and E[m]; sum E[m] = K = .5*mean_space(u²+v²+w²).

    m is integer Fourier mode magnitude, rounded to its nearest shell. No
    physical domain length is assumed. Includes the mean-flow shell m=0.
    """
    velocity = np.asarray(velocity, dtype=np.float64)
    if (velocity.ndim != 4 or velocity.shape[0] != 3
            or len(set(velocity.shape[1:])) != 1 or velocity.shape[1] < 2):
        raise ValueError('Expected three velocity components on a cubic grid')
    if not np.isfinite(velocity).all():
        raise ValueError('Nonfinite velocity in energy diagnostic')
    side = velocity.shape[1]
    shells, weights, count = spectral_geometry(side)
    transform = np.fft.rfftn(velocity, axes=(-3, -2, -1))
    power = .5 * (transform.real**2 + transform.imag**2).sum(axis=0) / side**6
    spectrum = np.bincount(shells.ravel(), weights=(power * weights).ravel(), minlength=count)
    kinetic = float(.5 * np.mean(np.sum(velocity**2, axis=0)))
    if not np.isclose(spectrum.sum(), kinetic, rtol=1e-10, atol=1e-12):
        raise ArithmeticError('Spectrum violates Parseval energy check')
    return kinetic, spectrum


def plot_energy(report, arrays, output):
    """Plot common-origin means and an actual single forecast; no error bands."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output = Path(output)
    label = report['model']
    leads = arrays['lead_steps']
    actual, predicted = arrays['actual_energy'], arrays['predicted_energy']
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(leads, actual.mean(axis=0), 'o-', label='Actual')
    ax.plot(leads, predicted.mean(axis=0), 's--', label=label)
    ax.set(xlabel='Forecast lead (saved steps)', ylabel='Volume-mean kinetic energy',
           title=f'{label}: mean over {len(actual)} common test origins')
    ax.set_xticks(leads); ax.legend(); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(output / 'energy-evolution.png', dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(arrays['times'][0], actual[0], 'o-', label='Actual')
    ax.plot(arrays['times'][0], predicted[0], 's--', label=label)
    ax.set(xlabel='Recorded simulation time', ylabel='Volume-mean kinetic energy',
           title=f'{label}: first test forecast (origin plus five predictions)')
    ax.legend(); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(output / 'energy-example.png', dpi=180); plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    modes = arrays['modes']
    # Beyond the axial Nyquist radius shells contain only a subset of directions.
    resolved = (modes > 0) & (modes <= arrays['grid_side'].item() // 2)
    for lead, ax in enumerate(axes.flat):
        ax.loglog(modes[resolved], arrays['actual_spectrum'][lead, resolved], label='Actual')
        ax.loglog(modes[resolved], arrays['predicted_spectrum'][lead, resolved], '--', label=label)
        ax.set(title='Observed origin' if lead == 0 else f'Forecast lead {lead}',
               xlabel='Fourier mode magnitude m', ylabel='Shell kinetic energy E[m]')
        ax.grid(alpha=.25); ax.legend(fontsize=8)
    fig.suptitle(f'{label}: spectra averaged over common test origins; m=0 omitted')
    fig.tight_layout(); fig.savefig(output / 'energy-spectra.png', dpi=180); plt.close(fig)


def plot_comparison(reports, arrays, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    first = next(iter(arrays.values()))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(first['lead_steps'], first['actual_energy'].mean(axis=0), 'ko-', label='Actual')
    for name, value in arrays.items():
        ax.plot(value['lead_steps'], value['predicted_energy'].mean(axis=0), 's--', label=name)
    ax.set(xlabel='Forecast lead (saved steps)', ylabel='Volume-mean kinetic energy',
           title=f'MATEY / Yalamanchi: {len(first["actual_energy"])} common test origins')
    ax.set_xticks(first['lead_steps']); ax.grid(alpha=.25); ax.legend()
    fig.tight_layout(); fig.savefig(Path(output) / 'energy-comparison.png', dpi=180); plt.close(fig)
