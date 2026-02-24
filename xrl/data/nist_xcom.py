"""
NIST XCOM Tabulated Mass Attenuation Coefficients
==================================================

Embedded tables of μ/ρ (cm²/g) vs photon energy (MeV) sourced directly from:
  NIST Physical Reference Data — X-Ray Mass Attenuation Coefficients
  https://physics.nist.gov/PhysRefData/XrayMassCoef/

Edge conventions
----------------
At each absorption edge the NIST table lists two rows with the same energy
value (below-edge and above-edge).  Here the above-edge row is stored at
``energy + _EDGE_EPS`` so that all energy arrays are strictly increasing
and ``interp1d`` works without modification.

_EDGE_EPS = 1e-6 MeV = 1 eV — negligible for XRL physics (features change
on 10 eV scale or more).

Author: Abhineet Agarwal
Data retrieved: 2026-02-24
"""

import numpy as np
from scipy.interpolate import interp1d

_EDGE_EPS = 1e-8   # MeV  (0.01 eV) — used to split edge-pair rows;
                   # small enough that queries 0.1 eV above/below an edge
                   # return the correct side, yet distinct for float64 and np.unique

# ---------------------------------------------------------------------------
# Elemental tables  (energy in MeV, mu/rho in cm²/g, total w/ coherent)
# All values from NIST XCOM (physics.nist.gov/PhysRefData/XrayMassCoef/)
# ---------------------------------------------------------------------------

# Hydrogen (Z=1)
_H = {
    'energy_MeV': np.array([
        1.00e-3, 1.50e-3, 2.00e-3, 3.00e-3, 4.00e-3, 5.00e-3,
        6.00e-3, 8.00e-3, 1.00e-2,
    ]),
    'mu_rho': np.array([
        7.217, 2.148, 1.059, 5.612e-1, 4.546e-1, 4.193e-1,
        4.042e-1, 3.914e-1, 3.854e-1,
    ]),
}

# Carbon (Z=6)
_C = {
    'energy_MeV': np.array([
        1.00e-3, 1.50e-3, 2.00e-3, 3.00e-3, 4.00e-3, 5.00e-3,
        6.00e-3, 8.00e-3, 1.00e-2,
    ]),
    'mu_rho': np.array([
        2.211e+3, 7.002e+2, 3.026e+2, 9.033e+1, 3.778e+1, 1.912e+1,
        1.095e+1, 4.576e+0, 2.373e+0,
    ]),
}

# Nitrogen (Z=7)  — no edges in 1–10 keV range (K-edge at 0.410 keV)
_N = {
    'energy_MeV': np.array([
        1.00e-3, 1.50e-3, 2.00e-3, 3.00e-3, 4.00e-3, 5.00e-3,
        6.00e-3, 8.00e-3, 1.00e-2,
    ]),
    'mu_rho': np.array([
        3.311e+3, 1.083e+3, 4.769e+2, 1.456e+2, 6.166e+1, 3.144e+1,
        1.809e+1, 7.562e+0, 3.879e+0,
    ]),
}

# Oxygen (Z=8)  — K-edge at 0.532 keV, below XRL range
_O = {
    'energy_MeV': np.array([
        1.00e-3, 1.50e-3, 2.00e-3, 3.00e-3, 4.00e-3, 5.00e-3,
        6.00e-3, 8.00e-3, 1.00e-2,
    ]),
    'mu_rho': np.array([
        4.590e+3, 1.549e+3, 6.949e+2, 2.171e+2, 9.315e+1, 4.790e+1,
        2.770e+1, 1.163e+1, 5.952e+0,
    ]),
}

# Silicon (Z=14)
# K-edge at 1.8389 keV — below-edge row stored at 1.83890e-3,
#                        above-edge row stored at 1.83890e-3 + _EDGE_EPS
_SI = {
    'energy_MeV': np.array([
        1.00e-3, 1.50e-3,
        1.83890e-3, 1.83890e-3 + _EDGE_EPS,   # K edge: 309.2 → 3192
        2.00e-3, 3.00e-3, 4.00e-3, 5.00e-3,
        6.00e-3, 8.00e-3, 1.00e-2,
    ]),
    'mu_rho': np.array([
        1.570e+3, 5.355e+2,
        3.092e+2, 3.192e+3,                    # K edge
        2.777e+3, 9.784e+2, 4.529e+2, 2.450e+2,
        1.470e+2, 6.468e+1, 3.389e+1,
    ]),
}

# Tantalum (Z=73), density 16.65 g/cm³ (NIST)  /  16.6 g/cm³ (MATERIALS dict)
# M-edges (keV): M5=1.7351, M4=1.7932, M3=2.1940, M2=2.4687, M1=2.7080
# Full M-edge structure from NIST XCOM Z=73
_TA = {
    'energy_MeV': np.array([
        1.00000e-3, 1.50000e-3,
        1.73510e-3, 1.73510e-3 + _EDGE_EPS,   # M5: 1154 → 1417
        1.76391e-3,                             # between M5 and M4
        1.79320e-3, 1.79320e-3 + _EDGE_EPS,   # M4: 3082 → 3421
        2.00000e-3,
        2.19400e-3, 2.19400e-3 + _EDGE_EPS,   # M3: 2985 → 3464
        2.32730e-3,
        2.46870e-3, 2.46870e-3 + _EDGE_EPS,   # M2: 2604 → 2768
        2.58558e-3,
        2.70800e-3, 2.70800e-3 + _EDGE_EPS,   # M1: 2233 → 2329
        3.00000e-3, 4.00000e-3, 5.00000e-3,
        6.00000e-3, 8.00000e-3, 9.88110e-3,
    ]),
    'mu_rho': np.array([
        3.510e+3, 1.566e+3,
        1.154e+3, 1.417e+3,                    # M5 edge
        2.053e+3,                               # between M5 and M4
        3.082e+3, 3.421e+3,                    # M4 edge
        3.771e+3,
        2.985e+3, 3.464e+3,                    # M3 edge
        3.003e+3,
        2.604e+3, 2.768e+3,                    # M2 edge
        2.486e+3,
        2.233e+3, 2.329e+3,                    # M1 edge
        1.838e+3, 9.222e+2, 5.328e+2,
        3.382e+2, 1.639e+2, 9.599e+1,
    ]),
}

# Tungsten (Z=74), density 19.30 g/cm³
# M-edges (keV): M5=1.8092, M4=1.8716, M3=2.2810, M2=2.5749, M1=2.8196
_W = {
    'energy_MeV': np.array([
        1.00000e-3, 1.50000e-3,
        1.80920e-3, 1.80920e-3 + _EDGE_EPS,   # M5: 1108 → 1327
        1.84014e-3,                             # between M5 and M4
        1.87160e-3, 1.87160e-3 + _EDGE_EPS,   # M4: 2901 → 3170
        2.00000e-3,
        2.28100e-3, 2.28100e-3 + _EDGE_EPS,   # M3: 2828 → 3279
        2.42350e-3,
        2.57490e-3, 2.57490e-3 + _EDGE_EPS,   # M2: 2445 → 2599
        2.69447e-3,
        2.81960e-3, 2.81960e-3 + _EDGE_EPS,   # M1: 2104 → 2194
        3.00000e-3, 4.00000e-3, 5.00000e-3,
        6.00000e-3, 8.00000e-3, 1.00000e-2,
    ]),
    'mu_rho': np.array([
        3.683e+3, 1.643e+3,
        1.108e+3, 1.327e+3,                    # M5 edge
        1.911e+3,                               # between M5 and M4
        2.901e+3, 3.170e+3,                    # M4 edge
        3.922e+3,
        2.828e+3, 3.279e+3,                    # M3 edge
        2.833e+3,
        2.445e+3, 2.599e+3,                    # M2 edge
        2.339e+3,
        2.104e+3, 2.194e+3,                    # M1 edge
        1.902e+3, 9.564e+2, 5.534e+2,
        3.514e+2, 1.705e+2, 9.691e+1,
    ]),
}

# Gold (Z=79), density 19.30 g/cm³
# M-edges (keV): M5=2.2057, M4=2.2911, M3=2.7430, M2=3.1478, M1=3.4249
_AU = {
    'energy_MeV': np.array([
        1.00000e-3, 1.50000e-3, 2.00000e-3,
        2.20570e-3, 2.20570e-3 + _EDGE_EPS,   # M5: 918.7 → 997.1
        2.24799e-3,                             # between M5 and M4
        2.29110e-3, 2.29110e-3 + _EDGE_EPS,   # M4: 2258 → 2389
        2.50689e-3,                             # between M4 and M3
        2.74300e-3, 2.74300e-3 + _EDGE_EPS,   # M3: 2203 → 2541
        3.00000e-3,
        3.14780e-3, 3.14780e-3 + _EDGE_EPS,   # M2: 1822 → 1933
        3.28343e-3,
        3.42490e-3, 3.42490e-3 + _EDGE_EPS,   # M1: 1585 → 1652
        4.00000e-3, 5.00000e-3, 6.00000e-3,
        8.00000e-3, 1.00000e-2,
    ]),
    'mu_rho': np.array([
        4.652e+3, 2.089e+3, 1.137e+3,
        9.187e+2, 9.971e+2,                    # M5 edge
        1.386e+3,                               # between M5 and M4
        2.258e+3, 2.389e+3,                    # M4 edge
        2.380e+3,                               # between M4 and M3
        2.203e+3, 2.541e+3,                    # M3 edge
        2.049e+3,
        1.822e+3, 1.933e+3,                    # M2 edge
        1.748e+3,
        1.585e+3, 1.652e+3,                    # M1 edge
        1.144e+3, 6.661e+2, 4.253e+2,
        2.072e+2, 1.181e+2,
    ]),
}

# ---------------------------------------------------------------------------
# Compound tables (mixture rule: (μ/ρ)_cmpd = Σ wᵢ (μ/ρ)ᵢ)
# ---------------------------------------------------------------------------

def _log_interp(e_arr, mu_arr):
    """Build a log-log linear interpolator from sorted (energy, mu/rho) arrays.

    The arrays must already be strictly increasing in energy (edge pairs should
    have been split with ``_EDGE_EPS`` before calling this function).
    """
    log_e = np.log10(np.asarray(e_arr, dtype=float))
    log_mu = np.log10(np.asarray(mu_arr, dtype=float))
    return interp1d(log_e, log_mu, kind='linear',
                    bounds_error=False, fill_value='extrapolate')


def _eval_interp(fn, energy_MeV):
    return 10.0 ** float(fn(np.log10(energy_MeV)))


def _make_compound(constituents):
    """Build compound (μ/ρ, energy) via the mixture rule.

    Parameters
    ----------
    constituents : list of (weight_fraction, element_dict)
        Each element_dict has keys 'energy_MeV' and 'mu_rho' with
        strictly-increasing energy arrays (edge pairs already split).

    Returns
    -------
    dict with 'energy_MeV' (sorted, unique) and 'mu_rho' arrays.
    """
    # Union of all element energy grids — use unique to avoid duplicates
    # from elements sharing the same nominal grid point (e.g. 1.5e-3 MeV).
    # Edge discontinuities are already distinct because of _EDGE_EPS.
    all_e = np.unique(np.concatenate([c['energy_MeV'] for _, c in constituents]))
    mu_compound = np.zeros(len(all_e))

    for w_i, elem in constituents:
        f = _log_interp(elem['energy_MeV'], elem['mu_rho'])
        mu_compound += w_i * np.array([_eval_interp(f, e) for e in all_e])

    return {'energy_MeV': all_e, 'mu_rho': mu_compound}


# Si₃N₄ (density 3.44 g/cm³)
#   MW = 3×28.085 + 4×14.007 = 140.28
#   w(Si) = 3×28.085/140.28 = 0.6007,  w(N) = 4×14.007/140.28 = 0.3993
_SI3N4 = _make_compound([
    (0.6007, _SI),
    (0.3993, _N),
])

# SiC (density 3.21 g/cm³)
#   MW = 28.085 + 12.011 = 40.096
#   w(Si) = 28.085/40.096 = 0.7005,  w(C) = 12.011/40.096 = 0.2995
_SIC = _make_compound([
    (0.7005, _SI),
    (0.2995, _C),
])

# PMMA (C₅H₈O₂, density 1.18 g/cm³)
#   MW = 5×12.011 + 8×1.008 + 2×15.999 = 100.12
#   w(C) = 60.055/100.12 = 0.5998,  w(H) = 8.064/100.12 = 0.0805,  w(O) = 31.998/100.12 = 0.3197
_PMMA_COMPOUND = _make_compound([
    (0.5998, _C),
    (0.0805, _H),
    (0.3197, _O),
])

# ---------------------------------------------------------------------------
# Registry and cached interpolators
# ---------------------------------------------------------------------------

_TABLES: dict[str, dict] = {
    # Full names (used by MaterialProperties.name)
    'Tantalum':        _TA,
    'Tungsten':        _W,
    'Gold':            _AU,
    'Silicon':         _SI,
    'Nitrogen':        _N,
    'Carbon':          _C,
    'Oxygen':          _O,
    'Hydrogen':        _H,
    'Silicon Nitride': _SI3N4,
    'Silicon Carbide': _SIC,
    'PMMA':            _PMMA_COMPOUND,
    # Short aliases
    'Ta':    _TA,
    'W':     _W,
    'Au':    _AU,
    'Si':    _SI,
    'N':     _N,
    'C':     _C,
    'O':     _O,
    'H':     _H,
    'Si3N4': _SI3N4,
    'SiC':   _SIC,
}

_INTERPOLATORS: dict[str, interp1d] = {}


def _get_interpolator(key: str) -> interp1d:
    if key not in _INTERPOLATORS:
        tbl = _TABLES[key]
        _INTERPOLATORS[key] = _log_interp(tbl['energy_MeV'], tbl['mu_rho'])
    return _INTERPOLATORS[key]


def get_mu_rho(material_key: str, energy_kev: float) -> float:
    """Return μ/ρ (cm²/g) at *energy_kev* using NIST XCOM log-log interpolation.

    Parameters
    ----------
    material_key : str
        Material name or alias — see ``_TABLES`` for valid keys.
    energy_kev : float
        Photon energy in keV.

    Returns
    -------
    float
        Mass attenuation coefficient μ/ρ in cm²/g.

    Notes
    -----
    Energies outside the tabulated range are extrapolated using the
    log-log slope at the nearest table boundary.
    """
    if material_key not in _TABLES:
        raise KeyError(
            f"No NIST XCOM data for '{material_key}'. "
            f"Available: {sorted(set(_TABLES.keys()))}"
        )
    return _eval_interp(_get_interpolator(material_key), energy_kev * 1e-3)
