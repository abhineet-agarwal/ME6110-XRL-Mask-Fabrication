"""
Visualization Module for X-ray Lithography Simulation
======================================================

All plotting functions live here, separated from physics code.
Every function returns ``(fig, ax)`` so plots are composable and
easily embeddable in GUIs (tkinter, PyQt, Streamlit, etc.).

Author: Abhineet Agarwal
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from typing import Tuple, Optional


# -------------------------------------------------------------------
# Shared styling constants
# -------------------------------------------------------------------
_FIGSIZE_SINGLE = (8, 5)
_FIGSIZE_WIDE = (10, 6)
_LABEL_SIZE = 12
_TITLE_SIZE = 13
_COLORS = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b']


def _style_ax(ax: Axes, xlabel: str, ylabel: str, title: str) -> None:
    """Apply consistent styling to an axis."""
    ax.set_xlabel(xlabel, fontsize=_LABEL_SIZE)
    ax.set_ylabel(ylabel, fontsize=_LABEL_SIZE)
    ax.set_title(title, fontsize=_TITLE_SIZE, fontweight='bold')
    ax.grid(True, alpha=0.3)


# -------------------------------------------------------------------
# Aerial image plots
# -------------------------------------------------------------------

def plot_aerial_image(
    x: np.ndarray,
    intensity: np.ndarray,
    title: str = 'Aerial Image',
    normalize: bool = True,
    fig: Optional[Figure] = None,
    ax: Optional[Axes] = None,
) -> Tuple[Figure, Axes]:
    """Plot aerial-image intensity vs position.

    Args:
        x: Position array in um.
        intensity: Intensity array.
        title: Plot title.
        normalize: If ``True``, normalise intensity to [0, 1].
        fig, ax: Optional existing figure/axes to draw into.

    Returns:
        ``(fig, ax)`` tuple.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=_FIGSIZE_WIDE)
    y = intensity / intensity.max() if normalize else intensity
    ax.plot(x, y, color=_COLORS[0], linewidth=2)
    ax.axhline(y=0.5, color=_COLORS[1], linestyle='--', alpha=0.5, label='50% threshold')
    _style_ax(ax, 'Position (um)', 'Normalised Intensity' if normalize else 'Intensity', title)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_contrast_vs_energy(
    energies: np.ndarray,
    contrasts: np.ndarray,
    fig: Optional[Figure] = None,
    ax: Optional[Axes] = None,
) -> Tuple[Figure, Axes]:
    """Contrast as a function of photon energy.

    Args:
        energies: Photon energies in keV.
        contrasts: Michelson contrast values (0--1).

    Returns:
        ``(fig, ax)`` tuple.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=_FIGSIZE_SINGLE)
    ax.plot(energies, contrasts, 'o-', color=_COLORS[0], linewidth=2, markersize=5)
    _style_ax(ax, 'Photon Energy (keV)', 'Contrast', 'Image Contrast vs Photon Energy')
    fig.tight_layout()
    return fig, ax


def plot_contrast_vs_gap(
    gaps: np.ndarray,
    contrasts: np.ndarray,
    fig: Optional[Figure] = None,
    ax: Optional[Axes] = None,
) -> Tuple[Figure, Axes]:
    """Contrast as a function of mask-resist gap.

    Args:
        gaps: Gaps in um.
        contrasts: Michelson contrast values.

    Returns:
        ``(fig, ax)`` tuple.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=_FIGSIZE_SINGLE)
    ax.plot(gaps, contrasts, 's-', color=_COLORS[2], linewidth=2, markersize=5)
    _style_ax(ax, 'Gap (um)', 'Contrast', 'Image Contrast vs Mask-Resist Gap')
    fig.tight_layout()
    return fig, ax


# -------------------------------------------------------------------
# Resist plots
# -------------------------------------------------------------------

def plot_resist_profile(
    x: np.ndarray,
    profile: np.ndarray,
    resist_name: str = '',
    fig: Optional[Figure] = None,
    ax: Optional[Axes] = None,
) -> Tuple[Figure, Axes]:
    """Developed resist profile (remaining thickness fraction).

    Args:
        x: Position array in um.
        profile: Remaining thickness (0--1).
        resist_name: Label for the legend.

    Returns:
        ``(fig, ax)`` tuple.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=_FIGSIZE_WIDE)
    label = resist_name or None
    ax.fill_between(x, 0, profile, alpha=0.4, color=_COLORS[0], label=label)
    ax.plot(x, profile, color=_COLORS[0], linewidth=1.5)
    _style_ax(ax, 'Position (um)', 'Remaining Thickness', f'Developed Profile{" - " + resist_name if resist_name else ""}')
    if label:
        ax.legend()
    ax.set_ylim(-0.05, 1.1)
    fig.tight_layout()
    return fig, ax


def plot_cd_vs_dose(
    doses: np.ndarray,
    cds: np.ndarray,
    resist_name: str = '',
    fig: Optional[Figure] = None,
    ax: Optional[Axes] = None,
) -> Tuple[Figure, Axes]:
    """Critical dimension vs dose factor.

    Args:
        doses: Dose factor array (multiples of D0).
        cds: CD values in um.
        resist_name: Optional label.

    Returns:
        ``(fig, ax)`` tuple.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=_FIGSIZE_SINGLE)
    label = resist_name or None
    ax.plot(doses, cds, 'o-', color=_COLORS[0], linewidth=2, markersize=5, label=label)
    _style_ax(ax, 'Dose Factor (x D0)', 'CD (um)', 'Critical Dimension vs Dose')
    if label:
        ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_ler_vs_dose(
    doses: np.ndarray,
    lers: np.ndarray,
    resist_name: str = '',
    fig: Optional[Figure] = None,
    ax: Optional[Axes] = None,
) -> Tuple[Figure, Axes]:
    """Line-edge roughness vs dose factor.

    Args:
        doses: Dose factor array (multiples of D0).
        lers: LER (3-sigma) in nm.
        resist_name: Optional label.

    Returns:
        ``(fig, ax)`` tuple.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=_FIGSIZE_SINGLE)
    label = resist_name or None
    ax.plot(doses, lers, 's-', color=_COLORS[1], linewidth=2, markersize=5, label=label)
    _style_ax(ax, 'Dose Factor (x D0)', 'LER 3-sigma (nm)', 'Line-Edge Roughness vs Dose')
    if label:
        ax.legend()
    fig.tight_layout()
    return fig, ax


# -------------------------------------------------------------------
# Thermal plots
# -------------------------------------------------------------------

def plot_thermal_deflection(
    powers: np.ndarray,
    deflections: np.ndarray | dict,
    materials: list[str] | None = None,
    fig: Optional[Figure] = None,
    ax: Optional[Axes] = None,
) -> Tuple[Figure, Axes]:
    """Membrane deflection vs beam power.

    ``deflections`` may be a 1-D array (single material) or a dict
    mapping material names to arrays (comparison plot).

    Args:
        powers: Beam powers in W.
        deflections: Deflection data (see above).
        materials: Material labels (only for 1-D array input).

    Returns:
        ``(fig, ax)`` tuple.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=_FIGSIZE_SINGLE)

    if isinstance(deflections, dict):
        for i, (name, defl) in enumerate(deflections.items()):
            ax.loglog(powers, defl, '-', color=_COLORS[i % len(_COLORS)],
                      linewidth=2, label=name)
        ax.legend()
    else:
        label = materials[0] if materials else None
        ax.loglog(powers, deflections, '-', color=_COLORS[2], linewidth=2, label=label)
        if label:
            ax.legend()

    _style_ax(ax, 'Beam Power (W)', 'Deflection (um)', 'Membrane Deflection vs Beam Power')
    fig.tight_layout()
    return fig, ax


def plot_thermal_stress(
    powers: np.ndarray,
    stresses: np.ndarray | dict,
    materials: list[str] | None = None,
    fig: Optional[Figure] = None,
    ax: Optional[Axes] = None,
) -> Tuple[Figure, Axes]:
    """Thermal stress vs beam power.

    ``stresses`` may be a 1-D array or a dict of arrays.

    Args:
        powers: Beam powers in W.
        stresses: Stress data in MPa.
        materials: Material labels (only for 1-D array input).

    Returns:
        ``(fig, ax)`` tuple.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=_FIGSIZE_SINGLE)

    if isinstance(stresses, dict):
        for i, (name, s) in enumerate(stresses.items()):
            ax.loglog(powers, s, '-', color=_COLORS[i % len(_COLORS)],
                      linewidth=2, label=name)
        ax.legend()
    else:
        label = materials[0] if materials else None
        ax.loglog(powers, stresses, '-', color=_COLORS[1], linewidth=2, label=label)
        if label:
            ax.legend()

    _style_ax(ax, 'Beam Power (W)', 'Thermal Stress (MPa)', 'Thermal Stress vs Beam Power')
    fig.tight_layout()
    return fig, ax


def plot_parameter_heatmap(
    gaps: np.ndarray,
    energies: np.ndarray,
    contrasts: np.ndarray,
    fig: Optional[Figure] = None,
    ax: Optional[Axes] = None,
) -> Tuple[Figure, Axes]:
    """2-D heatmap of contrast over (gap, energy) parameter space.

    Args:
        gaps: 1-D array of gaps in um (columns).
        energies: 1-D array of energies in keV (rows).
        contrasts: 2-D array of shape ``(len(energies), len(gaps))``.

    Returns:
        ``(fig, ax)`` tuple.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=_FIGSIZE_SINGLE)

    im = ax.imshow(
        contrasts,
        extent=[gaps[0], gaps[-1], energies[0], energies[-1]],
        origin='lower',
        aspect='auto',
        cmap='viridis',
    )
    fig.colorbar(im, ax=ax, label='Contrast')
    _style_ax(ax, 'Gap (um)', 'Energy (keV)', 'Contrast: Energy vs Gap')
    fig.tight_layout()
    return fig, ax


def plot_material_comparison_bar(
    material_names: list[str],
    values: list[float],
    ylabel: str = 'Value',
    title: str = 'Material Comparison',
    fig: Optional[Figure] = None,
    ax: Optional[Axes] = None,
) -> Tuple[Figure, Axes]:
    """Bar chart comparing a quantity across materials.

    Args:
        material_names: List of material labels.
        values: Corresponding numeric values.
        ylabel: Y-axis label.
        title: Plot title.

    Returns:
        ``(fig, ax)`` tuple.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=_FIGSIZE_SINGLE)

    colors = [_COLORS[i % len(_COLORS)] for i in range(len(material_names))]
    ax.bar(material_names, values, color=colors, alpha=0.7)
    _style_ax(ax, '', ylabel, title)
    fig.tight_layout()
    return fig, ax
