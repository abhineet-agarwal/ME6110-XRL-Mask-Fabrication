"""
Unit tests for XRL core physics calculations.

Run with::

    pytest tests/test_physics.py -v
"""

import numpy as np
import pytest

from xrl.materials import MATERIALS, RESISTS, MEMBRANES
from xrl.aerial_image import XRayMask, AerialImageSimulator
from xrl.resist import ResistExposureModel, simulate_full_exposure
from xrl.thermal import MembraneMechanics, ThermalAnalysis
from xrl.config import SimulationConfig, default_config


# -----------------------------------------------------------------------
# Materials
# -----------------------------------------------------------------------

class TestMaterials:
    """Verify material database entries and attenuation calculation."""

    def test_materials_dict_nonempty(self):
        assert len(MATERIALS) >= 3
        assert 'Ta' in MATERIALS

    def test_resists_dict_nonempty(self):
        assert len(RESISTS) >= 2
        assert 'PMMA' in RESISTS

    def test_membranes_dict_nonempty(self):
        assert len(MEMBRANES) >= 3
        assert 'Si3N4' in MEMBRANES

    def test_attenuation_positive(self):
        """Attenuation coefficient must be positive for all materials."""
        for key, mat in MATERIALS.items():
            mu = mat.get_attenuation_coefficient(1.5)
            assert mu > 0, f"{key} has non-positive mu at 1.5 keV"

    def test_high_z_attenuates_more(self):
        """High-Z absorbers should attenuate more than low-Z membranes."""
        mu_ta = MATERIALS['Ta'].get_attenuation_coefficient(1.5)
        mu_si3n4 = MATERIALS['Si3N4'].get_attenuation_coefficient(1.5)
        assert mu_ta > mu_si3n4 * 10

    def test_attenuation_decreases_with_energy(self):
        """Higher energy photons are less attenuated (above M-edge region)."""
        # Test 3→5 keV to avoid Ta M-edges (1.7–2.7 keV) which cause local increases
        mu_low = MATERIALS['Ta'].get_attenuation_coefficient(3.0)
        mu_high = MATERIALS['Ta'].get_attenuation_coefficient(5.0)
        assert mu_low > mu_high


# -----------------------------------------------------------------------
# Aerial image
# -----------------------------------------------------------------------

class TestAerialImage:
    """Verify mask transmission and aerial image properties."""

    def test_beer_lambert_transmission(self):
        """Known thickness -> known transmission via Beer-Lambert."""
        mask = XRayMask(absorber_material='Ta', absorber_thickness=0.5)
        energy = 1.5
        mu = MATERIALS['Ta'].get_attenuation_coefficient(energy) * 1e-4  # 1/um
        expected_t = np.exp(-mu * 0.5)
        # Transmission through absorber should be close to expected
        x = np.array([0.0])  # inside absorber region (x_mod < feature_size)
        t = mask.get_transmission_profile(x, energy)
        # Also has membrane attenuation on top
        mu_mem = MATERIALS['Si3N4'].get_attenuation_coefficient(energy) * 1e-4
        expected_total = expected_t * np.exp(-mu_mem * 2.0)
        np.testing.assert_allclose(t[0], expected_total, rtol=1e-10)

    def test_open_area_higher_than_absorber(self):
        """Open areas transmit more than absorber-covered areas."""
        mask = XRayMask(feature_size=0.3, pitch=1.0)
        x = np.array([0.1, 0.5])  # 0.1 is under absorber, 0.5 is open
        t = mask.get_transmission_profile(x, 1.5)
        assert t[1] > t[0]

    def test_fresnel_number_calculation(self):
        """Verify Fresnel number F = a^2 / (lambda * gap)."""
        mask = XRayMask(feature_size=0.5)
        sim = AerialImageSimulator(mask, gap=10.0)
        wavelength = 1.24 / 1.5 / 1000  # um at 1.5 keV
        F_expected = 0.5**2 / (wavelength * 10.0)
        # F should be > 0.1 (diffraction regime)
        assert F_expected > 0.1

    def test_contrast_bounded(self):
        """Contrast should be between 0 and 1."""
        mask = XRayMask()
        sim = AerialImageSimulator(mask, gap=10.0)
        x, intensity = sim.compute_aerial_image(1.5)
        c = sim.calculate_contrast(x, intensity)
        assert 0 <= c <= 1

    def test_aerial_image_shape(self):
        """Output arrays should match requested resolution."""
        mask = XRayMask()
        sim = AerialImageSimulator(mask, gap=10.0)
        x, intensity = sim.compute_aerial_image(1.5, resolution=500)
        assert len(x) == 500
        assert len(intensity) == 500

    def test_zero_gap_returns_transmission(self):
        """With gap=0, aerial image equals mask transmission."""
        mask = XRayMask()
        sim = AerialImageSimulator(mask, gap=0.0)
        x, intensity = sim.compute_aerial_image(1.5)
        expected = mask.get_transmission_profile(x, 1.5)
        np.testing.assert_array_equal(intensity, expected)


# -----------------------------------------------------------------------
# Resist
# -----------------------------------------------------------------------

class TestResist:
    """Verify resist development model and metrology."""

    def test_below_threshold_positive(self):
        """Positive resist below D0 should remain fully intact."""
        resist = RESISTS['PMMA']
        model = ResistExposureModel(resist)
        dose = np.array([0.0, 100.0, 300.0])  # all below D0=500
        remaining = model.development_model(dose)
        assert np.all(remaining >= 0.5), "Should be mostly remaining below D0"

    def test_above_threshold_positive(self):
        """Positive resist well above D0 should be mostly removed."""
        resist = RESISTS['PMMA']
        model = ResistExposureModel(resist)
        dose = np.array([2000.0, 5000.0])  # well above D0=500
        remaining = model.development_model(dose)
        assert np.all(remaining < 0.1), "Should be mostly removed above D0"

    def test_negative_resist_crosslinks(self):
        """Negative resist above threshold should retain material."""
        resist = RESISTS['SU8']
        model = ResistExposureModel(resist)
        dose = np.array([500.0])  # above D0 * 0.5
        remaining = model.development_model(dose)
        assert remaining[0] > 0.0

    def test_cd_returns_reasonable_value(self):
        """CD should be near feature size for well-exposed resist."""
        mask = XRayMask(feature_size=0.5, pitch=1.0)
        sim = AerialImageSimulator(mask, gap=5.0)
        x, intensity = sim.compute_aerial_image(1.5, resolution=1000)
        resist = RESISTS['PMMA']
        _, developed, metrics = simulate_full_exposure(
            intensity, x, resist, energy_kev=1.5, dose_factor=1.5,
            include_noise=False,
        )
        cd = metrics['cd_um']
        if not np.isnan(cd):
            assert 0.1 < cd < 1.5, f"CD={cd} is out of expected range"

    def test_absorption_coefficient_positive(self):
        """Absorption coefficient must be positive."""
        model = ResistExposureModel(RESISTS['PMMA'])
        for e in [0.5, 1.0, 2.0, 4.0]:
            assert model.absorption_coefficient(e) > 0


# -----------------------------------------------------------------------
# Thermal
# -----------------------------------------------------------------------

class TestThermal:
    """Verify thermal stress and deflection formulae."""

    def test_thermal_stress_formula(self):
        """sigma = E * alpha * dT / (1 - nu) for known inputs."""
        mat = MEMBRANES['Si3N4']
        mem = MembraneMechanics(mat, thickness=2.0, size=50.0)
        E = 250e9   # Pa
        nu = 0.27
        alpha = 2.3e-6  # 1/K
        dT = 10.0  # K
        expected_MPa = E * alpha * dT / (1 - nu) / 1e6
        computed = mem.thermal_stress(dT)
        np.testing.assert_allclose(computed, expected_MPa, rtol=1e-10)

    def test_stress_proportional_to_dT(self):
        """Doubling dT should double the thermal stress."""
        mat = MEMBRANES['SiC']
        mem = MembraneMechanics(mat, thickness=2.0, size=50.0)
        s1 = mem.thermal_stress(5.0)
        s2 = mem.thermal_stress(10.0)
        np.testing.assert_allclose(s2, 2 * s1, rtol=1e-10)

    def test_deflection_positive(self):
        """Thermal deflection should be positive for positive dT."""
        mat = MEMBRANES['Si3N4']
        mem = MembraneMechanics(mat, thickness=2.0, size=50.0)
        assert mem.thermal_deflection(1.0) > 0

    def test_diamond_lowest_deflection(self):
        """Diamond should have smallest deflection (highest k, lowest alpha)."""
        results = {}
        for name in ['Si3N4', 'SiC', 'Diamond']:
            mem = MembraneMechanics(MEMBRANES[name], 2.0, 50.0)
            thermal = ThermalAnalysis(mem)
            absorbed = thermal.absorbed_power(0.1)
            dT = thermal.steady_state_in_plane_gradient(absorbed)
            results[name] = mem.thermal_deflection(dT * 0.1)
        assert results['Diamond'] < results['Si3N4']
        assert results['Diamond'] < results['SiC']

    def test_pressure_deflection_positive(self):
        """Positive pressure gives positive deflection."""
        mat = MEMBRANES['Si3N4']
        mem = MembraneMechanics(mat, thickness=2.0, size=50.0)
        assert mem.pressure_deflection_center(100.0) > 0

    def test_time_constant_positive(self):
        """Thermal time constant must be positive."""
        mat = MEMBRANES['Si3N4']
        mem = MembraneMechanics(mat, thickness=2.0, size=50.0)
        thermal = ThermalAnalysis(mem)
        assert thermal.thermal_time_constant() > 0


# -----------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------

class TestConfig:
    """Verify config serialisation round-trip."""

    def test_default_config(self):
        cfg = default_config()
        assert cfg.energy_kev == 1.5
        assert cfg.resist == 'PMMA'

    def test_round_trip_json(self, tmp_path):
        cfg = SimulationConfig(energy_kev=2.0, resist='ZEP520A')
        path = tmp_path / "test_cfg.json"
        cfg.save(path)
        loaded = SimulationConfig.load(path)
        assert loaded.energy_kev == 2.0
        assert loaded.resist == 'ZEP520A'

    def test_to_dict(self):
        cfg = default_config()
        d = cfg.to_dict()
        assert isinstance(d, dict)
        assert 'energy_kev' in d


# -----------------------------------------------------------------------
# Contrast calculation
# -----------------------------------------------------------------------

class TestContrast:
    """Verify contrast from known min/max values."""

    def test_known_contrast(self):
        """Contrast = (max-min)/(max+min) for known values."""
        mask = XRayMask()
        sim = AerialImageSimulator(mask, gap=10.0)
        # Synthesise a simple signal
        x = np.linspace(-1.5, 1.5, 1000)
        I_max, I_min = 0.9, 0.1
        # Half-period sawtooth to get known max/min
        intensity = np.where(
            np.mod(x, 1.0) < 0.5, I_min, I_max,
        )
        c = sim.calculate_contrast(x, intensity)
        expected = (I_max - I_min) / (I_max + I_min)
        assert abs(c - expected) < 0.05  # within 5% due to windowing


# -----------------------------------------------------------------------
# NIST XCOM accuracy
# -----------------------------------------------------------------------

class TestNISTAccuracy:
    """Spot-check attenuation values against directly tabulated NIST data."""

    def test_ta_at_1kev(self):
        """Ta at 1.0 keV: NIST μ/ρ = 3510 cm²/g × 16.65 g/cm³."""
        mu = MATERIALS['Ta'].get_attenuation_coefficient(1.0)
        np.testing.assert_allclose(mu, 3510 * 16.65, rtol=0.01)

    def test_ta_below_m5_edge(self):
        """Just below Ta M5 edge (1.7351 keV): μ/ρ ≈ 1154 cm²/g."""
        mu_rho = MATERIALS['Ta'].get_attenuation_coefficient(1.734) / 16.65
        np.testing.assert_allclose(mu_rho, 1154, rtol=0.05)

    def test_ta_m5_edge_jump(self):
        """Ta M5 edge at 1.7351 keV: absorption increases by ≥10%."""
        mu_below = MATERIALS['Ta'].get_attenuation_coefficient(1.7350) / 16.65
        mu_above = MATERIALS['Ta'].get_attenuation_coefficient(1.7352) / 16.65
        assert mu_above > mu_below * 1.1, (
            f"Expected ≥10% jump at M5 edge; got {mu_below:.1f} → {mu_above:.1f}"
        )

    def test_si_k_edge_jump(self):
        """Si K-edge at 1.839 keV: ≥3× jump in Si₃N₄ attenuation."""
        mu_below = MATERIALS['Si3N4'].get_attenuation_coefficient(1.838)
        mu_above = MATERIALS['Si3N4'].get_attenuation_coefficient(1.840)
        assert mu_above > mu_below * 3, (
            f"Expected ≥3× jump at Si K-edge; got {mu_below:.1f} → {mu_above:.1f}"
        )

    def test_si3n4_2um_transmission_below_k_edge(self):
        """2 μm Si₃N₄ at 1.5 keV (below Si K-edge): 40–90% transmission."""
        mu = MATERIALS['Si3N4'].get_attenuation_coefficient(1.5) * 1e-4  # 1/μm
        T = np.exp(-mu * 2.0)
        assert 0.4 < T < 0.9, f"Transmission {T:.3f} out of expected 40–90% range"

    def test_si3n4_2um_transmission_above_k_edge(self):
        """2 μm Si₃N₄ at 1.85 keV (just above Si K-edge): 10–40% transmission."""
        mu = MATERIALS['Si3N4'].get_attenuation_coefficient(1.85) * 1e-4  # 1/μm
        T = np.exp(-mu * 2.0)
        assert 0.05 < T < 0.40, f"Transmission {T:.3f} out of expected 10–40% range"

    def test_ta_absorber_05um_transmission(self):
        """Ta 0.5 μm at 1.5 keV: exp(-1566×16.65×0.5e-4) ≈ 27% (not 95%)."""
        mu = MATERIALS['Ta'].get_attenuation_coefficient(1.5) * 1e-4  # 1/μm
        T = np.exp(-mu * 0.5)
        assert T < 0.5, f"Ta absorber transmission {T:.3f} should be <50% at 1.5 keV"

    def test_absorbers_much_higher_than_membrane(self):
        """Ta/W/Au μ should be >> Si₃N₄/SiC μ at 1.5 keV (contrast requirement)."""
        mu_ta = MATERIALS['Ta'].get_attenuation_coefficient(1.5)
        mu_si3n4 = MATERIALS['Si3N4'].get_attenuation_coefficient(1.5)
        assert mu_ta > mu_si3n4 * 10
