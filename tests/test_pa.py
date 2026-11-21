"""The test PA (dpd/testing/pa.py) against values computed by hand from matlab/link/rf_amp.m's formulas
(scalar arithmetic, independent of the vectorized port; the numbers are in README.md, step 1)."""
import numpy as np
import pytest

from dpd.testing.pa import GAN_3GPP, pa_chain, pa_params, rf_amp

AMAM = pa_params(ampm={"enable": False})                    # Rapp only
OFF = pa_params(enable=False, ampm={"enable": False})        # everything off


def test_parameters_are_the_nominal_links():
    assert GAN_3GPP["asat"] == 1.5 and GAN_3GPP["smooth"] == 1.94
    assert GAN_3GPP["ampm"] == {"enable": True, "alpha": 0.51, "beta": 1.93}
    assert GAN_3GPP["memory"] == {"enable": False, "delay_s": 0.6e-9, "linear": [-0.08, 0.03], "cubic": [-0.05, 0.02]}


def test_rapp_at_saturation():
    # |A| = asat: |A| / 2^(1/(2s))
    assert rf_amp(np.array([1.5]), AMAM)[0] == pytest.approx(1.2546026733935693, rel=1e-14)


def test_agc():
    # rop 2 dB below the reference: A / 10^(-0.2) before the PA
    assert rf_amp(np.array([1.0]), AMAM, rop_dbm=-12, rop_ref_dbm=-10)[0] == pytest.approx(1.287725872296698, rel=1e-14)


def test_amam_and_ampm():
    # 0.6 e^{j0.3}: |out| = 0.59566, phase + alpha r2/(1 + beta r2) = +0.062347 rad
    z = rf_amp(np.array([0.6 * np.exp(0.3j)]))[0]
    assert abs(z) == pytest.approx(0.5956588580691601, rel=1e-14)
    assert np.angle(z) - 0.3 == pytest.approx(0.062347188264058676, rel=1e-12)
    assert z == pytest.approx(0.556981175191062 + 0.21114318762166892j, rel=1e-14)


def test_memory():
    # fs 3.2 GS/s: D = round(0.6 ns * 3.2 GS/s) = round(1.92) = 2 samples; Rapp and AM/PM off
    p = pa_params(enable=False, ampm={"enable": False}, memory={"enable": True})
    u = np.array([0.5, 0, 0, 0, 0.8j])
    y = rf_amp(u, p, fs=3.2e9)
    assert y[2] == pytest.approx(-0.04625, rel=1e-14)                 # -0.08 u0 - 0.05 u0^3
    assert y[4] == pytest.approx(0.0175 + 0.8j, rel=1e-14)            # u4 + 0.03 u0 + 0.02 u0^3
    assert y[0] == 0.5 and y[1] == 0 and y[3] == 0


def test_memory_delay_rounds_like_matlab():
    # 2.5 samples: MATLAB's round gives 3 (numpy's would give 2)
    p = pa_params(enable=False, ampm={"enable": False}, memory={"enable": True, "linear": [1.0, 0.0], "cubic": [0.0, 0.0]})
    y = rf_amp(np.array([1.0, 0, 0, 0, 0, 0, 0]), p, fs=2.5 / 0.6e-9)
    assert y[3] == 1.0 and y[2] == 0


def test_zero_input_and_off():
    assert rf_amp(np.zeros(3))[0] == 0
    x = np.array([0.3 + 0.1j, 2.0])
    assert np.array_equal(rf_amp(x, OFF), x)


def test_chain_levels():
    # the tone at 1.104 a0 is compressed but stays; the chain removes it as the mean
    y = pa_chain(0.371 * np.exp(1j * np.linspace(0, 20, 1000)))
    assert abs(y.mean()) < 1e-12
