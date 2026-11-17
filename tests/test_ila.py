"""ILA on the test PA: a short run, the DPD must improve the PA output's NMSE and EVM.
The full check with its report is scripts/step1_check.py."""
import numpy as np

from dpd import metrics, pa, signals
from dpd.ila import ila
from dpd.models import DPD, TransformerDPD


def test_ila_improves_the_pa_output():
    s, _ = signals.baseband(1500, seed=1)
    x = s * pa.DATA_RMS_A0 / np.sqrt(np.mean(np.abs(s) ** 2))
    t, d = signals.baseband(800, seed=2)
    xt = t * pa.DATA_RMS_A0 / np.sqrt(np.mean(np.abs(t) ** 2))
    dpd = DPD(TransformerDPD(), "cpu")
    G, _ = ila(dpd, pa.pa_chain, x, passes=2, epochs=6, batch=256, lr=3e-3)
    y0, y1 = pa.pa_chain(xt), pa.pa_chain(dpd.predistort(xt))
    assert metrics.nmse_db(y1, xt) < metrics.nmse_db(y0, xt) - 6
    assert metrics.evm_symbols(y1, d) < 0.6 * metrics.evm_symbols(y0, d)
