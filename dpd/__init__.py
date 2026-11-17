"""Digital predistortion of the A-RoF link's baseband drive (PyTorch; numpy at the boundary).

models  the DPD networks and the numpy wrapper DPD (predistort, fit, save/load)
ila     indirect learning of the post-inverse
pa      a port of the link's PA (matlab/link/rf_amp.m), a test fixture
signals 16QAM/RRC test signals; metrics: NMSE and EVM
"""
