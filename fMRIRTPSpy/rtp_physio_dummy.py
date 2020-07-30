#!/usr/bin/env ipython3
# -*- coding: utf-8 -*-
"""
@author: mmisaki@laureateinstitute.org
"""


# %% import ===================================================================
from pathlib import Path
import numpy as np

try:
    from .rtp_common import RTP
except Exception:
    from rtp_common import RTP


# %% ==========================================================================
class RTP_PHYSIO_DUMMY(RTP):
    """Dummy class of RTP_PHYSIO

    """

    def __init__(self, ecg_f, resp_f, sample_freq, rtp_retrots, verb=True):
        """
        Options
        -------
        ecg_f: Path object or string
            ecg signal file
        resp_f: Path object or string
        sample_freq: float
            Frequency of signal in the files (Hz)
        rtp_retrots: RTP_RETROTS object
            instance of RTP_RETROTS for making RetroTS reggressor
        verb: bool
            verbose flag to print log message
        """

        super().__init__()  # call __init__() in RTP class

        # --- Set parameters ---
        self.effective_sample_freq = sample_freq
        self.rtp_retrots = rtp_retrots
        self._verb = verb

        # --- Load data ---
        assert Path(ecg_f).is_file()
        assert Path(resp_f).is_file()
        self.ecg_data = np.loadtxt(ecg_f)
        self.resp_data = np.loadtxt(resp_f)

        self.not_available = False

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def get_retrots(self, TR, Nvol=np.inf, tshift=0):
        if self.rtp_retrots is None:
            return None

        tlen_current = len(self.resp_data)
        max_vol = int((tlen_current/self.effective_sample_freq)//TR)
        if np.isinf(Nvol):
            Nvol = max_vol

        tlen_need = int(Nvol * TR * self.effective_sample_freq)
        while len(self.resp_data) < tlen_need:
            self.errmsg(f"Physio data is availabel up to {Nvol*TR} s")
            return

        Resp = self.resp_data[:tlen_need]
        ECG = self.ecg_data[:tlen_need]

        PhysFS = self.effective_sample_freq
        retroTSReg = self.rtp_retrots.do_proc(Resp, ECG, TR, PhysFS, tshift)

        return retroTSReg[:Nvol, :]
