#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 10:13:37 2018

@author: mmisaki@laureateinstitute.org
"""

# %% ==========================================================================
import os
import re
import numpy as np
import ctypes

try:
    librtp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'librtp.so')
except Exception:
    librtp_path = './librtp.so'


# %% ==========================================================================
class RetroTSOpt(ctypes.Structure):
    """
    RetroTSOpt struct class defined in rtp_restrots.h

    def struct {
        float VolTR; // TR of MRI acquisition in second
        float PhysFS; // Sampling frequency of physiological signal data
        float tshift;  // slice timing offset. 0.0 is the first slice
        float zerophaseoffset;
        int RVTshifts[256];
        int RVTshiftslength;
        int RespCutoffFreq;
        int fcutoff;
        int AmpPhase;
        int CardCutoffFreq;
        char ResamKernel[256];
        int FIROrder;
        int as_windwidth;
        int as_percover;
        int as_fftwin;
        } RetroTSOpt;

    VolTR, PhysFS, and tshift (=0 in default) sholud be be set manually. Other
    fields are initialized and used inside the rtp_retrots funtion.
    """

    _fields_ = [
            ('VolTR', ctypes.c_float),  # Volume TR in seconds
            ('PhysFS', ctypes.c_float),  # Sampling frequency (Hz)
            ('tshift', ctypes.c_float),  # slice timing offset
            ('zerophaseoffset', ctypes.c_float),
            ('RVTshifts', ctypes.c_int*256),
            ('RVTshiftslength', ctypes.c_int),
            ('RespCutoffFreq', ctypes.c_int),
            ('fcutoff', ctypes.c_int),  # cut off frequency for filter
            ('AmpPhase', ctypes.c_int),
            ('CardCutoffFreq', ctypes.c_int),
            ('ResamKernel', ctypes.c_char*256),
            ('FIROrder', ctypes.c_int),
            ('as_windwidth', ctypes.c_int),
            ('as_percover', ctypes.c_int),
            ('as_fftwin', ctypes.c_int)
            ]

    def __init__(self, VolTR, PhysFS, tshift=0):
        self.VolTR = VolTR
        self.PhysFS = PhysFS
        self.tshift = tshift


# %% ==========================================================================
class RTP_RETROTS:

    def __init__(self):
        librtp = ctypes.cdll.LoadLibrary(librtp_path)

        # -- Define rtp_align_setup --
        self.rtp_retrots = librtp.rtp_retrots
        """
        int rtp_retrots(RetroTSOpt *rtsOpt, double *Resp, double *ECG, int len,
                        *regOut);
        """

        self.rtp_retrots.argtypes = \
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
             ctypes.c_void_p]

        self.rtsOpt = None

    def setup_RetroTSOpt(self, TR, PhysFS, tshift=0):
        self.TR = TR
        self.PhysFS = PhysFS
        self.tshift = tshift
        self.rtsOpt = RetroTSOpt(TR, PhysFS, tshift)

    def do_proc(self, Resp, ECG, TR, PhysFS, tshift=0):
        """
        RetroTS process function, which will be called from RTP_PHYSIO instance

        Options
        -------
        Resp: list
            respiration data array
        ECG: list
            ECG data array

        Retrun
        ------
        RetroTS regressor
        """

        # Get pointer ot Resp and ECG data array
        Resp_arr = np.array(Resp, dtype=np.float64)
        Resp_ptr = Resp_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        ECG_arr = np.array(ECG, dtype=np.float64)
        ECG_ptr = ECG_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        # Set data length and prepare output array
        dlenR = len(Resp)
        dlenE = len(ECG)
        dlen = min(dlenR, dlenE)

        outlen = int((dlen * 1.0/PhysFS) / TR)
        regOut = np.ndarray((outlen, 13), dtype=np.float32)
        regOut_ptr = regOut.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        # Set rtsOpt
        if self.rtsOpt is None:
            self.setup_RetroTSOpt(TR, PhysFS, tshift)
        else:
            self.rtsOpt.VolTR = TR
            self.PhysFS = PhysFS
            self.tshift = tshift

        self.rtp_retrots(ctypes.byref(self.rtsOpt), Resp_ptr, ECG_ptr, dlen,
                         regOut_ptr)

        return regOut

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_param(self, attr, val=None, echo=True):
        # -- Set value --
        setattr(self, attr, val)
        if echo:
            print("{}.".format(self.__class__.__name__) + attr, '=',
                  getattr(self, attr))


# %% ==========================================================================
if __name__ == '__main__':

    # Test
    rtp_retots = RTP_RETROTS()

    # Set test data
    sample_dir = '../../test_data/E13270/'
    Resp_f = os.path.join(sample_dir, 'Resp_epiRTeeg_scan_7.1D')
    ECG_f = os.path.join(sample_dir, 'ECG_epiRTeeg_scan_7.1D')

    respstr = open(Resp_f, 'r').read()
    Resp = np.array([float(v) for v in respstr.rstrip().split('\n')],
                    dtype=np.float64)
    ecgstr = open(ECG_f, 'r').read()
    ECG = np.array([float(v) for v in ecgstr.rstrip().split('\n')],
                   dtype=np.float64)

    # Run proc
    TR = 2.0
    PhysFS = 40
    tshift = 0
    rtp_retots.setup_RetroTSOpt(TR, PhysFS, tshift)

    regOut = rtp_retots.do_proc(Resp, ECG)

    # Load reference result
    res_f = os.path.join(sample_dir, 'oba.slibase.scan_7.1D')
    C = open(res_f, 'r').read()
    colnames = [cn.strip() for cn in
                re.search('# ColumnLabels = "(.*)"',
                          C).groups()[0].rstrip().split(';')]
    regAll = np.ndarray((0, len(colnames)))
    for l in C.rstrip().split('\n'):
        if re.match('\s*#', l):
            continue

        regrow = np.reshape([float(v) for v in l.strip().split()], (1, -1))
        regAll = np.concatenate((regAll, regrow))

    ci = np.argwhere(['s0.' in cn for cn in colnames]).flatten()
    reg0 = regAll[:, ci]

    import matplotlib.pyplot as plt
    for ii in range(regOut.shape[1]):
        plt.figure()
        plt.plot(reg0[:regOut.shape[0], ii], label='RetoroTS.m')
        plt.plot(regOut[:, ii], label='RTP_RETROTS')
        plt.legend()
        r = np.corrcoef(reg0[:regOut.shape[0], ii], regOut[:, ii])[0, 1]
        plt.title("{}: r={}".format(colnames[ci[ii]], r))

    """DEBUG
    tmpf = 'RVTRS.bin'
    RVTRS = np.fromfile(tmpf, dtype=np.float64)
    tmpf = 'RVTR.bin'
    RVTR = np.fromfile(tmpf, dtype=np.float64)
    """
