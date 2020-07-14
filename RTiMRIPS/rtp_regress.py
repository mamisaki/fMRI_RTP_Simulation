#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mmisaki@laureateinstitute.org
"""


# %% import ===================================================================
from pathlib import Path
import sys
import time
import re
import numpy as np
import nibabel as nib
from six import string_types
import traceback
from PyQt5 import QtWidgets
import torch

import threading

from .rtp_common import RTP, MRI_data, MatplotlibWindow

onGPU = torch.cuda.is_available()


# %% lstsq_SVDsolver ==========================================================
def lstsq_SVDsolver(A, B, rcond=None):
    """
    Solve a linear system Ax = b in least square sense (minimize ||Ax-b||^2)
    using SVD.

    Parameters
    ----------
    A: 2D tensor (m x n)
        Number of rows (m) must be >= number of columns (n)
    B: 2D tensor (m x k)

    rcond: float, optional
        Cut-off ratio for small singular values. For the purposes of rank
        determination, singular values are treated as zero if they are smaller
        than rcond times the largest singular value.
        Default will use the machine precision times len(s).

    Returns
    -------
    x: 2D tensor (n x k)
    """

    if rcond is None:
        rcond = np.finfo(np.float32).eps * min(A.shape)

    # SVD for A
    U, S, V = torch.svd(A)

    # Clip singular values < S[0] * rcond
    sthresh = S[0] * rcond
    for ii in range(len(S)):
        if S[ii] < sthresh:
            break
        else:
            r = ii+1
    # r is rank of A

    # Diagnal matrix with Sinv: D = diag(1/S)
    Sinv = torch.zeros(len(S)).to(S.device)
    Sinv[:r] = 1.0/S[:r]
    D = torch.diag(Sinv)

    # X = V*D*UT*B
    X = torch.chain_matmul(V, D, U.transpose(1, 0), B)

    return X


# %% RTP_REGRESS class ========================================================
class RTP_REGRESS(RTP):
    """
    Real-time online General Linear Model Analysis
    """

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __init__(self, desMtx=None, wait_num=0, mot_reg='None', volreg=None,
                 GS_reg=False, WM_reg=False, Vent_reg=False,
                 mask_src_proc=None, phys_reg='None', physio=None,
                 max_poly_order=np.inf, TR=2.0, tshift=0,
                 mask_file=0, onGPU=onGPU, **kwargs):
        """
        Parameters
        ----------
        desMtx: 2D array
            design matrix other than motion, retrots, and polunomial regressors
        max_scan_length: int, optional
            maximum scan length. If desMtx is not None and
            max_scan_length < desMtx.shape[0], max_scan_length is set by
            desMtx.shape[0].
            This value is used to alloc memory space for X and Y data.
        wait_num: int
            minimum number of volumes to wait before staring REGRESS
        mot_reg: str, ['None'|'mot6'|'mot12'|'dmot6']
            None; no motion regressor
            mot6; six motions; yaw, pitch, roll, dx, dy, dz
            mot12; mot6 plus their temporal derivative
            dmot6; six motion derivatives
        volreg: object
            RTP_VOLREG instance to read motion parameter
        GS_reg: bool
            Regress out mean global mean signal
        WM_reg: bool
            Regress out mean white matter signal
        Vent_reg: bool
            Regress out ventricle signal
        mask_src_proc: RTP class object
            Source of image mask regressor. It should be unsmoothed one.
        pysh_reg: str, ['None'|'RVT5'|'RICOR8'|'RVT+RICOR13']
            None: no physio regressor
            RVT5: five RVT regressors
            RICOR8: four Resp and four Card regressors
            RVT+RICOR13: both RVT5 and RICOR8
        physio: object
            RTP_PHYSIO object to read retrots regressors
        max_poly_order: int, optional
            maximum number of polynomial regressors. Default is inf. Polynomial
            regressor is increasing automatically according to the scan length.
        mask_file: 0 or str
            filename of mask
            If mask_file == 0, zero_out initial volume is used as mask.
        TR: float
            TR in seconds
        tshift: float
            slice timing offset (second) for restrots. 0.0 is the first slice.
        onGPU: bool
            Run REGRESS on GPU
        """

        super(RTP_REGRESS, self).__init__(**kwargs)

        # Set instance parameters
        self.desMtx_read = desMtx  # Design matrix used in REGRESS
        self.mot_reg = mot_reg
        self.volreg = volreg
        self.mot0 = None
        self.physio = physio
        self.phys_reg = phys_reg

        self.max_poly_order = max_poly_order
        self.mask_file = mask_file

        self.TR = TR
        self.tshift = tshift
        self.onGPU = onGPU

        self.GS_reg = GS_reg
        self.GS_mask = ''
        self.GS_maskdata = None

        self.WM_reg = WM_reg
        self.WM_mask = ''
        self.WM_maskdata = None

        self.Vent_reg = Vent_reg
        self.Vent_mask = ''
        self.Vent_maskdata = None

        self.mask_src_proc = mask_src_proc

        # Initialize working data
        self.maskV = None
        self.reg_names = []
        self.col_names_read = []
        self.desMtx0 = None  # Initial design matrix
        self.desMtx = None
        self.YMtx = None  # Data matrix
        self.Y_mean = None  # Mean signal for data scaling

        reg_num = self.get_reg_num()
        self.set_wait_num(reg_num+1)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # onGPU getter, setter
    @property
    def onGPU(self):
        return self.device != 'cpu'

    @onGPU.setter
    def onGPU(self, _onGPU):
        if _onGPU:
            if torch.cuda.is_available():
                self.device = 'cuda'
            else:
                self.errmsg("CUDA device is not available.")
                self.device = 'cpu'
        else:
            self.device = 'cpu'

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    @property
    def proc_ready(self):
        self._proc_ready = True

        if self.physio is None or self.physio.not_available:
            self.set_param('phys_reg', 'None')

        if self.desMtx0 is None:
            self.errmsg('Design matrix or max scanlength is not set.')
            self._proc_ready = False

        if self.mot_reg != 'None' and self.volreg is None:
            self.errmsg('RTP_VOLREG object is not set.')
            self._proc_ready = False

        if self.phys_reg != 'None' and self.physio is None:
            self.errmsg('RTP_PHYSIO object is not set.')
            self._proc_ready = False

        if (self.GS_reg or self.WM_reg or self.Vent_reg) \
                and self.mask_src_proc is None:
            self.errmsg('mask_src_proc must be set for GS, WM, and Vent reg.')
            self._proc_ready = False

        if self.TR is None:
            self.errmsg('TR is not set.')
            self._proc_ready = False

        if self.next_proc:
            self._proc_ready &= self.next_proc.proc_ready

        # Prepare design matrix
        if self._proc_ready:
            self.setup_regressor_template(self.desMtx_read,
                                          self.max_scan_length,
                                          self.col_names_read)
            if isinstance(self.mask_file, string_types) or \
                    isinstance(self.mask_file, Path):
                self.set_mask(self.mask_file)

            self.set_wait_num(self.wait_num)

        return self._proc_ready

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_mask(self, maskdata, sub_i=0, method='zero_out'):
        if isinstance(maskdata, string_types) or isinstance(maskdata, Path):
            if not Path(maskdata).is_file():
                self.errmsg(f"Not found mask file: {maskdata}")
                self.mask_file = 0
                return

            if self._verb:
                msg = f"Mask is set by {maskdata}"
                self.logmsg(msg)
            maskdata = np.squeeze(nib.load(maskdata).get_data())

            if maskdata.ndim > 3:
                maskdata = maskdata[:, :, :, sub_i]

        if method == 'zero_out':
            self.maskV = maskdata > 0

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def setup_regressor_template(self, desMtx_read=None, max_scan_length=0,
                                 col_names_read=[]):
        if desMtx_read is None and max_scan_length == 0:
            self.errmsg('Either desMtx or max_scan_length must be given.')
            return

        col_names = col_names_read.copy()

        # Set provided design matrix
        if desMtx_read is None:
            desMtx = np.zeros([max_scan_length, 0], dtype=np.float32)
        else:
            if max_scan_length <= desMtx_read.shape[0]:
                max_scan_length = desMtx_read.shape[0]
            else:
                desMtx = np.zeros([max_scan_length, desMtx_read.shape[1]],
                                  dtype=np.float32)
                desMtx[:desMtx_read.shape[0], :] = desMtx_read

            # Adjust col_names and desMtx columns
            if len(col_names) > desMtx.shape[1]:
                col_names = col_names[:desMtx.shape[1]]
            elif len(col_names) < desMtx.shape[1]:
                while len(col_names) < desMtx.shape[1]:
                    col_names.append("Reg{len(col_names)+2}")

        # Append nuisunce regressors
        if self.mot_reg != 'None':
            if self.mot_reg in ('mot6', 'mot12'):
                # Append 6 motion parameters
                desMtx = np.concatenate(
                    [desMtx, np.zeros([max_scan_length, 6])], axis=1)
                col_names.extend(['roll', 'pitch', 'yaw', 'dS', 'dL', 'dP'])
                self.motcols = \
                    [ii for ii, cn in enumerate(col_names)
                     if cn in ('roll', 'pitch', 'yaw', 'dS', 'dL', 'dP')]
            else:
                self.motcols = []

            if self.mot_reg in ('mot12', 'dmot6'):
                # Append 6 motion derivative parameters
                desMtx = np.concatenate(
                        [desMtx, np.zeros([max_scan_length, 6])], axis=1)
                col_names.extend(['dtroll', 'dtpitch', 'dtyaw', 'dtdS', 'dtdL',
                                  'dtdP'])
                self.motcols.extend(
                        [ii for ii, cn in enumerate(col_names)
                         if cn in ('dtroll', 'dtpitch', 'dtyaw', 'dtdS',
                                   'dtdL', 'dtdP')])

        if self.phys_reg != 'None':
            # Append RVT, RETROICOR regresors
            if self.phys_reg == 'RVT5':
                nreg = 5
                col_add = ['RVT0', 'RVT1', 'RVT2', 'RVT3', 'RVT4']
            elif self.phys_reg == 'RICOR8':
                nreg = 8
                col_add = ['Resp0', 'Resp1', 'Resp2', 'Resp3',
                           'Card0', 'Card1', 'Card2', 'Card3']
            elif self.phys_reg == 'RVT+RICOR13':
                nreg = 13
                col_add = ['RVT0', 'RVT1', 'RVT2', 'RVT3', 'RVT4',
                           'Resp0', 'Resp1', 'Resp2', 'Resp3',
                           'Card0', 'Card1', 'Card2', 'Card3']
            desMtx = np.concatenate([desMtx,
                                     np.zeros([max_scan_length, nreg])],
                                    axis=1)
            col_names.extend(col_add)
            self.retrocols = \
                [ii for ii, cn in enumerate(col_names) if cn in col_add]

        if self.GS_reg:
            # Append global signal regressor
            desMtx = np.concatenate([desMtx,
                                     np.zeros([max_scan_length, 1])],
                                    axis=1)
            col_names.append('GS')
            self.GS_col = col_names.index('GS')

        if self.WM_reg:
            # Append mean WM signal regressor
            desMtx = np.concatenate([desMtx,
                                     np.zeros([max_scan_length, 1])],
                                    axis=1)
            col_names.append('WM')
            self.WM_col = col_names.index('WM')

        if self.Vent_reg:
            # Append mean ventricle signal regressor
            desMtx = np.concatenate([desMtx,
                                     np.zeros([max_scan_length, 1])],
                                    axis=1)
            col_names.append('Vent')
            self.Vent_col = col_names.index('Vent')

        self.desMtx0 = desMtx
        self.reg_names = col_names

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def legendre(self, x, m):
        """
        Legendre polynomial calculator
        Copied from misc_math.c in afni src

        Parameters
        ----------
        x: 1D array
            series of -1 .. 1 values
        m: int
            polynomial order
        """

        if m < 0:
            return None  # bad input
        elif m == 0:
            return np.ones_like(x)
        elif m == 1:
            return x
        elif m == 2:
            return (3.0 * x * x - 1.0)/2.0
        elif m == 3:
            return (5.0 * x * x - 3.0) * x/2.0
        elif m == 4:
            return ((35.0 * x * x - 30.0) * x * x + 3.0)/8.0
        elif m == 5:
            return ((63.0 * x * x - 70.0) * x * x + 15.0) * x/8.0
        elif m == 6:
            return (((231.0 * x * x - 315.0) * x * x + 105.0) * x * x - 5.0) \
                   / 16.0
        elif m == 7:
            return (((429.0 * x * x - 693.0) * x * x + 315.0) * x * x - 35.0) \
                   * x/16.0
        elif m == 8:
            return ((((6435.0 * x * x - 12012.0) * x * x + 6930.0) * x * x
                    - 1260.0) * x * x + 35.0) / 128.0
        elif m == 9:
            #  Feb 2005: this part generated by Maple, then hand massaged
            return (0.24609375e1 +
                    (-0.3609375e2 +
                     (0.140765625e3 +
                      (-0.20109375e3 +
                       0.949609375e2 * x * x) * x * x) * x * x) * x * x) * x
        elif m == 10:
            return -0.24609375e0 + \
                (0.1353515625e2 +
                 (-0.1173046875e3 +
                  (0.3519140625e3 +
                   (-0.42732421875e3 +
                    0.18042578125e3 * x * x)
                   * x * x) * x * x) * x * x) * x * x
        elif m == 11:
            return (-0.270703125e1 +
                    (0.5865234375e2 +
                     (-0.3519140625e3 +
                      (0.8546484375e3 +
                       (-0.90212890625e3 +
                        0.34444921875e3 * x * x)
                       * x * x) * x * x) * x * x) * x * x)
        elif m == 12:
            return 0.2255859375e0 + \
                (-0.17595703125e2 +
                 (0.2199462890625e3 +
                  (-0.99708984375e3 +
                   (0.20297900390625e4 +
                    (-0.1894470703125e4 +
                     0.6601943359375e3 * x * x) * x * x)
                   * x * x) * x * x) * x * x) * x * x
        elif m == 13:
            return (0.29326171875e1 +
                    (-0.87978515625e2 +
                     (0.7478173828125e3 +
                      (-0.270638671875e4 +
                       (0.47361767578125e4 +
                        (-0.3961166015625e4
                         + 0.12696044921875e4 * x * x) * x * x) * x * x)
                      * x * x) * x * x) * x * x) * x
        elif m == 14:
            return -0.20947265625e0 + \
                (0.2199462890625e2 +
                 (-0.37390869140625e3 +
                  (0.236808837890625e4 +
                   (-0.710426513671875e4 +
                    (0.1089320654296875e5 +
                     (-0.825242919921875e4 +
                      0.244852294921875e4 * x * x) * x * x) * x * x)
                   * x * x) * x * x) * x * x) * x * x
        elif m == 15:
            return (-0.314208984375e1 +
                    (0.12463623046875e3 +
                     (-0.142085302734375e4 +
                      (0.710426513671875e4 +
                       (-0.1815534423828125e5 +
                        (0.2475728759765625e5 +
                         (-0.1713966064453125e5 +
                          0.473381103515625e4 * x * x) * x * x) * x * x)
                       * x * x) * x * x) * x * x) * x * x) * x
        elif m == 16:
            return 0.196380615234375e0 + \
                (-0.26707763671875e2 +
                 (0.5920220947265625e3 +
                  (-0.4972985595703125e4 +
                   (0.2042476226806641e5 +
                    (-0.4538836059570312e5 +
                     (0.5570389709472656e5 +
                      (-0.3550358276367188e5 +
                       0.9171758880615234e4 * x * x) * x * x) * x * x)
                    * x * x) * x * x) * x * x) * x * x) * x * x
        elif m == 17:
            return (0.3338470458984375e1 +
                    (-0.169149169921875e3 +
                     (0.2486492797851562e4 +
                      (-0.1633980981445312e5 +
                       (0.5673545074462891e5 +
                        (-0.1114077941894531e6 +
                         (0.1242625396728516e6 +
                          (-0.7337407104492188e5 +
                           0.1780400253295898e5 * x * x) * x * x) * x * x)
                        * x * x) * x * x) * x * x) * x * x) * x * x) * x
        elif m == 18:
            return -0.1854705810546875e0 + \
                (0.3171546936035156e2 +
                 (-0.8880331420898438e3 +
                  (0.9531555725097656e4 +
                   (-0.5106190567016602e5 +
                    (0.153185717010498e6 +
                     (-0.2692355026245117e6 +
                      (0.275152766418457e6 +
                       (-0.1513340215301514e6 +
                        0.3461889381408691e5 * x * x) * x * x) * x * x)
                     * x * x) * x * x) * x * x) * x * x) * x * x) * x * x
        elif m == 19:
            return (-0.3523941040039062e1 +
                    (0.2220082855224609e3 +
                     (-0.4084952453613281e4 +
                      (0.3404127044677734e5 +
                       (-0.153185717010498e6 +
                        (0.4038532539367676e6 +
                         (-0.6420231216430664e6 +
                          (0.6053360861206055e6 +
                           (-0.3115700443267822e6 +
                            0.6741574058532715e5 * x * x) * x * x)
                          * x * x) * x * x) * x * x) * x * x) * x * x)
                        * x * x) * x * x) * x
        elif m == 20:
            return 0.1761970520019531e0 + \
                (-0.3700138092041016e2 +
                 (0.127654764175415e4 +
                  (-0.1702063522338867e5 +
                   (0.1148892877578735e6 +
                    (-0.4442385793304443e6 +
                     (0.1043287572669983e7 +
                      (-0.1513340215301514e7 +
                       (0.1324172688388824e7 +
                        (-0.6404495355606079e6 +
                         0.1314606941413879e6 * x * x) * x * x)
                       * x * x) * x * x) * x * x) * x * x) * x * x) * x * x)
                    * x * x) * x * x
        else:
            # if here, m > 20 ==> use recurrence relation
            pk = 0
            pkm2 = self.legendre(x, 19)
            pkm1 = self.legendre(x, 20)
            for k in range(21, m+1):
                pk = ((2.0 * k - 1.0) * x * pkm1 - (k - 1.0) * pkm2) / k
                pkm2 = pkm1
                pkm1 = pk

            return pk

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def poly_reg(self, nt, TR):
        """
        Make legendre polynomial regressor or nt length data

        Option
        ------
        nt: int
            data length (must be > 1)

        Retrun
        ------
        polyreg: nt * x array
            Matrix of Legendre polynomial regressors

        """

        # If nt is not enough even for linear trend, return
        if nt < 1:
            return None

        # Set polynomial order
        pnum = min(1 + int(nt*TR/150), self.max_poly_order)
        polyreg = np.ndarray((nt, pnum+1), dtype=np.float32)

        for po in range(pnum+1):
            xx = np.linspace(-1.0, 1.0, nt)
            polyreg[:, po] = self.legendre(xx, po)

        return polyreg

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def do_proc(self, mri_data, vol_idx=None, pre_proc_time=None, maskV=None,
                **kwargs):
        try:
            # Increment the number of received volume
            self.vol_num += 1
            if vol_idx is None:
                vol_idx = self.vol_num

            if not hasattr(self, 'rm1st'):
                self.rm1st = vol_idx - self.vol_num

            assert self.vol_num == vol_idx - self.rm1st

            # --- Initialization ----------------------------------------------
            # Set maskV
            if self.maskV is None:
                # Prepare mask
                if maskV is not None:
                    self.set_mask(maskV)
                    if self._verb:
                        msg = "Mask is set from the previous process."
                        self.logmsg(msg)
                else:
                    self.set_mask(mri_data.img_data)
                    if self._verb:
                        msg = f"Mask is set with a volume index {vol_idx}."
                        self.logmsg(msg)

            # Read GS mask
            if self.GS_reg and self.GS_maskdata is None:
                if Path(self.GS_mask).is_file():
                    GSimg = nib.load(self.GS_mask)
                    if not np.all(mri_data.img_data.shape == GSimg.shape):
                        errstr = f"GS mask shape {GSimg.shape} mismatches"
                        errstr += " to function image data"
                        errstr += f" {mri_data.img_data.shape}"
                        self.errmsg(errstr, no_pop=True)
                        self.errmsg("GS_reg is reset to False.", no_pop=True)
                        self.GS_reg = False
                    else:
                        self.GS_maskdata = (GSimg.get_fdata() != 0)[self.maskV]
                else:
                    errstr = f"Not found GS_mask file {self.GS_mask}"
                    self.errmsg(errstr, no_pop=True)
                    self.errmsg("GS_reg is reset to False.", no_pop=True)
                    self.GS_reg = False

            # Read WM mask
            if self.WM_reg and self.WM_maskdata is None:
                if Path(self.WM_mask).is_file():
                    WMimg = nib.load(self.WM_mask)
                    if not np.all(mri_data.img_data.shape == WMimg.shape):
                        errstr = f"WM mask shape {WMimg.shape} mismatches"
                        errstr += " to function image data"
                        errstr += f" {mri_data.img_data.shape}"
                        self.errmsg(errstr, no_pop=True)
                        self.errmsg("WM_reg is reset to False.", no_pop=True)
                        self.WM_reg = False
                    else:
                        self.WM_maskdata = (WMimg.get_fdata() != 0)[self.maskV]
                else:
                    errstr = f"Not found WM_mask file {self.WM_mask}"
                    self.errmsg(errstr, no_pop=True)
                    self.errmsg("WM_reg is reset to False.", no_pop=True)
                    self.WM_reg = False

            # Read Vent mask
            if self.Vent_reg and self.Vent_maskdata is None:
                if Path(self.Vent_mask).is_file():
                    Ventimg = nib.load(self.Vent_mask)
                    if not np.all(mri_data.img_data.shape == Ventimg.shape):
                        errstr = f"Vet mask shape {Ventimg.shape}"
                        errstr += " mismatches to function image data"
                        errstr += f" {mri_data.img_data.shape}"
                        self.errmsg(errstr, no_pop=True)
                        self.errmsg("Vent_reg is reset to False.", no_pop=True)
                        self.Vent_reg = False
                    else:
                        self.Vent_maskdata = \
                            (Ventimg.get_fdata() != 0)[self.maskV]
                else:
                    errstr = f"Not found Vent_mask file {self.Vent_mask}"

                    self.errmsg(errstr, no_pop=True)
                    self.errmsg("Vent_reg is reset to False.", no_pop=True)
                    self.Vent_reg = False

            # Initialize design matrix
            if self.desMtx is None:
                # Update self.desMtx0 and self.reg_names
                self.setup_regressor_template(self.desMtx_read,
                                              self.max_scan_length,
                                              self.col_names_read)
                self.set_wait_num(self.wait_num)

                desMtx = self.desMtx0[self.rm1st:, :].copy()

                # Add maximum number of polynomial regressors
                nt = desMtx.shape[0]
                pnum = min(1 + int(nt*self.TR/150), self.max_poly_order)
                desMtx = np.concatenate(
                        [desMtx, np.zeros((nt, pnum+1), dtype=np.float32)],
                        axis=1)
                self.desMtx = torch.from_numpy(
                        desMtx.astype(np.float32)).to(self.device)

            # Initialize Y matrix
            if self.YMtx is None:
                vox_num = np.sum(self.maskV)
                self.YMtx = torch.empty(
                        self.desMtx0.shape[0]-self.rm1st, vox_num,
                        dtype=torch.float32)
                try:
                    self.YMtx = self.YMtx.to(self.device)
                except Exception as e:
                    self.errmsg(str(e), no_pop=True)
                    self.errmsg("Failed to keep GPU tensor for Y.",
                                no_pop=True)
                    self.onGPU = False
                    self.desMtx = self.desMtx.to(self.device)
                    self.YMtx = self.YMtx.to(self.device)
                    raise e

            # --- Append data -------------------------------------------------
            ydata = mri_data.img_data[self.maskV]
            ydata = torch.from_numpy(ydata.astype(np.float32)).to(self.device)
            if self.Y_mean is not None:
                # Scaling data
                ydata[self.Y_mean_mask] = \
                    ydata[self.Y_mean_mask]/self.Y_mean[self.Y_mean_mask]*100
                ydata[ydata > 200] = 200

            self.YMtx[self.vol_num, :] = ydata

            # -- Update design matrix --
            # Append motion parameter
            if self.mot_reg != 'None':
                mot = self.volreg.motion[vol_idx, :]
                if self.mot_reg in ('mot12', 'dmot6'):
                    if self.vol_num > 0 and self.mot0 is not None:
                        dmot = mot - self.mot0
                    else:
                        dmot = np.zeros(6, dtype=np.float32)
                    self.mot0 = mot

                if self.mot_reg in ('mot6', 'mot12'):
                    mot = torch.from_numpy(
                        mot.astype(np.float32)).to(self.device)
                if self.mot_reg in ('mot12', 'dmot6'):
                    dmot = torch.from_numpy(
                        dmot.astype(np.float32)).to(self.device)

                # Assuming self.motcols is contiguous
                if self.mot_reg in ('mot6', 'mot12'):
                    self.desMtx[self.vol_num,
                                self.motcols[0]:self.motcols[0]+6] = mot
                    if self.mot_reg == 'mot12':
                        self.desMtx[self.vol_num,
                                    self.motcols[6]:self.motcols[6]+6] = dmot
                elif self.mot_reg == 'dmot6':
                    self.desMtx[self.vol_num,
                                self.motcols[0]:self.motcols[0]+6] = dmot

            # Append retroicor regressors
            if self.phys_reg != 'None' and self.physio is not None:
                retrots = self.physio.get_retrots(self.TR, vol_idx+1,
                                                  self.tshift)[self.rm1st:, :]
                if self.phys_reg == 'RVT5':
                    retrots = retrots[:, :5]
                elif self.phys_reg == 'RICOR8':
                    retrots = retrots[:, 5:]

                for ii, icol in enumerate(self.retrocols):
                    self.desMtx[:self.vol_num+1, icol] = \
                            torch.from_numpy(
                                    retrots[:, ii].astype(np.float32)
                                    ).to(self.device)

            # Append mask mean signal regressor from mask_src_proc
            if self.GS_reg or self.WM_reg or self.Vent_reg:
                msk_src_data = self.mask_src_proc.proc_data[self.maskV]
                if self.GS_reg:
                    self.desMtx[self.vol_num, self.GS_col] =  \
                        float(msk_src_data[self.GS_maskdata].mean())

                if self.WM_reg:
                    self.desMtx[self.vol_num, self.WM_col] =  \
                        float(msk_src_data[self.WM_maskdata].mean())

                if self.Vent_reg:
                    self.desMtx[self.vol_num, self.Vent_col] = \
                        float(msk_src_data[self.Vent_maskdata].mean())

            # --- Perform regression ------------------------------------------
            # If the number of samples is not enough, retrun without process
            if self.vol_num+1 < self.wait_num:
                if self._verb:
                    wait_idx = self.rm1st+self.wait_num-1
                    msg = f"Wait until volume number {wait_idx}"
                    self.logmsg(msg)
                return

            # Set Y_mean for scaling data
            if self.Y_mean is None:
                # Scaling
                YMtx = self.YMtx[:self.vol_num+1, :]
                self.Y_mean = YMtx.mean(axis=0)
                self.Y_mean_mask = self.Y_mean.abs() > 1e-6

                YMtx[:, self.Y_mean_mask] = \
                    YMtx[:, self.Y_mean_mask] / \
                    self.Y_mean[self.Y_mean_mask] * 100
                YMtx[YMtx > 200] = 200
                YMtx[:, ~self.Y_mean_mask] = 0.0
                # self.YMtx[:self.vol_num+1, :] = YMtx  # No need to return

                ydata = self.YMtx[self.vol_num, :]

            # Add polynomial in the design matrix
            polyreg = self.poly_reg(self.vol_num+1, self.TR)
            reg0_num = self.desMtx0.shape[1]
            polyreg_num = polyreg.shape[1]
            self.desMtx[:self.vol_num+1, reg0_num:reg0_num+polyreg_num] = \
                torch.from_numpy(polyreg).to(self.device)

            # Extract regressor
            Xp = self.desMtx[:self.vol_num+1, :reg0_num+polyreg_num].clone()

            # Standardizing regressors of motion, GS, WM, Vent
            norm_regs = ('roll', 'pitch', 'yaw', 'dS', 'dL', 'dP',
                         'dtroll', 'dtpitch', 'dtyaw', 'dtdS', 'dtdL', 'dtdP',
                         'GS', 'WM', 'Vent')
            for ii, reg_name in enumerate(self.reg_names):
                if reg_name in norm_regs:
                    reg = Xp[:, ii]
                    reg = (reg - reg.mean()) / reg.std()
                    Xp[:, ii] = reg

            # Extract Y
            Yp = self.YMtx[:self.vol_num+1, :]

            # calculate Beta with least sqare error of ||Y - XB||^2
            Beta = lstsq_SVDsolver(Xp, Yp[:, self.Y_mean_mask])
            Yh = torch.matmul(Xp, Beta)

            # Output only the last (current) volume
            Resid = ydata[self.Y_mean_mask] - Yh[-1, :]

            del Beta
            del Yh

            mri_data.img_data = np.zeros_like(self.maskV, dtype=np.float32)
            dat = np.zeros(np.sum(self.maskV), dtype=np.float32)
            dat[self.Y_mean_mask.cpu().numpy()] = Resid.cpu().numpy()
            mri_data.img_data[self.maskV] = dat

            # Save the processed data
            self.proc_data = mri_data.img_data.copy()

            # Record process time
            self.proc_time.append(time.time())
            if pre_proc_time is not None:
                proc_delay = self.proc_time[-1] - pre_proc_time
                if self.save_delay:
                    self.proc_delay.append(proc_delay)

            # log message
            if self._verb:
                f = Path(mri_data.save_name).name
                msg = f'{vol_idx}, Regression is done for {f}'
                if pre_proc_time is not None:
                    msg += f' (process time {proc_delay:.4f}s)'
                msg += '.'
                self.logmsg(msg)

            # Set save_name
            mri_data.save_name = 'regRes.' + mri_data.save_name

            # Run the next process
            if self.next_proc:
                self.next_proc.do_proc(mri_data, vol_idx, self.proc_time[-1])

            # Save processed image
            if self.save_proc:
                self.keep_processed_image(mri_data)

        except Exception:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            errmsg = f'{exc_type}, {exc_tb.tb_frame.f_code.co_filename}' + \
                     f':{exc_tb.tb_lineno}'
            self.errmsg(errmsg, no_pop=True)
            traceback.print_exc(file=self._err_out)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def get_reg_num(self):
        # update self.desMtx0 (regressor template)
        self.setup_regressor_template(self.desMtx_read,
                                      max_scan_length=self.max_scan_length,
                                      col_names_read=self.col_names_read)
        numReg = self.desMtx0.shape[1]
        numPolyReg = min(1 + int(numReg*self.TR/150), self.max_poly_order) + 1

        return numReg + numPolyReg

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_wait_num(self, wait_num=0):
        reg_num = self.get_reg_num()
        min_wait_num = max(int(np.ceil(reg_num*1.3)), reg_num+1)
        if wait_num < min_wait_num:
            wait_num = min_wait_num

        if hasattr(self, 'ui_waitNum_lb'):
            self.ui_waitNum_lb.setText(
                    f"Wait REGRESS until receiving {wait_num} volumes")

        self.wait_num = wait_num

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def reset(self):
        """Reset process parameters
        """

        if self.verb:
            self.logmsg(f"Reset {self.__class__.__name__} module.")

        if not isinstance(self.mask_file, string_types) and \
                not isinstance(self.mask_file, Path):
            self.maskV = None

        self.desMtx = None
        self.mot0 = None
        self.YMtx = None
        self.Y_mean = None
        self.Y_mean_mask = None
        self.GS_maskdata = None
        self.WM_maskdata = None
        self.Vent_maskdata = None

        if hasattr(self, 'rm1st'):
            del self.rm1st

        super(RTP_REGRESS, self).reset()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_param(self, attr, val=None, reset_fn=None, echo=False):
        # -- check value --
        if attr == 'enabled':
            if hasattr(self, 'ui_enabled_rdb'):
                self.ui_enabled_rdb.setChecked(val)

            if hasattr(self, 'ui_objs'):
                for ui in self.ui_objs:
                    ui.setEnabled(val)

            if self.desMtx_read is None and hasattr(self, 'ui_showDesMtx_btn'):
                self.ui_showDesMtx_btn.setEnabled(False)

        elif attr in ('watch_dir', 'save_dir'):
            pass

        elif attr == 'wait_num':
            if type(val) == int:
                self.set_wait_num(val)
                if reset_fn is None:
                    if hasattr(self, 'ui_waitNum_cmbBx'):
                        if self.wait_num == self.get_reg_num()+1:
                            self.ui_waitNum_cmbBx.setCurrentIndex(0)
                        else:
                            self.ui_waitNum_cmbBx.setCurrentIndex(1)
                return

            elif 'regressor' in val:
                self.set_wait_num(0)

            elif 'set' in val:
                num0 = self.get_reg_num()
                num1, okflag = QtWidgets.QInputDialog.getInt(
                        None, "Wait REGRESS until", "volume", self.wait_num,
                        num0)
                if not okflag:
                    return

                self.set_wait_num(num1)

            return

        elif attr == 'desMtx_f':
            if val is None:
                return

            elif (val == 'set' and
                  'Unset' not in self.ui_loadDesMtx_btn.text()) or \
                    Path(val).is_file():
                if val == 'set':
                    fname = self.select_file_dlg(
                            'REGRESS: Selct design matrix file',
                            self.watch_dir, "*.csv")
                    if fname[0] == '':
                        return -1
                    fname = fname[0]
                else:
                    fname = val

                self.desMtx_f = fname

                # Have header?
                ll = open(fname, 'r').readline()
                if np.any([isinstance(cc, string_types)
                           for cc in ll.split(',')]):
                    self.col_names_read = ll.split()
                    skiprows = 1
                else:
                    skiprows = 0

                self.desMtx_read = np.loadtxt(fname, delimiter=',',
                                              skiprows=skiprows)
                if self.desMtx_read.ndim == 1:
                    self.desMtx_read = self.desMtx_read[:, np.newaxis]

                if hasattr(self, 'ui_showDesMtx_btn'):
                    self.ui_showDesMtx_btn.setEnabled(True)

                if self.desMtx_read.shape[0] > self.max_scan_length:
                    self.set_param('max_scan_length',
                                   self.desMtx_read.shape[0])

                if hasattr(self, 'ui_loadDesMtx_btn'):
                    self.ui_loadDesMtx_btn.setText('Unset')

            elif val == 'unset' or (val == 'set' and
                                    'Unset' in self.ui_loadDesMtx_btn.text()):
                self.desMtx_read = None
                self.desMtx_f = None
                self.col_names_read = []
                if hasattr(self, 'ui_loadDesMtx_btn'):
                    self.ui_loadDesMtx_btn.setText('Set')

                if hasattr(self, 'ui_showDesMtx_btn'):
                    self.ui_showDesMtx_btn.setEnabled(False)

            # Update self.desMtx0
            if self.desMtx_read is not None or self.max_scan_length > 0:
                self.setup_regressor_template(
                        self.desMtx_read, max_scan_length=self.max_scan_length,
                        col_names_read=self.col_names_read)

            return

        elif attr == 'showDesMtx':
            self.plt_win = MatplotlibWindow()
            self.plt_win.setWindowTitle('Design matrix')
            ax = self.plt_win.canvas.figure.subplots(1, 1)
            ax.matshow(self.desMtx_read, cmap='gray')
            ax.set_aspect('auto')

            self.plt_win.show()
            self.plt_win.canvas.draw()
            self.plt_win.canvas.start_event_loop(0.005)

            return

        elif attr == 'max_scan_length':
            if self.desMtx_read is not None and \
                    self.desMtx_read.shape[0] > val:
                val = self.desMtx_read.shape[0]
                if reset_fn:
                    reset_fn(val)

            # Update self.desMtx0
            self.setup_regressor_template(self.desMtx_read,
                                          max_scan_length=val,
                                          col_names_read=self.col_names_read)

            if reset_fn is None:
                if hasattr(self, 'ui_maxLen_spBx'):
                    self.ui_maxLen_spBx.setValue(val)

        elif attr == 'mot_reg':
            if val == 'none':
                val = 'None'
            elif val.startswith('6 motions'):
                val = 'mot6'
            elif val.startswith('12 motions'):
                val = 'mot12'
            elif val.startswith('6 motion derivatives'):
                val = 'dmot6'

            # Update wait_num
            self.mot_reg = val
            if hasattr(self, 'ui_waitNum_cmbBx'):
                if 'regressor' in self.ui_waitNum_cmbBx.currentText():
                    self.set_wait_num(0)
            else:
                self.set_wait_num(self.wait_num)

            if reset_fn is None and hasattr(self, 'ui_motReg_cmbBx'):
                if val == 'None':
                    self.ui_motReg_cmbBx.setCurrentIndex(0)
                elif val == 'mot6':
                    self.ui_motReg_cmbBx.setCurrentIndex(1)
                elif val == 'mot12':
                    self.ui_motReg_cmbBx.setCurrentIndex(2)
                elif val == 'dmot6':
                    self.ui_motReg_cmbBx.setCurrentIndex(3)

            return

        elif attr == 'phys_reg':
            if val == 'none':
                val = 'None'
            elif val.startswith('8 RICOR'):
                val = 'RICOR8'
            elif val.startswith('5 RVT'):
                val = 'RVT5'
            elif val.startswith('13 RVT+RICOR'):
                val = 'RVT+RICOR13'

            # Update wait_num
            self.phys_reg = val
            if hasattr(self, 'ui_waitNum_cmbBx'):
                if 'regressor' in self.ui_waitNum_cmbBx.currentText():
                    self.set_wait_num(0)
            else:
                self.set_wait_num(self.wait_num)

            if reset_fn is None and hasattr(self, 'ui_physReg_cmbBx'):
                if val == 'None':
                    self.ui_physReg_cmbBx.setCurrentIndex(0)
                elif val == 'RICOR8':
                    self.ui_physReg_cmbBx.setCurrentIndex(1)
                elif val == 'RVT5':
                    self.ui_physReg_cmbBx.setCurrentIndex(2)
                elif val == 'RVT+RICOR13':
                    self.ui_physReg_cmbBx.setCurrentIndex(3)

            return

        elif attr == 'GS_reg':
            setattr(self, attr, val)
            if hasattr(self, 'ui_GS_reg_chb'):
                self.ui_GS_reg_chb.setChecked(self.GS_reg)

            # Update wait_num
            if hasattr(self, 'ui_waitNum_cmbBx'):
                if 'regressor' in self.ui_waitNum_cmbBx.currentText():
                    self.set_wait_num(0)
            else:
                self.set_wait_num(self.wait_num)

        elif attr == 'GS_mask':
            if reset_fn is not None:
                if Path(val).is_dir():
                    startdir = val
                else:
                    startdir = self.watch_dir

                dlgMdg = "REGRESS: Select global signal mask"
                fname = self.select_file_dlg(dlgMdg, startdir,
                                             "*.BRIK* *.nii*")
                if fname[0] == '':
                    return -1

                val = fname[0]
                if reset_fn:
                    reset_fn(str(val))
            else:
                if not Path(val).is_file():
                    val = ''

                if hasattr(self, f"ui_{attr}_lnEd"):
                    obj = getattr(self, f"ui_{attr}_lnEd")
                    obj.setText(str(val))

        elif attr == 'WM_reg':
            setattr(self, attr, val)
            if hasattr(self, 'ui_WM_reg_chb'):
                self.ui_WM_reg_chb.setChecked(self.WM_reg)

            # Update wait_num
            if hasattr(self, 'ui_waitNum_cmbBx'):
                if 'regressor' in self.ui_waitNum_cmbBx.currentText():
                    self.set_wait_num(0)
            else:
                self.set_wait_num(self.wait_num)

        elif attr == 'WM_mask':
            if reset_fn is not None:
                if Path(val).is_dir():
                    startdir = val
                else:
                    startdir = self.watch_dir

                dlgMdg = "REGRESS: Select white matter mask"
                fname = self.select_file_dlg(dlgMdg, startdir,
                                             "*.BRIK* *.nii*")
                if fname[0] == '':
                    return -1

                val = fname[0]
                if reset_fn:
                    reset_fn(str(val))
            else:
                if not Path(val).is_file():
                    val = ''

                if hasattr(self, f"ui_{attr}_lnEd"):
                    obj = getattr(self, f"ui_{attr}_lnEd")
                    obj.setText(str(val))

        elif attr == 'Vent_reg':
            setattr(self, attr, val)
            if hasattr(self, 'ui_Vent_reg_chb'):
                self.ui_Vent_reg_chb.setChecked(self.Vent_reg)

            # Update wait_num
            if hasattr(self, 'ui_waitNum_cmbBx'):
                if 'regressor' in self.ui_waitNum_cmbBx.currentText():
                    self.set_wait_num(0)
            else:
                self.set_wait_num(self.wait_num)

        elif attr == 'Vent_mask':
            if reset_fn is not None:
                if Path(val).is_dir():
                    startdir = val
                else:
                    startdir = self.watch_dir

                dlgMdg = "REGRESS: Select ventricle mask"
                fname = self.select_file_dlg(dlgMdg, startdir,
                                             "*.BRIK* *.nii*")
                if fname[0] == '':
                    return -1

                val = fname[0]
                if reset_fn:
                    reset_fn(str(val))
            else:
                if not Path(val).is_file():
                    val = ''

                if hasattr(self, f"ui_{attr}_lnEd"):
                    obj = getattr(self, f"ui_{attr}_lnEd")
                    obj.setText(str(val))

        elif attr == 'max_poly_order':
            if val == 'auto':
                if hasattr(self, 'ui_maxPoly_lb'):
                    self.ui_maxPoly_lb.setText('Increase polynomial order ' +
                                               'according to the length')
                val = np.inf
            elif val == 'set':
                num, okflag = QtWidgets.QInputDialog.getInt(
                        None, "Maximum polynomial order", "enter value")
                if not okflag:
                    if np.isinf(self.max_poly_order):
                        if reset_fn:
                            reset_fn(0)
                    return

                if hasattr(self, 'ui_maxPoly_lb'):
                    self.ui_maxPoly_lb.setText('Increase polynomial order ' +
                                               'according to the length' +
                                               f' up to {num}')
                val = num
            elif np.isinf(val) and reset_fn is None:
                if hasattr(self, 'ui_maxPoly_cmbBx'):
                    self.ui_maxPoly_cmbBx.setCurrentIndex(0)
                    self.ui_maxPoly_lb.setText('Increase polynomial order ' +
                                               'according to the length')
            elif type(val) == int and reset_fn is None:
                if hasattr(self, 'ui_maxPoly_cmbBx'):
                    self.ui_maxPoly_cmbBx.setCurrentIndex(1)
                    self.ui_maxPoly_lb.setText('Increase polynomial order ' +
                                               'according to the length' +
                                               f' up to {val}')

        elif attr == 'mask_file':
            if type(val) == int and val == 0:
                if hasattr(self, 'ui_mask_lb'):
                    self.ui_mask_lb.setText('zero-out initial received volume')
                self.mask_byte = None

            elif 'internal' in val:
                val = 0
                if hasattr(self, 'ui_mask_lb'):
                    self.ui_mask_lb.setText('zero-out initial received volume')
                self.mask_byte = None

            elif 'external' in val:
                fname = self.select_file_dlg('REGRESS: Selct mask volume',
                                             self.watch_dir, "*.BRIK* *.nii*")
                if fname[0] == '':
                    if reset_fn:
                        reset_fn(1)
                    return -1

                mask_img = nib.load(fname[0])
                mask_fname = fname[0]
                if len(mask_img.shape) > 3 and mask_img.shape[3] > 1:
                    num, okflag = QtWidgets.QInputDialog.getInt(
                            None, "Select sub volume",
                            "sub-volume index (0 is first)", 0, 0,
                            mask_img.shape[3])
                    if fname[0] == '':
                        if reset_fn:
                            reset_fn(1)
                        return

                    mask_fname += f"[{num}]"
                else:
                    num = 0

                self.set_mask(fname[0], num)
                if hasattr(self, 'ui_mask_lb'):
                    self.ui_mask_lb.setText(str(mask_fname))
                val = mask_fname

            elif type(val) == str:
                ma = re.search(r"\[(\d+)\]", val)
                if ma:
                    num = int(ma.groups()[0])
                    fname = re.sub(r"\[(\d+)\]", '', val)
                else:
                    fname = val
                    num = 0

                if not Path(fname).is_file():
                    return

                if reset_fn is None and hasattr(self, 'ui_mask_cmbBx'):
                    self.ui_mask_cmbBx.setCurrentIndex(0)

                self.set_mask(fname, num)
                if hasattr(self, 'ui_mask_lb'):
                    self.ui_mask_lb.setText(str(val))

        elif attr == 'save_proc':
            if hasattr(self, 'ui_saveProc_chb'):
                self.ui_saveProc_chb.setChecked(val)

        elif attr == '_verb':
            if hasattr(self, 'ui_verb_chb'):
                self.ui_verb_chb.setChecked(val)

        elif reset_fn is None:
            return

        # -- Set value --
        setattr(self, attr, val)
        if echo:
            print(f"{self.__class__.__name__}." + attr, '=',
                  getattr(self, attr))

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def ui_set_param(self):
        """
        When reset_fn is None, set_param is considered to be called from
        load_parameters function.
        """

        ui_rows = []
        self.ui_objs = []

        # enabled
        self.ui_enabled_rdb = QtWidgets.QRadioButton("Enable")
        self.ui_enabled_rdb.setChecked(self.enabled)
        self.ui_enabled_rdb.toggled.connect(
                lambda checked: self.set_param('enabled', checked,
                                               self.ui_enabled_rdb.setChecked))
        ui_rows.append((self.ui_enabled_rdb, None))

        # wait_num
        var_lb = QtWidgets.QLabel("Wait REGRESS until (volumes) :")
        self.ui_waitNum_cmbBx = QtWidgets.QComboBox()
        self.ui_waitNum_cmbBx.addItems(['number of regressors', 'set value'])
        self.ui_waitNum_cmbBx.activated.connect(
                lambda idx:
                self.set_param('wait_num',
                               self.ui_waitNum_cmbBx.currentText(),
                               self.ui_waitNum_cmbBx.setCurrentIndex))
        ui_rows.append((var_lb, self.ui_waitNum_cmbBx))

        self.ui_waitNum_lb = QtWidgets.QLabel()
        regNum = self.get_reg_num()
        self.ui_waitNum_lb.setText(
                f'Wait REGRESS until receiving {regNum} volumes')
        ui_rows.append((None, self.ui_waitNum_lb))
        self.ui_objs.extend([var_lb, self.ui_waitNum_cmbBx,
                             self.ui_waitNum_lb])
        ui_rows.append((None, self.ui_waitNum_lb))

        # desMtx
        var_lb = QtWidgets.QLabel("Design matrix :")

        desMtx_hBLayout = QtWidgets.QHBoxLayout()
        self.ui_loadDesMtx_btn = QtWidgets.QPushButton('Set')
        self.ui_loadDesMtx_btn.clicked.connect(
                lambda: self.set_param('desMtx_f', 'set'))
        desMtx_hBLayout.addWidget(self.ui_loadDesMtx_btn)

        self.ui_showDesMtx_btn = QtWidgets.QPushButton()
        self.ui_showDesMtx_btn.clicked.connect(
                lambda: self.set_param('showDesMtx'))
        desMtx_hBLayout.addWidget(self.ui_showDesMtx_btn)

        self.ui_objs.extend([var_lb, self.ui_loadDesMtx_btn,
                             self.ui_showDesMtx_btn])
        ui_rows.append((var_lb, desMtx_hBLayout))
        self.ui_showDesMtx_btn.setText('Show desing matrix')
        if self.desMtx_read is None:
            self.ui_showDesMtx_btn.setEnabled(False)
        else:
            self.ui_showDesMtx_btn.setEnabled(True)

        # max_scan_length
        var_lb = QtWidgets.QLabel("Maximum scan length :")
        self.ui_maxLen_spBx = QtWidgets.QSpinBox()
        self.ui_maxLen_spBx.setMinimum(1)
        self.ui_maxLen_spBx.setMaximum(9999)
        self.ui_maxLen_spBx.setValue(self.max_scan_length)
        self.ui_maxLen_spBx.editingFinished.connect(
                lambda: self.set_param('max_scan_length',
                                       self.ui_maxLen_spBx.value(),
                                       self.ui_maxLen_spBx.setValue))
        ui_rows.append((var_lb, self.ui_maxLen_spBx))
        self.ui_objs.extend([var_lb, self.ui_maxLen_spBx])

        # mot_reg
        var_lb = QtWidgets.QLabel("Motion regressor :")
        self.ui_motReg_cmbBx = QtWidgets.QComboBox()
        self.ui_motReg_cmbBx.addItems(
                ['none', '6 motions (yaw, pitch, roll, dS, dL, dP)',
                 '12 motions (6 motions and their temporal derivatives)',
                 '6 motion derivatives'])
        ci = {'None': 0, 'mot6': 1, 'mot12': 2, 'dmot6': 3}[self.mot_reg]
        self.ui_motReg_cmbBx.setCurrentIndex(ci)
        self.ui_motReg_cmbBx.currentIndexChanged.connect(
                lambda idx:
                self.set_param('mot_reg',
                               self.ui_motReg_cmbBx.currentText(),
                               self.ui_motReg_cmbBx.setCurrentIndex))
        ui_rows.append((var_lb, self.ui_motReg_cmbBx))
        self.ui_objs.extend([var_lb, self.ui_motReg_cmbBx])

        # phys_reg
        var_lb = QtWidgets.QLabel("RICOR/RVT regressor :")
        self.ui_physReg_cmbBx = QtWidgets.QComboBox()
        self.ui_physReg_cmbBx.addItems(
                ['none', '8 RICOR (4 Resp and 4 Card)',
                 '5 RVT [not recommended]',
                 '13 RVT+RICOR (5 RVT [not recommended]], 4 Resp, and 4 Card)']
                )
        ci = {'None': 0, 'RVT5': 1, 'RICOR8': 2,
              'RVT+RICOR13': 3}[self.phys_reg]
        self.ui_physReg_cmbBx.setCurrentIndex(ci)
        self.ui_physReg_cmbBx.currentIndexChanged.connect(
                lambda idx:
                self.set_param('phys_reg',
                               self.ui_physReg_cmbBx.currentText(),
                               self.ui_physReg_cmbBx.setCurrentIndex))
        ui_rows.append((var_lb, self.ui_physReg_cmbBx))
        self.ui_objs.extend([var_lb, self.ui_physReg_cmbBx])

        # GS ROI regressor
        self.ui_GS_reg_chb = QtWidgets.QCheckBox("Regress global signal :")
        self.ui_GS_reg_chb.setChecked(self.GS_reg)
        self.ui_GS_reg_chb.stateChanged.connect(
                lambda state: self.set_param('GS_reg', state > 0))

        GSmask_hBLayout = QtWidgets.QHBoxLayout()
        self.ui_GS_mask_lnEd = QtWidgets.QLineEdit()
        self.ui_GS_mask_lnEd.setText(str(self.GS_mask))
        self.ui_GS_mask_lnEd.setReadOnly(True)
        GSmask_hBLayout.addWidget(self.ui_GS_mask_lnEd)

        self.ui_GSmask_btn = QtWidgets.QPushButton('Set')
        self.ui_GSmask_btn.clicked.connect(
                lambda: self.set_param(
                        'GS_mask',
                        Path(self.ui_GS_mask_lnEd.text()).parent,
                        self.ui_GS_mask_lnEd.setText))
        GSmask_hBLayout.addWidget(self.ui_GSmask_btn)

        self.ui_objs.extend([self.ui_GS_reg_chb, self.ui_GS_mask_lnEd,
                             self.ui_GSmask_btn])
        ui_rows.append((self.ui_GS_reg_chb, GSmask_hBLayout))

        # WM ROI regressor
        self.ui_WM_reg_chb = QtWidgets.QCheckBox("Regress WM signal :")
        self.ui_WM_reg_chb.setChecked(self.WM_reg)
        self.ui_WM_reg_chb.stateChanged.connect(
                lambda state: self.set_param('WM_reg', state > 0))

        WMmask_hBLayout = QtWidgets.QHBoxLayout()
        self.ui_WM_mask_lnEd = QtWidgets.QLineEdit()
        self.ui_WM_mask_lnEd.setText(str(self.WM_mask))
        self.ui_WM_mask_lnEd.setReadOnly(True)
        WMmask_hBLayout.addWidget(self.ui_WM_mask_lnEd)

        self.ui_WMmask_btn = QtWidgets.QPushButton('Set')
        self.ui_WMmask_btn.clicked.connect(
                lambda: self.set_param(
                        'WM_mask',
                        Path(self.ui_WM_mask_lnEd.text()).parent,
                        self.ui_WM_mask_lnEd.setText))
        WMmask_hBLayout.addWidget(self.ui_WMmask_btn)

        self.ui_objs.extend([self.ui_WM_reg_chb, self.ui_WM_mask_lnEd,
                             self.ui_WMmask_btn])
        ui_rows.append((self.ui_WM_reg_chb, WMmask_hBLayout))

        # Vent ROI regressor
        self.ui_Vent_reg_chb = QtWidgets.QCheckBox("Regress Vent signal :")
        self.ui_Vent_reg_chb.setChecked(self.Vent_reg)
        self.ui_Vent_reg_chb.stateChanged.connect(
                lambda state: self.set_param('Vent_reg', state > 0))

        Ventmask_hBLayout = QtWidgets.QHBoxLayout()

        self.ui_Vent_mask_lnEd = QtWidgets.QLineEdit()
        self.ui_Vent_mask_lnEd.setText(str(self.Vent_mask))
        self.ui_Vent_mask_lnEd.setReadOnly(True)
        Ventmask_hBLayout.addWidget(self.ui_Vent_mask_lnEd)

        self.ui_Ventmask_btn = QtWidgets.QPushButton('Set')
        self.ui_Ventmask_btn.clicked.connect(
                lambda: self.set_param(
                        'Vent_mask',
                        Path(self.ui_Vent_mask_lnEd.text()).parent,
                        self.ui_Vent_mask_lnEd.setText))
        Ventmask_hBLayout.addWidget(self.ui_Ventmask_btn)

        self.ui_objs.extend([self.ui_Vent_reg_chb, self.ui_Vent_mask_lnEd,
                             self.ui_Ventmask_btn])
        ui_rows.append((self.ui_Vent_reg_chb, Ventmask_hBLayout))

        # max_poly_order
        var_lb = QtWidgets.QLabel("Maximum polynomial order :\n"
                                  "regressors for slow fluctuation")
        self.ui_maxPoly_cmbBx = QtWidgets.QComboBox()
        self.ui_maxPoly_cmbBx.addItems(['auto', 'set'])
        self.ui_maxPoly_cmbBx.activated.connect(
                lambda idx:
                self.set_param('max_poly_order',
                               self.ui_maxPoly_cmbBx.currentText(),
                               self.ui_maxPoly_cmbBx.setCurrentIndex))
        ui_rows.append((var_lb, self.ui_maxPoly_cmbBx))

        self.ui_maxPoly_lb = QtWidgets.QLabel()
        ui_rows.append((None, self.ui_maxPoly_lb))
        self.ui_objs.extend([var_lb, self.ui_maxPoly_cmbBx,
                             self.ui_maxPoly_lb])
        if np.isinf(self.max_poly_order):
            self.ui_maxPoly_cmbBx.setCurrentIndex(0)
            self.ui_maxPoly_lb.setText('Increase polynomial order ' +
                                       'according to the length')
        else:
            self.ui_maxPoly_cmbBx.setCurrentIndex(1)
            self.ui_maxPoly_lb.setText('Increase polynomial order ' +
                                       'according to the length' +
                                       f' up to {self.max_poly_order}')

        # mask_file
        var_lb = QtWidgets.QLabel("Mask :")
        self.ui_mask_cmbBx = QtWidgets.QComboBox()
        self.ui_mask_cmbBx.addItems(['external file',
                                     'initial volume of internal run'])
        self.ui_mask_cmbBx.activated.connect(
                lambda idx:
                self.set_param('mask_file',
                               self.ui_mask_cmbBx.currentText(),
                               self.ui_mask_cmbBx.setCurrentIndex))
        ui_rows.append((var_lb, self.ui_mask_cmbBx))

        self.ui_mask_lb = QtWidgets.QLabel()
        ui_rows.append((None, self.ui_mask_lb))
        self.ui_objs.extend([var_lb, self.ui_mask_cmbBx,
                             self.ui_mask_lb])

        if self.mask_file == 0:
            self.ui_mask_cmbBx.setCurrentIndex(1)
            self.ui_mask_lb.setText('zero-out initial received volume')
        else:
            self.ui_mask_cmbBx.setCurrentIndex(0)
            self.ui_mask_lb.setText(str(self.mask_file))

        # Save
        self.ui_saveProc_chb = QtWidgets.QCheckBox("Save processed image")
        self.ui_saveProc_chb.setChecked(self.save_proc)
        self.ui_saveProc_chb.stateChanged.connect(
                lambda state: setattr(self, 'save_proc', state > 0))
        ui_rows.append((self.ui_saveProc_chb, None))
        self.ui_objs.append(self.ui_saveProc_chb)

        # verb
        self.ui_verb_chb = QtWidgets.QCheckBox("Verbose logging")
        self.ui_verb_chb.setChecked(self.verb)
        self.ui_verb_chb.stateChanged.connect(
                lambda state: setattr(self, 'verb', state > 0))
        ui_rows.append((self.ui_verb_chb, None))
        self.ui_objs.append(self.ui_verb_chb)

        return ui_rows

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def get_options(self):
        all_opts = super().get_options()
        excld_opts = ('save_dir', 'watch_dir', 'desMtx0', 'mot0', 'tshift',
                      'mask_byte', 'retrocols', 'volreg', 'YMtx', 'TR',
                      'desMtx', 'maskV', 'motcols', 'physio', 'col_names_read',
                      'Y_mean', 'Y_mean_mask', 'GS_maskdata', 'WM_maskdata',
                      'Vent_maskdata', 'GS_col', 'WM_col', 'Vent_col',
                      'desMtx_read')
        sel_opts = {}
        for k, v in all_opts.items():
            if k in excld_opts:
                continue
            if isinstance(v, Path):
                v = str(v)
            sel_opts[k] = v

        return sel_opts


# %% Thread class for debug to simulate watch_dog observer thread =============
class WatchThread(threading.Thread):
    """
    Thread class to simulate watch_dog observer thread
    """

    def __init__(self, img_data, affine, img_header):
        threading.Thread.__init__(self)
        self.img_data = img_data
        self.affine = affine
        self.img_header = img_header
        self.daemon = True

    def run(self):
        for iv in range(5, 100):  # img_data.shape[3]):
            st = time.time()
            mri_data = MRI_data(self.img_data[:, :, :, iv], self.affine,
                                self.img_header, f'temp{iv}')
            rtp_volreg.do_proc(mri_data, iv, st)

        rtp_volreg.reset()


# %% ==========================================================================
if __name__ == '__main__':
    if '__file__' not in locals():
        __file__ = './this.py'

    try:
        from .rtp_volreg import RTP_VOLREG
        from .rtp_smooth import RTP_SMOOTH
        from .rtp_retrots import RTP_RETROTS
        from .rtp_physio_dummy import RTP_PHYSIO_DUMMY
    except Exception:
        from rtpfmri.rtp_volreg import RTP_VOLREG
        from rtpfmri.rtp_smooth import RTP_SMOOTH
        from rtpfmri.rtp_retrots import RTP_RETROTS
        from rtpfmri.rtp_physio_dummy import RTP_PHYSIO_DUMMY

    app = QtWidgets.QApplication(sys.argv)  # to run volreg and physio plot

    multithread = False

    # --- Set the test data files ---------------------------------------------
    src_dir = Path(__file__).absolute().parent.parent / 'test_data'
    testdata_f = src_dir / 'func_epi.nii.gz'
    mask_data_f = src_dir / 'rtp_mask.nii.gz'
    ref_vol = src_dir / "vr_base.nii.gz"
    ecg_f = src_dir / 'ECG.1D'
    resp_f = src_dir / 'Resp.1D'

    # --- RTO_VOLREG ----------------------------------------------------------
    rtp_volreg = RTP_VOLREG()
    rtp_volreg.regmode = 2  # CUBIC
    rtp_volreg.verb = True

    # Set reference volume
    rtp_volreg.set_ref_vol(ref_vol)

    # --- RTO_SMOOTH ----------------------------------------------------------
    rtp_smooth = RTP_SMOOTH()
    rtp_smooth.set_mask(mask_data_f)

    # --- RTP_PHYSIO with RTP_RETROTS -----------------------------------------
    sample_freq = 40
    TR = 2.0

    rtp_retrots = RTP_RETROTS()
    rtp_physio = RTP_PHYSIO_DUMMY(ecg_f, resp_f, sample_freq, rtp_retrots)

    # --- RTP_REGRESS ---------------------------------------------------------
    rtp_regress = RTP_REGRESS()

    # Set regressors
    desMtx_f = src_dir / 'exp_seq_1_tech02_desMtx.csv'
    rtp_regress.set_param('desMtx_f', desMtx_f)

    rtp_regress.mot_reg = 'mot12'
    rtp_regress.volreg = rtp_volreg
    rtp_regress.phys_reg = 'None'  # 'RVT+RICOR13'
    rtp_regress.physio = rtp_physio
    # rtp_regress.GS_reg = True
    # rtp_regress.GS_mask = src_dir / "WM_on_vr_base.nii.gz"
    rtp_regress.WM_reg = True
    rtp_regress.WM_mask = src_dir / "WM_on_vr_base.nii.gz"
    rtp_regress.Vent_reg = True
    rtp_regress.Vent_mask = src_dir / "Vent_on_vr_base.nii.gz"
    rtp_regress.max_poly_order = np.inf
    rtp_regress.mask_src_proc = rtp_volreg

    # Set parameters
    rtp_regress.mask_file = 0
    rtp_regress.TR = 2.0
    rtp_regress.tshift = 0.0
    rtp_regress.onGPU = True
    rtp_regress.verb = True
    rtp_regress.save_proc = True

    # --- connect volreg -> smooth -> regress ---------------------------------
    rtp_volreg.next_proc = rtp_smooth
    rtp_smooth.next_proc = rtp_regress

    # Set design matrix: must be the original number of sequence
    # (wihtout discarding initial volumes)
    """
    desMtx = np.zeros((263, 2), dtype=np.float32)
    col_names = ['BlkHappy', 'BlkCount']
    desMtx[23:43, 0] = 1
    desMtx[43:63, 1] = 1
    desMtx[83:103, 0] = 1
    desMtx[103:123, 1] = 1
    desMtx[143:163, 0] = 1
    desMtx[163:183, 1] = 1
    desMtx[203:223, 0] = 1
    desMtx[223:243, 1] = 1
    """

    # --- Run test ------------------------------------------------------------
    # Load test data
    img = nib.load(testdata_f)
    img_data = np.squeeze(img.get_fdata()).astype(np.float32)
    affine = img.affine
    img_header = img.header

    # Start physio recording
    # rtp_physio.start_recording()

    if multithread:
        # simulation on another thread
        # rtp_physio.cmd('SCAN_START')
        wth = WatchThread(img_data, affine, img_header)
        wth.start()
        wth.join()
    else:
        # simulation in the same process
        for ii in range(1):  # run several times
            # Reset and Ready
            assert rtp_volreg.proc_ready

            for iv in range(3, 100):  # img_data.shape[3]):
                st = time.time()
                mri_data = MRI_data(img_data[:, :, :, iv], affine, img_header,
                                    f'temp{iv}')
                rtp_volreg.do_proc(mri_data, iv, st)

            rtp_volreg.reset()
            rtp_physio.reset()
