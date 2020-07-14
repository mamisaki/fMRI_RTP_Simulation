#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mmisaki@laureateinstitute.org
"""


# %% import ===================================================================
from pathlib import Path
import re
import time
import numpy as np
import nibabel as nib
import ctypes
import sys
from scipy.ndimage import gaussian_filter
from six import string_types
import traceback
from PyQt5 import QtWidgets

from .rtp_common import RTP, MRI_data

try:
    librtp_path = str(Path(__file__).absolute().parent / 'librtp.so')
except Exception:
    librtp_path = './librtp.so'


# %% RTP_SMOOTH class =========================================================
class RTP_SMOOTH(RTP):
    """
    Real-time online spatial smoothing
    """

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __init__(self, blur_fwhm=6.0, mask_file=0, **kwargs):
        """
        Parameters
        ----------
        blur_fwhm: float
            Gaussina FWHM of smoothing kernel
        mask_file: 0 or str
            filename of mask
            If mask_file == 0, zero_out initial volume is used as mask.
        next_proc: object
            Object for the next process. do_proc(mri_data) method, reset()
            method, and proc_ready property should be implemented.
        verb: bool
            verbose flag to print log message
        """

        super(RTP_SMOOTH, self).__init__(**kwargs)

        # Set instance parameters
        self.blur_fwhm = blur_fwhm
        self.mask_file = mask_file

        if isinstance(self.mask_file, string_types) or \
                isinstance(self.mask_file, Path):
            # Set byte mask
            self.set_mask(self.mask_file)
        else:
            self.mask_byte = None

        # Initialize C library function call
        self.setup_libfuncs()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    @property
    def proc_ready(self):
        self._proc_ready = True

        if self.next_proc:
            self._proc_ready &= self.next_proc.proc_ready

        return self._proc_ready

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def setup_libfuncs(self):
        """ Setup library function access """

        librtp = ctypes.cdll.LoadLibrary(librtp_path)

        # -- Define rtp_smooth --
        self.rtp_smooth = librtp.rtp_smooth
        self.rtp_smooth.argtypes = \
            [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
             ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_void_p,
             ctypes.c_float]

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_mask(self, maskdata, sub_i=0, method='zero_out'):
        if isinstance(maskdata, string_types) or isinstance(maskdata, Path):
            ma = re.search(r"\'*\[(\d+)\]\'*$", str(maskdata))
            if ma:
                sub_i = int(ma.groups()[0])
                maskdata = re.sub(r"\'*\[(\d+)\]\'*$", '', str(maskdata))

            if not Path(maskdata).is_file():
                self.errmsg(f"Not found mask file: {maskdata}")
                self.mask_file = 0
                return

            self.mask_file = str(maskdata)

            maskdata = np.squeeze(nib.load(maskdata).get_fdata())
            if maskdata.ndim > 3:
                maskdata = maskdata[:, :, :, sub_i]

            if self._verb:
                msg = f"Mask = {self.mask_file }"
                if ma:
                    msg += f"[{sub_i}]"
                self.logmsg(msg)

        if method == 'zero_out':
            self.maskV = maskdata != 0

        # Set byte mask
        self.mask_byte = self.maskV.astype('u2')
        self.mask_byte_p = self.mask_byte.ctypes.data_as(
                ctypes.POINTER(ctypes.c_ushort))

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def smooth(self, mri_data):

        """
        C function definition
        ---------------------
        int rtp_smooth(float *im, int nx, int ny, int nz, float dx, float dy,
                       float dz, unsigned short *mask, float fwhm)
        """

        """DEBUG
        import matplotlib.pyplot as plt
        self = rtp_smooth

        """

        # image dimension
        nx, ny, nz = mri_data.img_data.shape
        if hasattr(mri_data.img_header, 'info'):
            dx, dy, dz = np.abs(mri_data.img_header.info['DELTA'])
        elif hasattr(mri_data.img_header, 'get_zooms'):
            dx, dy, dz = mri_data.img_header.get_zooms()[:3]
        else:
            self.errmsg("No voxel size information in mri_data header")

        # Copy function image data and get pointer
        fim0 = mri_data.img_data
        if fim0.dtype != np.float32:
            fim_arr = fim0.astype(np.float32)
        else:
            fim_arr = fim0
        fim_arr = np.moveaxis(np.moveaxis(fim_arr, 0, -1), 0, 1).copy()
        fim_p = fim_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        self.rtp_smooth(fim_p, nx, ny, nz, dx, dy, dz, self.mask_byte_p,
                        self.blur_fwhm)
        fim_arr = np.moveaxis(np.moveaxis(fim_arr, 0, -1), 0, 1)

        return fim_arr

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def smooth_scipy(self, mri_data):
        """Apply Gaussian smoothing (within the mask).
        https://stackoverflow.com/questions/18697532/gaussian-filtering-a-image-with-nan-in-python/36307291#36307291
        """

        # image dimension
        if hasattr(mri_data.img_header, 'info'):
            dx, dy, dz = np.abs(mri_data.img_header.info['DELTA'])
        elif hasattr(mri_data.img_header, 'get_zooms'):
            dx, dy, dz = mri_data.img_header.get_zooms()[:3]
        else:
            self.errmsg("No voxel size information in mri_data header")

        # Set gaussian sigma in image dimension
        sigma = (self.blur_fwhm / np.array((dx, dy, dz))) / 2.354820
        imgdata = mri_data.img_data.astype(np.float64)

        # Apply mask
        if hasattr(self, 'maskV'):
            imgdata[~self.maskV] = 0

        # Apply Gaussian filter
        filt_img = gaussian_filter(imgdata, sigma, mode='constant')

        if hasattr(self, 'maskV'):
            # Adjust voxels with out of the mask (0) convolution
            aux_img = np.ones_like(imgdata)
            aux_img[~self.maskV] = 0
            filt_aux_img = gaussian_filter(aux_img, sigma, mode='constant')
            filt_img[self.maskV] /= filt_aux_img[self.maskV]

        return filt_img.astype(mri_data.img_data.dtype)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def do_proc(self, mri_data, vol_idx=None, pre_proc_time=None, **kwargs):
        try:
            # Increment the number of received volume
            self.vol_num += 1
            if vol_idx is None:
                vol_idx = self.vol_num

            # Unless the mask is set, set it by a received volume
            if self.mask_byte is None or not hasattr(self, 'maskV'):
                if self._verb:
                    msg = f"Mask is set by a received volume, index {vol_idx}"
                    self.logmsg(msg)

                self.set_mask(mri_data.img_data)

            # Perform smoothing
            mri_data.img_data = self.smooth(mri_data)

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
                msg = f'{vol_idx}, Smoothing is done for {f}'
                if pre_proc_time is not None:
                    msg += f' (process time {proc_delay:.4f}s)'
                msg += '.'

                self.logmsg(msg)

            # Set save_name
            mri_data.save_name = 'sm.' + str(mri_data.save_name)

            # Run the next process
            if self.next_proc:
                if self.vol_num == 0:
                    self.next_proc.do_proc(mri_data, vol_idx,
                                           self.proc_time[-1],
                                           maskV=self.maskV)
                else:
                    self.next_proc.do_proc(mri_data, vol_idx,
                                           self.proc_time[-1])

            # Save processed image
            if self.save_proc:
                self.keep_processed_image(mri_data)

        except Exception:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            errmsg = f'{exc_type}, {exc_tb.tb_frame.f_code.co_filename}:' + \
                f'{exc_tb.tb_lineno}'
            self.errmsg(errmsg, no_pop=True)
            traceback.print_exc(file=self._err_out)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def reset(self):
        """Reset process parameters
        """

        if self.verb:
            self.logmsg(f"Reset {self.__class__.__name__} module.")

        if type(self.mask_file) == int and self.mask_file == 0:
            self.mask_byte = None

        super(RTP_SMOOTH, self).reset()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_param(self, attr, val=None, reset_fn=None, echo=False):
        """
        When reset_fn is None, set_param is considered to be called from
        load_parameters function.
        """

        # -- check value --
        if attr == 'enabled':
            if hasattr(self, 'ui_enabled_rdb'):
                self.ui_enabled_rdb.setChecked(val)

            if hasattr(self, 'ui_objs'):
                for ui in self.ui_objs:
                    ui.setEnabled(val)

        elif attr in ('watch_dir', 'save_dir'):
            pass

        if attr == 'blur_fwhm' and reset_fn is None:
            if hasattr(self, 'ui_FWHM_dSpBx'):
                self.ui_FWHM_dSpBx.setValue(val)

        elif attr == 'mask_file':
            if isinstance(val, Path):
                val = str(Path)

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
                fname = self.select_file_dlg('SMOOTH: Selct mask volume',
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
                        return -1

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
                    # set 'external'
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

        return 0

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def ui_set_param(self):

        ui_rows = []
        self.ui_objs = []

        # enabled
        self.ui_enabled_rdb = QtWidgets.QRadioButton("Enable")
        self.ui_enabled_rdb.setChecked(self.enabled)
        self.ui_enabled_rdb.toggled.connect(
                lambda checked: self.set_param('enabled', checked,
                                               self.ui_enabled_rdb.setChecked))
        ui_rows.append((self.ui_enabled_rdb, None))

        # blur_fwhm
        var_lb = QtWidgets.QLabel("Gaussian FWHM :")
        self.ui_FWHM_dSpBx = QtWidgets.QDoubleSpinBox()
        self.ui_FWHM_dSpBx.setMinimum(0.0)
        self.ui_FWHM_dSpBx.setSingleStep(1.0)
        self.ui_FWHM_dSpBx.setDecimals(2)
        self.ui_FWHM_dSpBx.setSuffix(" mm")
        self.ui_FWHM_dSpBx.setValue(self.blur_fwhm)
        self.ui_FWHM_dSpBx.valueChanged.connect(
                lambda x: self.set_param('blur_fwhm', x,
                                         self.ui_FWHM_dSpBx.setValue))
        ui_rows.append((var_lb, self.ui_FWHM_dSpBx))
        self.ui_objs.extend([var_lb, self.ui_FWHM_dSpBx])

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

        if type(self.mask_file) == int and self.mask_file == 0:
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
        excld_opts = ('watch_dir', 'save_dir', 'mask_byte', 'maskV')
        sel_opts = {}
        for k, v in all_opts.items():
            if k in excld_opts:
                continue
            if isinstance(v, Path):
                v = str(v)
            sel_opts[k] = v

        return sel_opts


# %% main (for test) ==========================================================
if __name__ == '__main__':

    if '__file__' not in locals():
        __file__ = './this.py'

    # Test
    rtp_smooth = RTP_SMOOTH()
    rtp_smooth.verb = True

    # Set mask data
    src_dir = Path(__file__).absolute().parent.parent / 'test_data'
    maskname = src_dir / 'rtp_mask.nii.gz'
    rtp_smooth.set_mask(maskname)

    # Get test data
    fname = src_dir / 'func_epi.nii.gz'
    img = nib.load(fname)
    img_data = np.squeeze(img.get_fdata()).astype(np.float32)
    affine = img.affine
    img_header = img.header

    img_data_ts = np.ndarray(img_data.shape, dtype=np.float32)
    ets = []

    for iv in range(img_data.shape[-1]):
        st = time.time()
        mri_data = MRI_data(img_data[:, :, :, iv], affine, img_header,
                            f'temp{iv}')
        rtp_smooth.do_proc(mri_data, iv, st)
        ets.append(time.time() - st)
