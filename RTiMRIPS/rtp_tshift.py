#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real-time processing class of slice-timeing correction

@author: mmisaki@laureateinstitute.org
"""


# %% import ===================================================================
from pathlib import Path
import sys
import time
import numpy as np
import nibabel as nib
import copy
import traceback
from six import string_types
from PyQt5 import QtWidgets

from .rtp_common import RTP, MRI_data


# %% class RTP_TSHIFT =========================================================
class RTP_TSHIFT(RTP):
    """
    Real-time online slice-timing correction
    """

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __init__(self, ignore_init=3, method='cubic', ref_time=0, TR=2.0,
                 slice_timing=[], slice_dim=2, **kwargs):
        """
        TR and slice_timing can be set from a sample fMRI data using the
        'slice_timing_from_sample' method

        Parameters
        ----------
        ignore_init: int
            Number of volumes to ignore before staring a process
        method: str, ['linear'|'cubic']
            temporal interpolation method for the correction
        ref_time: float, optional
            reference time to shift slice timing
        TR: float, optional
            scan interval in second
        slice_timing: array, optional
            timing of each slices
        slice_dim: int, optional
            silce dimension
                0: x axis (sagital slice)
                1: y axis (coronal slice)
                2: z axis (axial slice)
        """

        super(RTP_TSHIFT, self).__init__(**kwargs)

        # Set instance parameters
        self.ignore_init = ignore_init
        self.method = method
        self.ref_time = ref_time

        self.TR = TR
        self.slice_timing = slice_timing
        self.slice_dim = slice_dim

        # Init varoiables for interpolation
        self.prep_weight = False  # Flag if precalculation has done
        self.pre_data = []  # previous data volumes

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    @property
    def proc_ready(self):
        if self.slice_timing is None or self.TR is None:
            errmsg = "slice timing is not set. "
            errmsg += "It will be set by the first image received."
            self.errmsg(errmsg)

        self._proc_ready = True  # ready in any case
        if self.next_proc:
            self._proc_ready &= self.next_proc.proc_ready

        return self._proc_ready

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def slice_timing_from_sample(self, mri_data):
        """
        Set slice timing from a sample mri_data

        Parameters
        ----------
        mri_data: string_types or MRI_data object
            If mri_data is string_types, it should be a BRIK filename of sample
            data.
        """

        if isinstance(mri_data, string_types) or isinstance(mri_data, Path):
            img = nib.load(mri_data)
            vol_shape = img.shape[:3]
            header = img.header
            fname = mri_data
        else:
            vol_shape = mri_data.img_data.shape[:3]
            fname = mri_data.save_name
            header = mri_data.header

        if hasattr(header, 'get_slice_times'):
            self.slice_timing = header.get_slice_times()
        elif hasattr(header, 'info') and 'TAXIS_FLOATS' in header.info:
            self.slice_timing = header.info['TAXIS_OFFSETS']
        else:
            self.errmsg(f'{fname} has no slice timing info.')
            return

        if self._verb:
            msg = 'Slice timing = {}.'.format(self.slice_timing)
            self.logmsg(msg)

        # set interpolation weight
        self._pre_culc_weight(vol_shape)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _pre_culc_weight(self, vol_shape):
        """
        Pre-calculation of weights for temporal interpolation

        Set variables
        -------------
        Interpolation weights
        self.wm1: weigth for minus 1 TR data
        self.w0: weigth for current TR data

        Only for cubic interpolation
        self.wm2: weigth for minus 2 TR data
        self.wp1: weigth for plus 1 TR data
                  used by cubic interpolation assuming the next data is the
                  same as current data

        Retrospective extrapolation weights
        self.r1wm1: 1-TR retrospective weigth for minus 1 TR data
        self.r1w0: 1-TR retrospective weigth for current TR data

        Only for cubic interpolation
        self.r1wm2: 1-TR retrospective weigth for minus 2 TR data
        self.r1wp1: 1-TR retrospective weigth for plus 1 TR data

        self.r2wm1: 2-TR retrospective weigth for minus 1 TR data
        self.r2w0: 2-TR retrospective weigth for current TR data
        self.r2wm2: 2-TR retrospective weigth for minus 2 TR data
        self.r2wp1: 2-TR retrospective weigth for plus 1 TR data
        """

        if self.slice_timing is None or self.TR is None:
            self.errmsg("slice timing is not set.")
            return

        if self.method not in ['linear', 'cubic']:
            self.errmsg("{} is not supported.".format(self.method))
            return

        # Set reference time
        ref_time = self.ref_time

        # slice timing shift from ref_time (relative to TR)
        shf = [(slt - ref_time)/self.TR for slt in self.slice_timing]

        # Initialize masked weight
        self.wm1 = np.ones(vol_shape, dtype=np.float32)  # weight for t-1
        self.w0 = np.ones(vol_shape, dtype=np.float32)  # weight for t
        # retrospective weight
        self.r1wm1 = np.ones(vol_shape, dtype=np.float32)  # weight for t-1
        self.r1w0 = np.ones(vol_shape, dtype=np.float32)  # weight for t

        if self.method == 'cubic':
            self.wm2 = np.ones(vol_shape, dtype=np.float32)  # weight for t-2
            self.wp1 = np.ones(vol_shape, dtype=np.float32)  # weight for t+1
            # retrospective weight
            self.r1wm2 = np.ones(vol_shape, dtype=np.float32)  # weight for t-1
            self.r1wp1 = np.ones(vol_shape, dtype=np.float32)  # weight for t

        # Set weight
        for sli in range(len(self.slice_timing)):
            if self.method == 'linear':
                wm1 = shf[sli]  # <=> x1
                w0 = 1.0 - wm1  # <=> -x0
                if self.slice_dim == 0:
                    self.wm1[sli, :, :] *= wm1
                    self.w0[sli, :, :] *= w0
                elif self.slice_dim == 1:
                    self.wm1[:, sli, :] *= wm1
                    self.w0[:, sli, :] *= w0
                elif self.slice_dim == 2:
                    self.wm1[:, :, sli] *= wm1
                    self.w0[:, :, sli] *= w0

                # Retrospective weight
                r1wm1 = shf[sli] + 1.0  # <=> x1 - (-1)
                r1w0 = -shf[sli]  # <=> -X1
                if self.slice_dim == 0:
                    self.r1wm1[sli, :, :] *= r1wm1
                    self.r1w0[sli, :, :] *= r1w0
                elif self.slice_dim == 1:
                    self.r1wm1[:, sli, :] *= r1wm1
                    self.r1w0[:, sli, :] *= r1w0
                elif self.slice_dim == 2:
                    self.r1wm1[:, :, sli] *= r1wm1
                    self.r1w0[:, :, sli] *= r1w0

            elif self.method == 'cubic':
                aa = 1.0 - shf[sli]
                wm2 = aa * (1.0-aa) * (aa-2.0) * 0.1666667
                wm1 = (aa+1.0) * (aa-1.0) * (aa-2.0) * 0.5
                w0 = aa * (aa+1.0) * (2.0-aa) * 0.5
                wp1 = aa * (aa+1.0) * (aa-1.0) * 0.1666667
                if self.slice_dim == 0:
                    self.wm2[sli, :, :] *= wm2
                    self.wm1[sli, :, :] *= wm1
                    self.w0[sli, :, :] *= w0
                    self.wp1[sli, :, :] *= wp1
                elif self.slice_dim == 1:
                    self.wm2[:, sli, :] *= wm2
                    self.wm1[:, sli, :] *= wm1
                    self.w0[:, sli, :] *= w0
                    self.wp1[:, sli, :] *= wp1
                elif self.slice_dim == 2:
                    self.wm2[:, :, sli] *= wm2
                    self.wm1[:, :, sli] *= wm1
                    self.w0[:, :, sli] *= w0
                    self.wp1[:, :, sli] *= wp1

                # 1-TR Retrospective weight
                aa = 1.0 - shf[sli] - 1.0
                r1wm2 = aa * (1.0-aa) * (aa-2.0) * 0.1666667
                r1wm1 = (aa+1.0) * (aa-1.0) * (aa-2.0) * 0.5
                r1w0 = aa * (aa+1.0) * (2.0-aa) * 0.5
                r1wp1 = aa * (aa+1.0) * (aa-1.0) * 0.1666667
                if self.slice_dim == 0:
                    self.r1wm2[sli, :, :] *= r1wm2
                    self.r1wm1[sli, :, :] *= r1wm1
                    self.r1w0[sli, :, :] *= r1w0
                    self.r1wp1[sli, :, :] *= r1wp1
                elif self.slice_dim == 1:
                    self.r1wm2[:, sli, :] *= r1wm2
                    self.r1wm1[:, sli, :] *= r1wm1
                    self.r1w0[:, sli, :] *= r1w0
                    self.r1wp1[:, sli, :] *= r1wp1
                elif self.slice_dim == 2:
                    self.r1wm2[:, :, sli] *= r1wm2
                    self.r1wm1[:, :, sli] *= r1wm1
                    self.r1w0[:, :, sli] *= r1w0
                    self.r1wp1[:, :, sli] *= r1wp1

        self.prep_weight = True

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def do_proc(self, mri_data, vol_idx=None, pre_proc_time=None, **kwargs):
        try:
            # Increment the number of received volume: vol_num is 0-based
            self.vol_num += 1
            if self.vol_num < self.ignore_init:
                return

            if vol_idx is None:
                vol_idx = self.vol_num

            # --- Initialization ----------------------------------------------
            # Check if slice_timing is set. Unless it is set by the current
            # data
            if self.slice_timing is None:
                self.slice_timing_from_sample(mri_data)

            # Get the volume data
            dataV = mri_data.img_data.astype(np.float32)
            if dataV.shape[self.slice_dim] != len(self.slice_timing):
                self.errmsg('Slice timing array mismathces to data.',
                            no_pop=True)
                return

            # Check if interpolation weights has been culculated. Unless
            # calculate it here.
            if not self.prep_weight:
                self._pre_culc_weight(mri_data.img_data.shape)

            # If there is no previous data return
            if len(self.pre_data) == 0:
                self.pre_mri_data = copy.deepcopy(mri_data)
                self.pre_data.append(dataV)
                return

            # --- Retrospective correction ------------------------------------
            if hasattr(self, 'pre_mri_data'):
                if self.method == 'linear':
                    retro_shft_dataV = \
                        self.r1wm1 * self.pre_data[-1] + self.r1w0 * dataV

                elif self.method == 'cubic':
                    if len(self.pre_data) == 1:
                        retro_shft_dataV = \
                            self.r1wm2 * self.pre_data[-1] + \
                            self.r1wm1 * self.pre_data[-1] + \
                            self.r1w0 * dataV + \
                            self.r1wp1 * dataV
                    elif len(self.pre_data) == 2:
                        retro_shft_dataV = \
                            self.r1wm2 * self.pre_data[-2] + \
                            self.r1wm1 * self.pre_data[-1] + \
                            self.r1w0 * dataV + \
                            self.r1wp1 * dataV

                pre_mri_data = self.pre_mri_data
                pre_mri_data.img_data = retro_shft_dataV

                # log message
                if self._verb:
                    f = Path(pre_mri_data.save_name).name
                    msg = f"{vol_idx-1}, "
                    msg += "Retrospective slice-timing correction"
                    msg += f" is done for {f}."
                    self.logmsg(msg)

                # Set save_name
                pre_mri_data.save_name = 'ts.' + pre_mri_data.save_name

                # Save processed image
                if self.save_proc:
                    savefname = self.save_data(pre_mri_data)
                    self.saved_files.append(savefname)

                # Run the next process
                if self.next_proc:
                    self.next_proc.do_proc(pre_mri_data, vol_idx-1,
                                           time.time())

                if self.method == 'cubic':
                    # For cubic intepolation, two previous volumes are needed
                    if len(self.pre_data) < 2:
                        self.pre_data.append(dataV)
                        self.pre_mri_data = copy.deepcopy(mri_data)
                        return

                del self.pre_mri_data

            # --- Perform timing correction -----------------------------------
            if self.method == 'linear':
                shift_dataV = self.wm1 * self.pre_data[-1] + self.w0 * dataV
            elif self.method == 'cubic':
                shift_dataV = \
                    self.wm2 * self.pre_data[-2] + \
                    self.wm1 * self.pre_data[-1] + \
                    self.w0 * dataV + \
                    self.wp1 * dataV

            # set corrected data in mri_data
            mri_data.img_data = shift_dataV

            # Save the processed data
            self.proc_data = mri_data.img_data.copy()

            # update pre_data list
            self.pre_data.append(dataV)
            self.pre_data.pop(0)

            # Record process time
            self.proc_time.append(time.time())
            if pre_proc_time is not None:
                proc_delay = self.proc_time[-1] - pre_proc_time
                if self.save_delay:
                    self.proc_delay.append(proc_delay)

            # log message
            if self._verb:
                f = Path(mri_data.save_name).name
                msg = f'{vol_idx}, Slice-timing correction is done for {f}'
                if pre_proc_time is not None:
                    msg += f' (process time {proc_delay:.4f}s)'
                msg += '.'

                self.logmsg(msg)

            # Set save_name
            mri_data.save_name = 'ts.' + mri_data.save_name

            # Run the next process
            if self.next_proc:
                self.next_proc.do_proc(mri_data, vol_idx, self.proc_time[-1])

            # Save processed image
            if self.save_proc:
                self.keep_processed_image(mri_data)

        except Exception:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            errmsg = '{}, {}:{}'.format(
                    exc_type, exc_tb.tb_frame.f_code.co_filename,
                    exc_tb.tb_lineno)
            self.errmsg(errmsg, no_pop=True)
            traceback.print_exc(file=self._err_out)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def reset(self):
        """Reset process parameters
        """

        if self.verb:
            self.logmsg(f"Reset {self.__class__.__name__} module.")

        self.pre_data = []
        super(RTP_TSHIFT, self).reset()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_param(self, attr, val=None, reset_fn=None, echo=False):
        """
        When reset_fn is None, set_param is considered to be called from
        load_parameters function
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

        elif attr == 'ignore_init' and reset_fn is None:
            if hasattr(self, 'ui_ignorInit_spBx'):
                self.ui_ignorInit_spBx.setValue(val)

        elif attr == 'method' and reset_fn is None:
            if hasattr(self, 'ui_method_cmbBx'):
                self.ui_method_cmbBx.setCurrentText(val)

        elif attr == 'ref_time' and reset_fn is None:
            if hasattr(self, 'ui_refTime_dSpBx'):
                self.ui_refTime_dSpBx.setValue(val)

        elif attr == 'slice_timing_from_sample':
            if val is None:
                fname = self.select_file_dlg(
                        'TSHIFT: Selct slice timing sample',
                        self.watch_dir, "*.BRIK*")
                if fname[0] == '':
                    return -1

                val = fname[0]

            self.slice_timing_from_sample(val)
            self.set_slice_timing_uis()
            return 0

        elif attr == 'TR' and reset_fn is None:
            if hasattr(self, 'ui_TR_dSpBx'):
                self.ui_TR_dSpBx.setValue(val)

        elif attr == 'slice_timing':
            if type(val) == str:
                try:
                    val = eval(val)
                except Exception:
                    if reset_fn:
                        reset_fn(str(getattr(self, attr)))
                    return
            else:
                if hasattr(self, 'ui_SlTiming_lnEd'):
                    self.ui_SlTiming_lnEd.setText(str(val))

        elif attr == 'slice_dim' and reset_fn is None:
            if hasattr(self, 'ui_sliceDim_cmbBx'):
                self.ui_sliceDim_cmbBx.setCurrentIndex(val)

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
            print("{}.".format(self.__class__.__name__) + attr, '=',
                  getattr(self, attr))

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_slice_timing_uis(self):
        if hasattr(self, 'ui_TR_dSpBx'):
            self.ui_TR_dSpBx.setValue(self.TR)

        if hasattr(self, 'ui_SlTiming_lnEd'):
            self.ui_SlTiming_lnEd.setText("{}".format(self.slice_timing))

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def ui_set_param(self):

        ui_rows = []
        self.ui_objs = []

        # enabled
        self.ui_enabled_rdb = QtWidgets.QRadioButton("Enable")
        self.ui_enabled_rdb.setChecked(self.enabled)
        self.ui_enabled_rdb.toggled.connect(
                lambda checked:
                self.set_param('enabled', checked,
                               self.ui_enabled_rdb.setChecked))
        ui_rows.append((self.ui_enabled_rdb, None))

        # ignore_init
        var_lb = QtWidgets.QLabel("Ignore initial volumes :")
        self.ui_ignorInit_spBx = QtWidgets.QSpinBox()
        self.ui_ignorInit_spBx.setValue(self.ignore_init)
        self.ui_ignorInit_spBx.setMinimum(0)
        self.ui_ignorInit_spBx.valueChanged.connect(
                lambda x: self.set_param('ignore_init', x,
                                         self.ui_ignorInit_spBx.setValue))
        ui_rows.append((var_lb, self.ui_ignorInit_spBx))
        self.ui_objs.extend([var_lb, self.ui_ignorInit_spBx])

        # method
        var_lb = QtWidgets.QLabel("Temporal interpolation method :")
        self.ui_method_cmbBx = QtWidgets.QComboBox()
        self.ui_method_cmbBx.addItems(['linear', 'cubic'])
        self.ui_method_cmbBx.setCurrentText(self.method)
        self.ui_method_cmbBx.currentIndexChanged.connect(
                lambda idx:
                self.set_param('method',
                               self.ui_method_cmbBx.currentText(),
                               self.ui_method_cmbBx.setCurrentText))
        ui_rows.append((var_lb, self.ui_method_cmbBx))
        self.ui_objs.extend([var_lb, self.ui_method_cmbBx])

        # ref_time
        var_lb = QtWidgets.QLabel("Reference time :")
        self.ui_refTime_dSpBx = QtWidgets.QDoubleSpinBox()
        self.ui_refTime_dSpBx.setMinimum(0.000)
        self.ui_refTime_dSpBx.setSingleStep(0.001)
        self.ui_refTime_dSpBx.setDecimals(3)
        self.ui_refTime_dSpBx.setSuffix(" seconds")
        self.ui_refTime_dSpBx.setValue(self.ref_time)
        self.ui_refTime_dSpBx.valueChanged.connect(
                lambda x: self.set_param('ref_time', x,
                                         self.ui_refTime_dSpBx.setValue))
        ui_rows.append((var_lb, self.ui_refTime_dSpBx))
        self.ui_objs.extend([var_lb, self.ui_refTime_dSpBx])

        # Load from sample button
        self.ui_setfrmSample_btn = QtWidgets.QPushButton(
                'Get slice timing parameters from a sample file')
        self.ui_setfrmSample_btn.clicked.connect(
                lambda: self.set_param('slice_timing_from_sample'))
        ui_rows.append((self.ui_setfrmSample_btn,))
        self.ui_objs.append(self.ui_setfrmSample_btn)

        # TR
        var_lb = QtWidgets.QLabel("TR :")
        self.ui_TR_dSpBx = QtWidgets.QDoubleSpinBox()
        self.ui_TR_dSpBx.setMinimum(0.000)
        self.ui_TR_dSpBx.setSingleStep(0.001)
        self.ui_TR_dSpBx.setDecimals(3)
        self.ui_TR_dSpBx.setSuffix(" seconds")
        self.ui_TR_dSpBx.setValue(self.TR)
        self.ui_TR_dSpBx.valueChanged.connect(
                lambda x: self.set_param('TR', x, self.ui_TR_dSpBx.setValue))
        ui_rows.append((var_lb, self.ui_TR_dSpBx))
        self.ui_objs.extend([var_lb, self.ui_TR_dSpBx])

        # slice_timing
        var_lb = QtWidgets.QLabel("Slice timings (sec.) :\n[1st, 2nd, ...]")
        self.ui_SlTiming_lnEd = QtWidgets.QLineEdit()
        self.ui_SlTiming_lnEd.setText("{}".format(self.slice_timing))
        self.ui_SlTiming_lnEd.editingFinished.connect(
                lambda: self.set_param('slice_timing',
                                       self.ui_SlTiming_lnEd.text(),
                                       self.ui_SlTiming_lnEd.setText))
        ui_rows.append((var_lb, self.ui_SlTiming_lnEd))
        self.ui_objs.extend([var_lb, self.ui_SlTiming_lnEd])

        # slice_dim
        var_lb = QtWidgets.QLabel("Slice orientation :")
        self.ui_sliceDim_cmbBx = QtWidgets.QComboBox()
        self.ui_sliceDim_cmbBx.addItems(['x (Sagital)', 'y (Coronal)',
                                         'z (Axial)'])
        self.ui_sliceDim_cmbBx.setCurrentIndex(self.slice_dim)
        self.ui_sliceDim_cmbBx.currentIndexChanged.connect(
                lambda idx:
                self.set_param('slice_dim',
                               self.ui_sliceDim_cmbBx.currentIndex(),
                               self.ui_sliceDim_cmbBx.setCurrentInde))
        ui_rows.append((var_lb, self.ui_sliceDim_cmbBx))
        self.ui_objs.extend([var_lb, self.ui_sliceDim_cmbBx])

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
        excld_opts = ('watch_dir', 'save_dir', 'pre_data', 'pre_mri_data',
                      'prep_weight', 'wm2', 'wm1', 'w0', 'wp1', 'r1wm2',
                      'r1wm1', 'r1w0', 'r1wp1')
        sel_opts = {}
        for k, v in all_opts.items():
            if k in excld_opts:
                continue
            if isinstance(v, Path):
                v = str(v)
            sel_opts[k] = v

        return sel_opts


# %% main =====================================================================
if __name__ == '__main__':

    # Test
    rtp_tshift = RTP_TSHIFT()
    rtp_tshift.method = 'cubic'
    rtp_tshift.ignore_init = 3
    rtp_tshift.verb = True
    rtp_tshift.ref_time = 0

    # Set slice timing from a sample data
    src_dir = Path.home() / 'Development' / 'RTPfMRI' / 'test_data' / 'E13270'
    sample_epi = src_dir / 'epiRT_scan_6__003+orig.BRIK'
    assert sample_epi.is_file()
    rtp_tshift.slice_timing_from_sample(sample_epi)

    # Set parameters for debug
    rtp_tshift.save_dir = src_dir / 'RTP'
    rtp_tshift.save_proc = True
    rtp_tshift.verb = True

    # Get test data
    fname = src_dir / 'epiRTeeg_scan_7__004+orig.BRIK'
    img = nib.load(fname)
    img_data = np.squeeze(img.get_fdata()).astype(np.float32)
    affine = img.affine
    img_header = img.header

    ets = []
    for iv in range(30):
        st = time.time()
        mri_data = MRI_data(img_data[:, :, :, iv], affine, img_header,
                            'temp{}'.format(iv))
        rtp_tshift.do_proc(mri_data, iv, st)
        ets.append(time.time() - st)

    # --- Load processed data  ---
    # Preapre data array
    img0 = nib.load(rtp_tshift.saved_files[0])
    img_data_ts = np.zeros([*img0.shape, len(rtp_tshift.saved_files)],
                           dtype=img0.get_data_dtype())
    for ii, f in enumerate(rtp_tshift.saved_files):
        img_data_ts[:, :, :, ii] = nib.load(f).get_fdata()

    # --- Plot ---
    if False:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.hist(ets, bins=30)
        np.median(ets)

        sli = int(len(rtp_tshift.slice_timing)//2)
        xi = np.arange(30-rtp_tshift.ignore_init)
        plt.figure()
        xraw = (np.array(xi)+rtp_tshift.ignore_init) * rtp_tshift.TR + \
            rtp_tshift.slice_timing[sli]
        plt.plot(xraw, img_data[64, 64, sli, rtp_tshift.ignore_init+xi],
                 label='raw')
        xshift = (np.array(xi)+rtp_tshift.ignore_init) * rtp_tshift.TR + \
            rtp_tshift.ref_time
        plt.plot(xshift, img_data_ts[64, 64, sli, xi], '--', label='shift')
        plt.legend()
        plt.title("Shift -{}s".format(rtp_tshift.slice_timing[sli]))
