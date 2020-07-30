#!/usr/bin/env ipython
# -*- coding: utf-8 -*-
"""
Real-time processing class of volume registration for motion correction

@author: mmisaki@laureateinstitute.org
"""


# %% import ===================================================================
from pathlib import Path
import sys
import time
import re
import numpy as np
import nibabel as nib
import ctypes
from six import string_types
import traceback
from PyQt5 import QtWidgets, QtCore
import matplotlib as mpl

from .rtp_common import RTP, MRI_data, MatplotlibWindow

mpl.rcParams['font.size'] = 8

try:
    librtp_path = str(Path(__file__).absolute().parent / 'librtp.so')
except Exception:
    librtp_path = './librtp.so'


# %% RTP_VOLREG ===============================================================
class RTP_VOLREG(RTP):
    """
    Real-time online volume registration for motion correction
    Calls AFNI functions in librtp.so, which is compiled from libmri.so
    """

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __init__(self, ref_vol=0, regmode=2, **kwargs):
        """
        Parameters
        ----------
        ref_vol: int or str
            if type(ref_vol) == int:
                ref_vol is an index of volume in the current sequence to be
                used as the reference for alignment.
            if isinstance(ref_vol, string_types) or isinstance(ref_vol, Path):
                ref_vol is a path to a volume file used as the reference for
                alignment. Sub volume index can be given by '[x]' format like
                AFNI.
        regmode: int
            ID of image resampling method.
            0: NN,  1: LINEAR, 2: CUBIC, 3: FOURIER, 4: QUINTIC, 5: HEPTIC,
            6: TSSHIFT
            Default is CUBIC (2)
        """

        super(RTP_VOLREG, self).__init__(**kwargs)

        # Set instance parameters
        self.regmode = regmode

        # Initialize parameters and C library function call
        self.motion = np.ndarray([0, 6], dtype=np.float32)  # motion parameter
        self.set_ref_vol(ref_vol)
        self.setup_libfuncs()

        # alignment parameters (Set from AFNI plug_realtime default values)
        self.max_iter = 9
        self.dxy_thresh = 0.05  # pixels
        self.phi_thresh = 0.07  # degree

        # --- initialize for motion plot ---
        self.plt_xi = []
        self.plt_motion = []
        for ii in range(6):
            self.plt_motion.append(list([]))

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    @property
    def proc_ready(self):
        self._proc_ready = self.ref_vol is not None
        if not self._proc_ready:
            errmsg = "Refence volume or volume index has not been set."
            self.errmsg(errmsg)

        if self.next_proc:
            self._proc_ready &= self.next_proc.proc_ready

        return self._proc_ready

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def setup_libfuncs(self):
        """ Setup library function access """

        librtp = ctypes.cdll.LoadLibrary(librtp_path)

        # -- Define rtp_align_setup --
        self.rtp_align_setup = librtp.rtp_align_setup
        self.rtp_align_setup.argtypes = \
            [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
             ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_int,
             ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
             ctypes.c_void_p, ctypes.c_void_p]

        # -- Define rtp_align_one --
        self.rtp_align_one = librtp.rtp_align_one
        self.rtp_align_one.argtypes = \
            [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
             ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_void_p,
             ctypes.c_void_p, ctypes.c_int, ctypes.c_int,  ctypes.c_int,
             ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
             ctypes.c_void_p]

        # -- Define THD_rota_vol --
        self.THD_rota_vol = librtp.THD_rota_vol
        self.THD_rota_vol.argtypes = \
            [ctypes.c_int, ctypes.c_int, ctypes.c_int,
             ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_void_p,
             ctypes.c_int, ctypes.c_float, ctypes.c_int, ctypes.c_float,
             ctypes.c_int, ctypes.c_float,
             ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float]

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_ref_vol(self, ref_vol, ref_vi=0):
        if isinstance(ref_vol, string_types) or isinstance(ref_vol, Path):
            ref_vol0 = ref_vol
            ma = re.search(r"\'*\[(\d+)\]\'*$", str(ref_vol))
            if ma:
                ref_vi = int(ma.groups()[0])
                ref_vol = re.sub(r"\'*\[(\d+)\]\'*$", '', str(ref_vol))

            # refname should be a filename
            img = nib.load(ref_vol)
            ref_vol = img.get_fdata()
            affine = img.affine
            if ref_vol.ndim == 4:
                ref_vol = ref_vol[:, :, :, ref_vi]
            assert ref_vol.ndim == 3

            self.ref_vol = MRI_data(ref_vol.astype(np.float32), affine,
                                    img.header, 'volreg_reference')
            self.align_setup()  # Prepare alignment volume

            if self._verb:
                msg = f"Alignment reference = {ref_vol0}"
                if ma is None:
                    msg += f"[{ref_vi}]"
                self.logmsg(msg)

        else:
            # ref_vol is a number. Get reference from ref_vol-th volume in
            # ongoing scan. Once this reference is set, this will be kept
            # during the lifetime of the instance.
            self.ref_vol = int(ref_vol)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def align_setup(self):
        """
        Set up refernce image array and Choleski decomposed matrix for lsqfit

        Reference image array includes original image, three rotation gradient
        images and three shift gradient images

        Calls rtp_align_setup in librtp.so, which is a modified version of
        mri_3dalign_setup in mri_3dalign.c of afni source.

        Setup variables
        ---------------
        self.fitim: numpy array of float32
            seven reference volumes
        self.chol_fitim: numpy array of float64
            Choleski decomposition of weighted covariance matrix between
            refrence images.

        C function definition
        ---------------------
        int rtp_align_setup(float *base_im, int nx, int ny, int nz,
                            float dx, float dy, float dz,
                            int ax1, int ax2, int ax3, int regmode,
                            int nref, float *ref_ims, double *chol_fitim)
        """

        # -- Set parameters --
        # image dimension
        self.nx, self.ny, self.nz = self.ref_vol.img_data.shape
        if hasattr(self.ref_vol.img_header, 'info'):
            self.dx, self.dy, self.dz = \
                np.abs(self.ref_vol.img_header.info['DELTA'])
        elif hasattr(self.ref_vol.img_header, 'get_zooms'):
            self.dx, self.dy, self.dz = self.ref_vol.img_header.get_zooms()[:3]
        else:
            self.errmsg("No voxel size information in ref_vol header")

        # rotate orientation
        self.ax1 = 2  # z-axis, roll
        self.ax2 = 0  # x-axis, pitch
        self.ax3 = 1  # y-axis, yow

        # Copy base image data and get pointer
        base_image0 = self.ref_vol.img_data
        base_im_arr = base_image0.astype(np.float32)
        # x,y,z -> y,z,x -> z,y,x
        base_im_arr = np.moveaxis(np.moveaxis(base_im_arr, 0, -1), 0, 1).copy()
        base_im_p = base_im_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        # -- Prepare return arrays --
        # Reference image array
        nxyz = self.nx * self.ny * self.nz
        nref = 7
        ref_img_arr = np.ndarray(nref * nxyz, dtype=np.float32)
        ref_ims_p = ref_img_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        # Choleski decomposed matrix for lsqfit
        chol_fitim_arr = np.ndarray((nref, nref), dtype=np.float64)
        chol_fitim_p = chol_fitim_arr.ctypes.data_as(
                ctypes.POINTER(ctypes.c_double))

        # -- run func --
        self.rtp_align_setup(base_im_p, self.nx, self.ny, self.nz, self.dx,
                             self.dy, self.dz, self.ax1, self.ax2, self.ax3,
                             self.regmode, nref, ref_ims_p, chol_fitim_p)

        self.fitim = ref_img_arr
        self.chol_fitim = chol_fitim_arr

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def align_one(self, mri_data):
        """
        Align one volume [0] of mri_data to the reference image (self.ref_vol)

        Calls rtp_align_one in librtp.so, which is a modified version of
        mri_3dalign_one in mri_3dalign.c of afni source.

        Return
        ------
        tim: 3D array of float32
            aligned volume data
        motpar: array of float32
            six motion parameters: roll, pitch, yaw, dx, dy,dz

        C function definition
        ---------------------
        int rtp_align_one(float *fim, int nx, int ny, int nz, float dx,
                          float dy, float dz, float *fitim, double *chol_fitim,
                          int nref, int ax1, int ax2, int ax3,
                          float *init_motpar, int regmode, float *tim,
                          float *motpar)
        """

        """DEBUG
        tmpf = Path.home() / 'Development/RTPfMRI/sim_tmp/tim.bin'
        tempdata = np.fromfile(tmpf, dtype=np.float32)
        tmpimg = np.reshape(tempdata, (34, 128, 128))
        tmp_tim = np.moveaxis(np.moveaxis(tmpimg, 0, -1), 0, 1)

        import matplotlib.pyplot as plt
        self = rtp_volreg
        mri_data = mri_data0
        """

        # Copy function image data and get pointer
        fim0 = mri_data.img_data
        if fim0.dtype != np.float32:
            fim_arr = fim0.astype(np.float32)
        else:
            fim_arr = fim0
        fim_arr = np.moveaxis(np.moveaxis(fim_arr, 0, -1), 0, 1).copy()
        fim_p = fim_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        # Get fitim and chol_fitim data pointer
        fitim_p = self.fitim.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        chol_fitim_p = self.chol_fitim.ctypes.data_as(
                ctypes.POINTER(ctypes.c_double))

        # Initial motion parameter
        if len(self.motion) and not np.any(np.isnan(self.motion[-1])):
            init_motpar = self.motion[-1]
        else:
            init_motpar = np.zeros(6, dtype=np.float32)

        init_motpar_p = init_motpar.ctypes.data_as(
                ctypes.POINTER(ctypes.c_float))

        # Prepare return arrays
        nxyz = self.nx * self.ny * self.nz
        tim_arr = np.ndarray(nxyz, dtype=np.float32)
        tim_p = tim_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        motpar = np.ndarray(7, dtype=np.float32)
        motpar_p = motpar.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        # -- run func --
        self.rtp_align_one(fim_p, self.nx, self.ny, self.nz, self.dx, self.dy,
                           self.dz, fitim_p, chol_fitim_p, 7, self.ax1,
                           self.ax2, self.ax3, init_motpar_p, self.regmode,
                           tim_p, motpar_p)

        tim = np.reshape(tim_arr, (self.nz, self.ny, self.nx))
        tim = np.moveaxis(np.moveaxis(tim, 0, -1), 0, 1)

        return tim, motpar[1:]

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    class PlotMotion(QtCore.QObject):
        finished = QtCore.pyqtSignal()

        def __init__(self, root, main_win=None):
            super().__init__()

            self.root = root
            self.main_win = main_win
            self.abort = False

            # Initialize figure
            plt_winname = 'Motion'
            self.plt_win = MatplotlibWindow()
            self.plt_win.setWindowTitle(plt_winname)

            # set position
            if main_win is not None:
                main_geom = main_win.geometry()
                x = main_geom.x() + main_geom.width() + 10
                y = main_geom.y() + 205
            else:
                x, y = (0, 0)
            self.plt_win.setGeometry(x, y, 500, 500)

            # Set axis
            self.mot_labels = ['roll (deg.)', 'pitch (deg.)', 'yaw (deg.)',
                               'dS (mm)', 'dL (mm)', 'dP (mm)']
            self._axes = self.plt_win.canvas.figure.subplots(6, 1)
            self.plt_win.canvas.figure.subplots_adjust(
                    left=0.15, bottom=0.08, right=0.95, top=0.97, hspace=0.35)
            self._ln = []
            for ii, ax in enumerate(self._axes):
                ax.set_ylabel(self.mot_labels[ii])
                ax.set_xlim(0, 10)
                self._ln.append(ax.plot(0, 0))

            ax.set_xlabel('TR')

            # show window
            self.plt_win.show()

            self.plt_win.canvas.draw()
            self.plt_win.canvas.start_event_loop(0.005)

        # ---------------------------------------------------------------------
        def run(self):
            plt_xi = self.root.plt_xi.copy()
            while self.plt_win.isVisible() and not self.abort:
                if self.main_win is not None and not self.main_win.isVisible():
                    break

                if len(self.root.plt_xi) == len(plt_xi):
                    time.sleep(0.1)
                    continue

                try:
                    # Plot motion
                    plt_xi = self.root.plt_xi.copy()
                    plt_motion = self.root.plt_motion
                    for ii, ax in enumerate(self._axes):
                        ll = min(len(plt_xi), len(plt_motion[ii]))
                        if ll == 0:
                            continue

                        self._ln[ii][0].set_data(plt_xi[:ll],
                                                 plt_motion[ii][:ll])
                        ax.relim()
                        ax.autoscale_view()

                        xl = ax.get_xlim()
                        if (plt_xi[-1]//10 + 1)*10 > xl[1]:
                            ax.set_xlim([0, (plt_xi[-1]//10 + 1)*10])

                    self.plt_win.canvas.draw()
                    self.plt_win.canvas.start_event_loop(0.01)

                except IndexError:
                    continue

                except Exception as e:
                    self.root.errmsg(e)
                    sys.stdout.flush()
                    continue

            self.end_thread()

        # ---------------------------------------------------------------------
        def end_thread(self):
            if self.plt_win.isVisible():
                self.plt_win.close()

            self.finished.emit()

            if self.main_win is not None:
                if hasattr(self.main_win, 'chbShowMotion'):
                    self.main_win.chbShowMotion.setCheckState(0)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def open_motion_plot(self):
        if hasattr(self, 'thPltMotion') and self.thPltMotion.isRunning():
            return

        self.thPltMotion = QtCore.QThread()
        self.pltMotion = RTP_VOLREG.PlotMotion(self, main_win=self.main_win)
        self.pltMotion.moveToThread(self.thPltMotion)
        self.thPltMotion.started.connect(self.pltMotion.run)
        self.pltMotion.finished.connect(self.thPltMotion.quit)
        self.thPltMotion.start()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def close_motion_plot(self):
        if hasattr(self, 'thPltMotion') and self.thPltMotion.isRunning():
            self.pltMotion.abort = True
            if not self.thPltMotion.wait(1):
                self.pltMotion.finished.emit()
                self.thPltMotion.wait()

            del self.thPltMotion

        if hasattr(self, 'pltMotion'):
            del self.pltMotion

        if self.main_win is not None:
            if self.main_win.chbShowMotion.checkState() != 0:
                self.main_win.chbShowMotion.setCheckState(0)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def do_proc(self, mri_data, vol_idx=None, pre_proc_time=None, **kwargs):
        try:
            # Increment the number of received volume
            self.vol_num += 1
            if vol_idx is None:
                vol_idx = self.vol_num

            # Fill motion vector for missing volumes with zero
            while self.motion.shape[0] < vol_idx:
                self.motion = np.concatenate(
                        [self.motion, np.zeros((1, 6), dtype=np.float32)],
                        axis=0)

            # if self.ref_vol is int (index of reference volume in the current
            # sequence), wait for the reference index and set the reference
            # when it comes
            if type(self.ref_vol) is int:
                if vol_idx < self.ref_vol:
                    # Append nan vector
                    mot = np.zeros((1, 6), dtype=np.float32)
                    self.motion = np.concatenate([self.motion, mot], axis=0)
                    return
                elif vol_idx >= self.ref_vol:
                    ref_vi = self.ref_vol
                    self.ref_vol = mri_data
                    self.align_setup()
                    # Append zero vector
                    mot = np.zeros((1, 6), dtype=np.float32)
                    self.motion = np.concatenate([self.motion, mot], axis=0)
                    if self._verb:
                        msg = f"Alignment reference is set to volume {ref_vi}"
                        msg += " of current sequence."
                        self.logmsg(msg)
                    return

            # Perform volume alignment
            reg_dataV, mot = self.align_one(mri_data)

            # set aligned data in mri_data and save motion parameters
            mri_data.img_data = reg_dataV
            mot = mot[np.newaxis, [0, 1, 2, 5, 3, 4]]
            self.motion = np.concatenate([self.motion, mot], axis=0)

            # Save the processed data
            self.proc_data = mri_data.img_data.copy()

            # Update motion plot
            self.plt_xi.append(vol_idx)
            for ii in range(6):
                self.plt_motion[ii].append(self.motion[-1][ii])

            # Record process time
            self.proc_time.append(time.time())
            if pre_proc_time is not None:
                proc_delay = self.proc_time[-1] - pre_proc_time
                if self.save_delay:
                    self.proc_delay.append(proc_delay)

            # log message
            if self._verb:
                f = Path(mri_data.save_name).name
                msg = f'{vol_idx}, Volume registration is done for {f}'
                if pre_proc_time is not None:
                    msg += f' (process time {proc_delay:.4f}s)'
                msg += '.'
                self.logmsg(msg)

            # Set save_name
            mri_data.save_name = 'vr.' + mri_data.save_name

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
    def reset(self):
        """Reset process parameters
        """

        if self.verb:
            self.logmsg(f"Reset {self.__class__.__name__} module.")

        # Reset running variables
        self.motion = np.ndarray([0, 6], dtype=np.float32)

        # Reset plot values
        self.plt_xi[:] = []
        for ii in range(6):
            self.plt_motion[ii][:] = []

        super(RTP_VOLREG, self).reset()

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

        elif attr == 'ref_vol':
            if isinstance(val, Path):
                val = str(val)

            if type(val) == int:
                self.set_ref_vol(val)
                if hasattr(self, 'ui_baseVol_lb'):
                    self.ui_baseVol_lb.setText(
                            f"Internal run volume index {val}")

            elif isinstance(val, MRI_data):
                setattr(self, attr, val)
                if hasattr(self, 'ui_baseVol_cmbBx'):
                    self.ui_baseVol_cmbBx.setCurrentIndex(0)

            elif 'internal' in val:
                num, okflag = QtWidgets.QInputDialog.getInt(
                        None, "Internal base volume index",
                        "volume index (0 is first)")
                if not okflag:
                    return

                self.set_ref_vol(num)
                if hasattr(self, 'ui_baseVol_lb'):
                    self.ui_baseVol_lb.setText(
                            "Internal run volume index {num}")
                self.ref_fname = ''

            elif val == 'external file':
                fname = self.select_file_dlg('VOLREG: Selct base volume',
                                             self.watch_dir, "*.BRIK* *.nii*")
                if fname[0] == '':
                    if reset_fn:
                        reset_fn(1)
                    return -1

                ref_img = nib.load(fname[0])
                ref_fname = fname[0]
                if len(ref_img.shape) > 3 and ref_img.shape[3] > 1:
                    num, okflag = QtWidgets.QInputDialog.getInt(
                            None, "VOLREG: Select sub volume",
                            "sub-volume index (0 is first)", 0, 0,
                            ref_img.shape[3])
                    if fname[0] == '':
                        if reset_fn:
                            reset_fn(1)
                        return

                    ref_fname += f"[{num}]"
                else:
                    num = 0

                self.ref_fname = ref_fname
                self.set_ref_vol(fname[0], num)
                if hasattr(self, 'ui_baseVol_lb'):
                    self.ui_baseVol_lb.setText(str(ref_fname))

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

                if reset_fn is None and hasattr(self, 'ui_baseVol_cmbBx'):
                    self.ui_baseVol_cmbBx.setCurrentIndex(0)

                self.ref_fname = val
                self.set_ref_vol(fname, num)
                if hasattr(self, 'ui_baseVol_lb'):
                    self.ui_baseVol_lb.setText(str(self.ref_fname))

            return

        elif attr == 'ref_fname':
            if len(val):
                ma = re.search(r"\[(\d+)\]", val)
                if ma:
                    num = int(ma.groups()[0])
                    val = re.sub(r"\[(\d+)\]", '', val)
                else:
                    num = 0

                if not Path(val).is_file():
                    return

                if hasattr(self, 'ui_baseVol_lb'):
                    self.ui_baseVol_lb.setText(str(val))
                self.set_ref_vol(val, num)

        elif attr == 'regmode' and reset_fn is None:
            if hasattr(self, 'ui_regmode_cmbBx'):
                self.ui_regmode_cmbBx.setCurrentIndex(val)

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

        ui_rows = []
        self.ui_objs = []

        # enabled
        self.ui_enabled_rdb = QtWidgets.QRadioButton("Enable")
        self.ui_enabled_rdb.setChecked(self.enabled)
        self.ui_enabled_rdb.toggled.connect(
                lambda checked: self.set_param('enabled', checked,
                                               self.ui_enabled_rdb.setChecked))
        ui_rows.append((self.ui_enabled_rdb, None))

        # ref_vol
        var_lb = QtWidgets.QLabel("Base volume :")
        self.ui_baseVol_cmbBx = QtWidgets.QComboBox()
        self.ui_baseVol_cmbBx.addItems(['external file',
                                        'index of internal run'])
        self.ui_baseVol_cmbBx.activated.connect(
                lambda idx:
                self.set_param('ref_vol',
                               self.ui_baseVol_cmbBx.currentText(),
                               self.ui_baseVol_cmbBx.setCurrentIndex))
        ui_rows.append((var_lb, self.ui_baseVol_cmbBx))
        self.ui_baseVol_lb = QtWidgets.QLabel()
        ui_rows.append((None, self.ui_baseVol_lb))
        self.ui_objs.extend([var_lb, self.ui_baseVol_cmbBx,
                             self.ui_baseVol_lb])

        if isinstance(self.ref_vol, string_types):
            self.ui_baseVol_cmbBx.setCurrentIndex(0)
            self.ui_baseVol_lb.setText(str(self.ref_vol))
        else:
            self.ui_baseVol_cmbBx.setCurrentIndex(1)
            self.ui_baseVol_lb.setText(
                    f"Internal run volume index {self.ref_vol}")

        # regmode
        var_lb = QtWidgets.QLabel("Resampling interpolation :")
        self.ui_regmode_cmbBx = QtWidgets.QComboBox()
        self.ui_regmode_cmbBx.addItems(['Nearest Neighbor', 'Linear', 'Cubic',
                                        'Fourier', 'Quintic', 'Heptic'])
        self.ui_regmode_cmbBx.setCurrentIndex(self.regmode)
        self.ui_regmode_cmbBx.currentIndexChanged.connect(
                lambda idx:
                self.set_param('regmode',
                               self.ui_regmode_cmbBx.currentIndex(),
                               self.ui_regmode_cmbBx.setCurrentIndex))
        ui_rows.append((var_lb, self.ui_regmode_cmbBx))
        self.ui_objs.extend([var_lb, self.ui_regmode_cmbBx])

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
        excld_opts = ('watch_dir', 'save_dir', 'motion', 'plt_xi',
                      'plt_motion', 'chol_fitim', 'ref_vol', 'nx', 'ny', 'nz',
                      'fitim', 'dx', 'dy', 'dz', 'ax1', 'ax2', 'ax3')
        sel_opts = {}
        for k, v in all_opts.items():
            if k in excld_opts:
                continue
            if isinstance(v, Path):
                v = str(v)
            sel_opts[k] = v

        return sel_opts

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __del__(self):
        self.close_motion_plot()


# %% main =====================================================================
if __name__ == '__main__':

    # app = QtWidgets.QApplication(sys.argv)

    # Set test data
    src_dir = Path.home() / 'Development' / 'RTPfMRI' / 'test_data' / 'E13270'

    # epi volumes
    sample_epi = src_dir / 'epiRTeeg_scan_7__004+orig.BRIK'
    assert sample_epi.is_file()
    img = nib.load(sample_epi)
    img_data = np.squeeze(img.get_fdata()).astype(np.float32)
    affine = img.affine
    img_header = img.header

    ets = {}
    regmode = 2
    rtp_volreg = RTP_VOLREG(regmode=regmode)

    # open plot
    # rtp_volreg.open_motion_plot()

    # Set referece volume
    refname = src_dir / 'epiRT_scan_6__005+orig.BRIK[0]'
    rtp_volreg.set_ref_vol(refname)

    # Set parameters for debug
    rtp_volreg.save_dir = src_dir / 'RTP'

    rtp_volreg.reset()

    ets[regmode] = []
    for iv in range(50):  # img_data.shape[3]):
        mri_data = MRI_data(img_data[:, :, :, iv], affine, img_header,
                            f'temp{iv}')
        st = time.time()
        rtp_volreg.do_proc(mri_data, iv, st)
        ets[regmode].append(time.time() - st)
        time.sleep(1)

    # --- Compare with 3dvolreg result ---
    motf = src_dir / 'dfile.scan_7__004.1D'
    assert motf.is_file()
    C = open(motf, 'r').read()
    mot = np.array([[float(v) for v in ln.split()]
                    for ln in C.rstrip().split('\n')])

    # Plot
    import matplotlib.pyplot as plt
    motlab = ('roll', 'pitch', 'yaw', 'dS', 'dL', 'dP')
    nt = rtp_volreg.motion.shape[0]
    for ii in range(6):
        plt.figure()
        plt.plot(rtp_volreg.motion[:, ii], label='rtp_volreg')
        plt.plot(mot[:nt, ii], label='3dvolreg')
        plt.legend()
        r = np.corrcoef(rtp_volreg.motion[:, ii], mot[:nt, ii])[0, 1]
        plt.title(f"{motlab[ii]}: r={r}")

    """
    rtp_volreg tends to estimate a less temporal difference than 3dvolreg
    because rtp_volreg starts estimation from the same value as the previous
    one for faster covergence, while 3dvolreg starts from 0.
    """
