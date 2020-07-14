#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mmisaki@laureateinstitute.org
"""

# %% import ===================================================================
from pathlib import Path
import os
import sys
import subprocess
import time
import multiprocessing
import pexpect
import re
import nibabel as nib
import numpy as np
from PyQt5 import QtWidgets, QtCore
import matplotlib.pyplot as plt
import datetime
from functools import partial
import pty
import socket

# Try to load modules from the same directory
from .rtp_common import RTP, boot_afni, MatplotlibWindow, DlgProgressBar
from .rtp_watch import RTP_WATCH
from .rtp_tshift import RTP_TSHIFT
from .rtp_volreg import RTP_VOLREG
from .rtp_smooth import RTP_SMOOTH
from .rtp_regress import RTP_REGRESS
from .rtp_scanonset import RTP_SCANONSET
from .rtp_physio import RTP_PHYSIO
from .rtp_retrots import RTP_RETROTS
from .mri_sim import rtMRISim


# %% RTP_APP class ============================================================
class RTP_APP(RTP):
    """rtp application class
    """

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __init__(self, verb=True):
        super().__init__()  # call __init__() in RTP class

        # --- Initialize parameters -------------------------------------------
        self.verb = verb

        # The default template images
        template_dir = Path(__file__).parent / 'template'
        self.Template = template_dir / 'MNI152_2009_template.nii.gz'
        self.WM_template = template_dir / 'MNI152_2009_template_WM.nii.gz'
        self.Vent_template = template_dir / 'MNI152_2009_template_Vent.nii.gz'
        self.ROI_template = template_dir / 'MNI152_2009_template_LAmy.nii.gz'

        # Image files
        self.anat_orig = ''
        self.anat_orig_seg = ''
        self.alAnat = ''
        self.func_orig = ''

        self.WM_orig = ''
        self.Vent_orig = ''
        self.ROI_orig = ''
        self.ROI_mask = None

        self.RTP_mask = ''
        self.GSR_mask = ''

        # Set proc time for proc_anat progress bar
        self.proc_times = {}
        self.proc_times["BiasCorr"] = 40
        self.proc_times["dlSS"] = 15
        self.proc_times["AlAnat"] = 35
        self.proc_times["ANTs"] = 225
        self.proc_times["ApplyWarp_WM_template"] = 5
        self.proc_times["ApplyWarp_Vent_template"] = 5
        self.proc_times["ApplyWarp_ROI_template"] = 5

        # Prepare the timer to check the running status
        self.chk_run_timer = QtCore.QTimer()
        self.chk_run_timer.setSingleShot(True)
        self.chk_run_timer.timeout.connect(self.chkRunTimerEvent)
        self.max_watch_wait = np.inf  # seconds

        # Simulation data
        self.simEnabled = False
        self.simfMRIData = ''
        self.simECGData = ''
        self.simRespData = ''

        # Set psuedo serial port for simulating physio recording
        master, slave = pty.openpty()
        s_name = os.ttyname(slave)
        m_name = os.ttyname(master)
        self.simCom_descs = ['None', m_name + f",{master} (slave:{s_name})"]
        self.simPhysPort = self.simCom_descs[0]

        self.mri_sim = None

        # --- External application --------------------------------------------
        # Define an external application that receives the RTP signal
        # If self.ext_app is None, the signal will be saved in
        # RTP/feedback_values.csv file.
        # self.ext_app = Path(__file__).parent / 'rtp_signal_receiver.py'
        self.ext_app = None
        self.sig_save_file = Path(self.save_dir) / 'feedback_values.csv'

        self.extApp_proc = None  # external application process
        self.extApp_com = None   # communication soccket to extApp_proc
        self.app_com_port = 55555  # port number of the communication soccket

        # Open inter-process communication socket
        self.ext_com_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.ext_com_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        while True:
            try:
                self.ext_com_sock.bind(('localhost', self.app_com_port))
                break
            except Exception:
                self.app_com_port += 1

        self.ext_com_sock.listen()

        # --- Initialize ROI signal plot --------------------------------------
        self.num_ROIs = 1
        self.roi_labels = ['ROI']
        self.plt_xi = []
        self.plt_roi_sig = []
        for ii in range(self.num_ROIs):
            self.plt_roi_sig.append(list([]))

        # --- Create RTP module instances -------------------------------------
        self.rtp_watch = RTP_WATCH()
        self.rtp_tshift = RTP_TSHIFT()
        self.rtp_volreg = RTP_VOLREG()
        self.rtp_smooth = RTP_SMOOTH()
        self.rtp_regress = RTP_REGRESS()

        rtp_objs = dict()
        rtp_objs['SCANONSET'] = RTP_SCANONSET()
        rtp_objs['WATCH'] = self.rtp_watch
        rtp_objs['TSHIFT'] = self.rtp_tshift
        rtp_objs['VOLREG'] = self.rtp_volreg
        rtp_objs['SMOOTH'] = self.rtp_smooth
        rtp_objs['PHYSIO'] = RTP_PHYSIO(rtp_objs['SCANONSET'])
        rtp_objs['RETROTS'] = RTP_RETROTS()
        rtp_objs['REGRESS'] = self.rtp_regress
        rtp_objs['REGRESS'].physio = rtp_objs['PHYSIO']
        self.rtp_objs = rtp_objs

        self.enable_RTP = 2

    # --- Override these functions for custom application ---------------------
    #  boot_external_app, checkExtCom, proc_ready, reset, do_proc
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def boot_external_app(self):
        """Setup external appllication; e.g., neurofeedbakc presentation
        """

        if hasattr(self, 'extApp_proc') and self.extApp_proc is not None \
                and self.extApp_proc.poll() is None:
            # Kill running App
            self.extApp_com.send('Abort;'.encode())
            while self.extApp_proc.poll() is None:
                time.sleep(0.1)
            self.extApp_proc.kill()

        if hasattr(self, 'extApp_com') and self.extApp_com is not None:
            self.extApp_com.shutdown(1)  # shutdown communication server
            self.extApp_com.close()
            self.extApp_com = None

        # Boot extarnel application; rtp_signal_receiver.py
        cmd = f"{self.ext_app} --save_file={self.sig_save_file}"
        cmd += f" --host=localhost --port={self.app_com_port}"
        self.extApp_proc = subprocess.Popen(cmd, shell=True)

        # Check if the app is booted
        time.sleep(1)
        if self.extApp_proc.poll() is not None:
            self.extApp_proc.kill()
            outs, errs = self.extApp_proc.communicate()
            self.errmsg("Failed to boot external application;\n"
                        f" {cmd}")
            self.errmsg(errs.decode())
            self.extApp_proc = None
            return -1
        else:
            # Wait for connection from extApp_proc
            self.extApp_com, addr = self.ext_com_sock.accept()

            # Connected
            app_name = self.extApp_com.recv(1024).decode()
            self.logmsg("Connect from {}:{}.".format(app_name, addr))

        return 0

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    @property
    def proc_ready(self):
        """Check the procsess is ready
        """

        self._proc_ready = True
        if not Path(self.ROI_orig).is_file():
            self.errmsg(f'Not found ROI mask on orig space" {self.ROI_orig}.')
            self._proc_ready = False

        if self._proc_ready and self.ROI_mask is None:
            # Load ROI mask
            self.ROI_mask = nib.load(self.ROI_orig).get_data()

        return self._proc_ready

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def do_proc(self, mri_data, vol_idx=None, pre_proc_time=None):
        """ Processing for the RTP image (e.g., extracting the ROI average)

        This is called from the real-time ptocessing thread.
        """

        self.vol_num += 1
        if vol_idx is None:
            vol_idx = self.vol_num

        if self.ROI_mask is None:
            # Load ROI mask
            self.ROI_mask = nib.load(self.ROI_orig).get_data()

        try:
            # Mean signal in the ROI
            roimask = (self.ROI_mask > 0) & (np.abs(mri_data.img_data) > 0.0)
            mean_sig = np.nanmean(mri_data.img_data[roimask])

            # Send data to the external process
            if self.extApp_com is not None:
                try:
                    msg = f"{vol_idx}:{mean_sig};"
                    self.extApp_com.send(msg.encode())
                    self.logmsg(f"Sent '{msg}' to external app")

                except Exception as e:
                    self.errmsg(str(e), no_pop=True)
            else:
                # Save
                scan_onset = self.rtp_objs['SCANONSET'].scan_onset
                save_vals = (f"{time.time()-scan_onset:.4f}," +
                             f"{vol_idx},{mean_sig:.6f}")
                with open(self.sig_save_file, 'a') as save_fd:
                    print(save_vals, file=save_fd)
                self.logmsg(f"Write data '{save_vals}'")

            # Record process time
            self.proc_time.append(time.time())
            if pre_proc_time is not None:
                proc_delay = self.proc_time[-1] - pre_proc_time
                if self.save_delay:
                    self.proc_delay.append(proc_delay)

            # log message
            if self._verb:
                f = Path(mri_data.save_name).stem
                msg = f'{vol_idx}, ROI signal extraction is done for {f}'
                if pre_proc_time is not None:
                    msg += f' (delay {proc_delay:.4f})'
                msg += '.'
                self.logmsg(msg)

            # Update signal plot
            self.plt_xi.append(vol_idx)
            self.plt_roi_sig[0].append(mean_sig)

        except Exception as e:
            self.errmsg(str(e), no_pop=True)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def reset(self):
        """Reset process status
        """
        if self.verb:
            self.logmsg(f"Reset {self.__class__.__name__} module.")

        # Save proc_delay
        if self.save_delay and len(self.proc_delay):
            super().save_proc_delay()

        # Reset running variables
        self.vol_num = -1
        self.proc_time = []

        # Reset plot values
        self.plt_xi[:] = []
        for ii in range(self.num_ROIs):
            self.plt_roi_sig[ii][:] = []

        if hasattr(self, 'scan_onset'):
            del self.scan_onset

    # --- Internal utility functions ------------------------------------------
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _edit_command(self, labelTxt='Commdand line:', cmdTxt=''):
        cmd, okflag = QtWidgets.QInputDialog.getText(
                self.main_win, 'Edit command', labelTxt, text=cmdTxt)
        return cmd, okflag

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _proc_print_progress(self, proc):
        while proc.isalive():
            try:
                out = proc.read_nonblocking(size=100, timeout=0).decode()
                print('\n'.join(out.splitlines()), end='')
            except pexpect.TIMEOUT:
                pass
            except pexpect.EOF:
                break

            if hasattr(self, 'isVisible') and not self.isVisible():
                break

        try:
            out = proc.read_nonblocking(size=10000, timeout=0).decode()
            print('\n'.join(out.splitlines()) + '\n\n', end='')
        except pexpect.EOF:
            pass

        proc.close()
        return proc.exitstatus

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _show_cmd_progress(self, cmd, progress_bar=None, msgTxt='', desc=''):
        if progress_bar is not None:
            if len(msgTxt):
                progress_bar.set_msgTxt(msgTxt)
                QtWidgets.QApplication.processEvents()

            if len(desc):
                progress_bar.add_desc('\n' + desc)
                QtWidgets.QApplication.processEvents()

        try:
            ostr = subprocess.check_output(cmd, stderr=subprocess.STDOUT,
                                           shell=True).decode()
            if progress_bar is not None:
                progress_bar.add_msgTxt(ostr)
                QtWidgets.QApplication.processEvents()
            else:
                print(ostr)
        except Exception:
            self.errmsg(f"Failed execution:\n{cmd}")
            if hasattr(self, 'ui_procAnat_btn'):
                self.ui_procAnat_btn.setEnabled(True)
            if progress_bar is not None and progress_bar.isVisible():
                progress_bar.close()
            return 1

        return 0

        if progress_bar is not None:
            progress_bar.set_msgTxt("Align anat to func ... done.")

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _show_proc_progress(self, proc, progress_bar=None, msgTxt='', desc='',
                            ETA=None, total_ETA=None):
        if progress_bar is not None:
            if len(msgTxt):
                progress_bar.set_msgTxt(msgTxt)
                QtWidgets.QApplication.processEvents()
            if len(desc):
                progress_bar.add_desc(f"+++ {desc} ...\n")
                QtWidgets.QApplication.processEvents()

            if ETA is not None:
                bar_inc = 100 * (ETA / total_ETA)
            else:
                bar_inc = None
            ret = progress_bar.proc_print_progress(
                proc, bar_inc=bar_inc, ETA=ETA)

            if not progress_bar.isVisible():
                ret = -1
        else:
            ret = self._proc_print_progress(proc)

        # Check error
        if ret != 0:
            if ret == -1:
                self.logmsg(f"Cancel {desc}")
            else:
                self.errmsg(f"Failed in {desc}")
            if hasattr(self, 'ui_procAnat_btn'):
                self.ui_procAnat_btn.setEnabled(True)

            if progress_bar is not None and progress_bar.isVisible():
                progress_bar.close()

        return ret

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _segmentation(self, work_dir, anat_orig, out_prefix,
                      progress_bar=None, ask_cmd=False, overwrite=False):
        """Segment anatomy image to extract brain
        """

        # --- Convert anat_orig to NIfTI --------------------------------------
        stem = anat_orig.stem
        ext = anat_orig.suffix
        if ext == '.gz':
            stem = Path(anat_orig.stem).stem
            ext = Path(anat_orig.stem).suffix

        if ext == '.nii':
            anat_orig_nii = anat_orig
        else:
            anat_orig_nii = work_dir / \
                ('rm_be.' + stem.replace('+orig', '') + '.nii.gz')
            if not anat_orig_nii.is_file() or overwrite:
                cmd = f"3dAFNItoNIFTI -overwrite -prefix {anat_orig_nii}"
                cmd += f" {self.anat_orig}"
                ret = self._show_cmd_progress(
                    cmd, progress_bar=progress_bar,
                    msgTxt="Brain extraction ... covert to NIfTI",
                    desc=f"Convert {Path(anat_orig).name} to NIfTI\n")

                if ret != 0:
                    return ret

        total_ETA = np.sum(list(self.proc_times.values()))

        # --- Bias Field Correction -------------------------------------------
        bc_anat_orig = work_dir / ('rm_be.bc_' + stem.replace('+orig', '') +
                                   '.nii.gz')
        if not bc_anat_orig.is_file() or overwrite:
            st = time.time()
            cmd = f"N4BiasFieldCorrection -v -d 3"
            cmd += f" -i {os.path.relpath(anat_orig_nii, work_dir)} -s 4"
            cmd += f" --output {os.path.relpath(bc_anat_orig, work_dir)}"
            proc = pexpect.spawn(cmd, cwd=str(work_dir))
            ret = self._show_proc_progress(
                proc, progress_bar,
                msgTxt="Brain extraction ... bias correction",
                desc='N4BiasFieldCorrection',
                ETA=self.proc_times["BiasCorr"], total_ETA=total_ETA)

            if ret != 0:
                return ret

            self.proc_times["BiasCorr"] = np.ceil(time.time() - st)

        # --- Segment anatomy image -------------------------------------------
        seg_f = work_dir / f"brainmask.{stem.replace('+orig', '')}.nii.gz"
        if not seg_f.is_file() or overwrite:
            st = time.time()
            Seg_cmd = Path(__file__).parent / 'dlSS' / 'dlSS.py'
            cmd = f"{Seg_cmd}"
            cmd += f" --input {os.path.relpath(bc_anat_orig, work_dir)}"
            cmd += f" --prefix {os.path.relpath(seg_f, work_dir)}"
            cmd += " --crop_stride 64 --batch_size 8 --thresh 0.5"
            if ask_cmd:
                labelTxt = 'Commdand line:'
                cmd, okflag = self._edit_command(labelTxt=labelTxt, cmdTxt=cmd)
                if not okflag:
                    return -1

            proc = pexpect.spawn(cmd, cwd=str(work_dir))
            ret = self._show_proc_progress(
                proc, progress_bar, msgTxt='Brain extraction ... dlSS',
                desc='dlSS', ETA=self.proc_times["dlSS"],
                total_ETA=total_ETA)

            if ret != 0:
                return ret

            self.proc_times["dlSS"] = np.ceil(time.time() - st)

        # --- Apply the brainmask ---------------------------------------------
        cmd = f"3dcalc -overwrite -a {bc_anat_orig} -b {seg_f}"
        cmd += f" -prefix {out_prefix} -expr 'a*step(b)'"
        ret = self._show_cmd_progress(
            cmd, progress_bar, msgTxt='Brain extraction ... Apply mask')
        if ret != 0:
            return ret

        for delf in work_dir.glob('rm_be.*'):
            delf.unlink()

        if ret != 0:
            return ret

        if progress_bar is not None:
            progress_bar.set_msgTxt("Brain extraction ... done.")
            progress_bar.add_desc(
                    f"\nSave the brain image as {out_prefix.name}.\n")
            QtWidgets.QApplication.processEvents()

        return 0

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _align_anat2epi(self, work_dir, anat_orig, func_orig, ask_cmd=False):

        # --- Align anat to func ----------------------------------------------
        anat_orig_rel = os.path.relpath(anat_orig, work_dir)
        func_orig_rel = os.path.relpath(func_orig, work_dir)
        cmd = f"align_epi_anat.py -overwrite -anat2epi -anat {anat_orig_rel}"
        cmd += f" -epi {func_orig_rel} -epi_base 0 -suffix _al_func"
        cmd += " -epi_strip 3dAutomask -anat_has_skull no"
        cmd += " -volreg off -tshift off -giant_move"
        if ask_cmd:
            labelTxt = 'Commdand line: (see '
            labelTxt += 'https://afni.nimh.nih.gov/pub/dist/doc/program_help/'
            labelTxt += 'align_epi_anat.py.html)'
            labelTxt += "\nConsider -ginormous_move or"
            labelTxt += " -partial_coverage option"
            cmd, okflag = self._edit_command(labelTxt=labelTxt, cmdTxt=cmd)
            if not okflag:
                return None

        try:
            proc = pexpect.spawn(cmd, cwd=str(work_dir))
            return proc
        except Exception as e:
            self.errmsg(str(e)+'\n')
            self.errmsg("'{}' failed.".format(cmd))
            return None

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _warp_template_ANTs(self, work_dir, anat_nii, template, ask_cmd=False,
                            nr_threads=0):
        """Warp template to (aligned) anatomy with ANTs
        """

        # Set number of thread
        if nr_threads == 0:
            nr_threads = multiprocessing.cpu_count()//2

        # Warp template to anat_nii
        anat_nii_rel = os.path.relpath(anat_nii.resolve(),
                                       work_dir.resolve())
        template_rel = os.path.relpath(Path(template).resolve(),
                                       work_dir.resolve())
        cmd = f"antsRegistrationSyNQuick.sh -d 3 -f {anat_nii_rel}"
        cmd += f" -m {template_rel} -o template2orig_ -n {nr_threads}"
        if ask_cmd:
            labelTxt = 'Commdand line: (see '
            labelTxt += 'https://github.com/stnava/alphANTs)'
            cmd, okflag = self._edit_command(labelTxt=labelTxt, cmdTxt=cmd)
            if not okflag:
                return None

        try:
            proc = pexpect.spawn(cmd, cwd=str(work_dir))
            return proc
        except Exception as e:
            self.errmsg(str(e)+'\n')
            self.errmsg("'{}' failed.".format(cmd))
            return None

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _apply_warp2ROI(self, work_dir, ROI_template, anat_ref, out_f,
                        res_mode='Linear', ask_cmd=False):
        """Apply warp to ROI on template
        """

        # --- Check warp files ---
        aff_f = work_dir / 'template2orig_0GenericAffine.mat'
        if not aff_f.is_file():
            self.errmsg(f"Not found {aff_f}\n")
            return None

        wrp_f = work_dir / 'template2orig_1Warp.nii.gz'
        if not wrp_f.is_file():
            self.errmsg(f"Not found {wrp_f}\n")
            return None

        # --- Apply warp to ROI_template ---
        anat_ref_rel = os.path.relpath(anat_ref, work_dir)
        wrp_f_rel = os.path.relpath(wrp_f, work_dir)
        aff_f_rel = os.path.relpath(aff_f, work_dir)
        out_f_rel = os.path.relpath(out_f, work_dir)
        cmd = f"antsApplyTransforms -d 3 -e 3 -i {ROI_template.absolute()}"
        cmd += f" -o {out_f_rel} -r {anat_ref_rel}"
        cmd += f" -t {wrp_f_rel} -t {aff_f_rel} -n {res_mode} --float"
        if ask_cmd:
            labelTxt = 'Commdand line: (see '
            labelTxt += 'http://manpages.org/antsapplytransforms)'
            cmd, okflag = self._edit_command(labelTxt=labelTxt, cmdTxt=cmd)
            if not okflag:
                return None

        try:
            proc = pexpect.spawn(cmd, cwd=str(work_dir))
            return proc
        except Exception as e:
            self.errmsg(str(e)+'\n')
            self.errmsg("'{}' failed.".format(cmd))
            return None

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _erode_ROI(self, work_dir, src, out_f, erode=0, ask_cmd=False):
        """Erode ROI
        """

        src = Path(os.path.relpath(src, work_dir))
        out_f = Path(os.path.relpath(out_f, work_dir))

        # --- Erode mask ---
        cmd = f"3dmask_tool -overwrite -input {src} -dilate_input {-erode}"
        cmd += f" -prefix {out_f}"
        if ask_cmd:
            labelTxt = 'Commdand line: (see '
            labelTxt += 'https://afni.nimh.nih.gov/pub/dist/doc/'
            labelTxt += 'program_help/3dmask_tool.html)'
            cmd, okflag = self._edit_command(labelTxt=labelTxt, cmdTxt=cmd)
            if not okflag:
                return None

        try:
            proc = pexpect.spawn(cmd, cwd=str(work_dir))
            return proc
        except Exception as e:
            self.errmsg(str(e)+'\n')
            self.errmsg("'{}' failed.".format(cmd))
            return None

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _resample_ROI(self, work_dir, src, func_ref, out_f, clip=None,
                      res_mode='NN', ask_cmd=False, progress_bar=None):
        """Threshold and resample warped ROI
        """

        src = Path(os.path.relpath(src, work_dir))
        func_ref = Path(os.path.relpath(func_ref, work_dir))
        out_f = Path(os.path.relpath(out_f, work_dir))

        cmd = ''

        if clip is not None:
            cmd += f"3dfractionize -overwrite -template {func_ref}"
            cmd += f" -input {src} -prefix {out_f} -clip {clip}"
        else:
            cmd += f"3dresample -overwrite -master {func_ref}"
            cmd += f" -input {src} -prefix {out_f} -rmode {res_mode}"

        if ask_cmd:
            labelTxt = 'Commdand line: (see '
            labelTxt += 'https://afni.nimh.nih.gov/pub/dist/doc/'
            if clip is not None:
                labelTxt += 'program_help/3dfractionize.html)'
            else:
                labelTxt += 'program_help/3dresample.html)'
            cmd, okflag = self._edit_command(labelTxt=labelTxt, cmdTxt=cmd)
            if not okflag:
                return None

        try:
            proc = pexpect.spawn(cmd, cwd=str(work_dir))
            return proc
        except Exception as e:
            self.errmsg(str(e)+'\n')
            self.errmsg("'{}' failed.".format(cmd))
            return None

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def proc_anat(self, ask_cmd=False, nr_threads=0, overwrite=False,
                  progress_dlg=False):
        """Process anatomy image to make functional masks
        1.  Bias correction and segmente anat_orig to extract brain using
            N4BiasFieldCorrection and dlSeg
        2.  Align brain anatomy image to function image with align_anat_epi.py
            in AFNI.
        3.  Warp template to aligned brain anatomy with ANTs.
        4.  Apply the warp to WM, Vent, and ROI masks.

        Parameters
        ----------
        ask_cmd : bool, optional

        nr_threads : TYPE, optional
            DESCRIPTION. The default is 0.
        overwrite : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        None.

        """

        # --- Set parameters --------------------------------------------------
        for attr in ('Template', 'func_orig', 'anat_orig'):
            if not hasattr(self, attr) or getattr(self, attr) is None:
                errmsg = f"\n{attr} is not set.\n"
                self.errmsg(errmsg)
                return

            fpath = Path(getattr(self, attr))
            fpath = fpath.parent / \
                re.sub("\'", '', re.sub(r"\[.+\]", '', fpath.name))

            if not Path(fpath).is_file():
                errmsg = f"\nNot found {attr}:{getattr(self, attr)}.\n"
                self.errmsg(errmsg)
                return

            self.set_param(attr, Path(getattr(self, attr)))

        # Set the number of thread
        if nr_threads == 0:
            nr_threads = multiprocessing.cpu_count()//2 - 2
        else:
            nr_threads = min(multiprocessing.cpu_count()//2 - 2, nr_threads)
        nr_threads = max(int(nr_threads), 1)

        # Disable warp button
        if hasattr(self, 'ui_procAnat_btn'):
            self.ui_procAnat_btn.setEnabled(False)

        # Check shift (overwrite) and ctrl (ask_cmd) key press
        modifiers = QtWidgets.QApplication.keyboardModifiers()
        if modifiers == QtCore.Qt.ShiftModifier:
            overwrite = True
        elif modifiers == QtCore.Qt.ControlModifier:
            ask_cmd = True
        elif modifiers == (QtCore.Qt.ControlModifier |
                           QtCore.Qt.ShiftModifier):
            overwrite = True
            ask_cmd = True

        # progress bar
        if progress_dlg:
            progress_bar = DlgProgressBar(
                title=f"Process anatomy image", modal=False,
                parent=self.main_win, st_time=time.time())
            progress_bar.set_value(0)
        else:
            progress_bar = None

        # Copy func_orig as vr_base_external
        if self.func_orig.stem != 'vr_base_external+orig':
            srcf = self.func_orig
            if re.search(r"\[\d+\]", self.func_orig.name) is None:
                srcf = f"'{srcf}[0]'"
            dst_f = self.func_orig.parent / 'vr_base_external+orig.BRIK'
            cmd = f"3dbucket -overwrite -prefix {dst_f} {srcf}"
            ret = self._show_cmd_progress(cmd, progress_bar)
            if ret != 0:
                return

            self.set_param('func_orig', dst_f)

        total_ETA = np.sum(list(self.proc_times.values()))
        work_dir = self.work_dir
        st0 = time.time()

        # --- 1. Bias correction and segmente anat_orig to extract brain ------
        self.brain_anat_orig = work_dir / ('brain_' + self.anat_orig.name)
        if self.brain_anat_orig.suffix == '.gz':
            brikf = self.brain_anat_orig.parent / self.brain_anat_orig.stem
        else:
            brikf = Path(str(self.brain_anat_orig) + '.gz')
        if not self.brain_anat_orig.is_file() and brikf.is_file():
            self.brain_anat_orig = brikf

        if not self.brain_anat_orig.is_file() or overwrite:
            # Print job description
            descStr = '+' * 60 + '\n'
            descStr += f"+++ Brain extraction\n"
            if progress_bar is not None:
                progress_bar.add_desc(descStr)
                QtWidgets.QApplication.processEvents()
            else:
                sys.stdout.write(descStr)

            # Run the process
            ret = self._segmentation(
                    work_dir, self.anat_orig, self.brain_anat_orig,
                    progress_bar=progress_bar, ask_cmd=ask_cmd,
                    overwrite=overwrite)

            # Check error
            if ret != 0:
                if ret == -1:
                    self.logmsg("Cancel process anatomy")
                else:
                    self.errmsg("Failed in brain extaction")
                if hasattr(self, 'ui_procAnat_btn'):
                    self.ui_procAnat_btn.setEnabled(True)
                if progress_bar is not None and progress_bar.isVisible():
                    progress_bar.close()
                return
        else:
            if progress_bar is not None:
                bar_inc = (self.proc_times["BiasCorr"]
                           + self.proc_times["dlSeg"]) / total_ETA * 100
                progress_bar.set_value(bar_inc)
                progress_bar.add_desc("+++ Brain extraction\n")
                progress_bar.add_desc(
                    f"Use existing file: {self.brain_anat_orig}\n")

        # check .gz in the filename
        if not self.brain_anat_orig.is_file():
            brikgz = Path(str(self.brain_anat_orig) + '.gz')
            if brikgz.is_file():
                self.brain_anat_orig = brikgz

        # --- 2. Align anat to function ---------------------------------------
        if '+orig' in self.brain_anat_orig.name:
            alAnat = work_dir / \
                self.brain_anat_orig.name.replace('+orig', '_al_func+orig')
        else:
            suff = self.brain_anat_orig.suffix
            if suff == '.gz':
                suff = Path(self.brain_anat_orig.stem).suffix + suff

            alAnat = work_dir / \
                (self.brain_anat_orig.name.replace(suff, '') +
                 '_al_func+orig.BRIK')

        if alAnat.suffix == '.gz':
            brikf = alAnat.parent / alAnat.stem
        else:
            brikf = Path(str(alAnat) + '.gz')
        if not alAnat.is_file() and brikf.is_file():
            alAnat = brikf

        if not alAnat.is_file() or overwrite:
            st = time.time()
            proc = self._align_anat2epi(work_dir, self.brain_anat_orig,
                                        self.func_orig, ask_cmd=ask_cmd)
            if proc is None:
                if progress_bar is not None and progress_bar.isVisible():
                    progress_bar.close()
                if hasattr(self, 'ui_procAnat_btn'):
                    self.ui_procAnat_btn.setEnabled(True)
                return

            ret = self._show_proc_progress(
                proc, progress_bar, msgTxt='Align anat to func ...',
                desc='Align anat func', ETA=self.proc_times["AlAnat"],
                total_ETA=total_ETA)

            if ret != 0:
                return

            self.proc_times["AlAnat"] = np.ceil(time.time() - st)

        else:
            if progress_bar is not None:
                bar_inc = self.proc_times["AlAnat"] / total_ETA * 100
                bar_val0 = progress_bar.progBar.value()
                progress_bar.set_value(bar_val0+bar_inc)
                progress_bar.add_desc("+++ Align anat func\n")
                progress_bar.add_desc(f"Use existing file: {self.alAnat}\n")

        self.set_param('alAnat', alAnat)

        # Convert aligned anatomy to NIfTI
        alAnat_f_stem = alAnat.stem.replace('+orig', '')
        alAnat_nii = self.func_orig.parent / (alAnat_f_stem + '.nii.gz')
        if not alAnat_nii.is_file():
            cmd = f"3dAFNItoNIFTI -overwrite -prefix {alAnat_nii}"
            cmd += f" {self.alAnat}"
            ret = self._show_cmd_progress(cmd, progress_bar,
                                          desc='Convert alAnat to NIfTI')
            if ret != 0:
                return

        if progress_bar is not None:
            progress_bar.set_msgTxt("Align anat to func ... done.")

        # --- 3. Warp template to alAnat --------------------------------------
        aff_f = work_dir / 'template2orig_0GenericAffine.mat'
        wrp_f = work_dir / 'template2orig_1Warp.nii.gz'
        if not aff_f.is_file() or not wrp_f.is_file() or overwrite:
            st = time.time()
            proc = self._warp_template_ANTs(
                work_dir, alAnat_nii, self.Template, ask_cmd=ask_cmd,
                nr_threads=nr_threads)
            if proc is None:
                if progress_bar is not None and progress_bar.isVisible():
                    progress_bar.close()
                return

            ret = self._show_proc_progress(
                proc, progress_bar, msgTxt="ANTs registraion ...",
                desc='ANTs registraion', ETA=self.proc_times["ANTs"],
                total_ETA=total_ETA)
            if ret != 0:
                return

            self.proc_times["ANTs"] = np.ceil(time.time() - st)

        else:
            if progress_bar is not None:
                bar_inc = self.proc_times["ANTs"] / total_ETA * 100
                bar_val0 = progress_bar.progBar.value()
                progress_bar.set_value(bar_val0+bar_inc)
                progress_bar.add_desc("+++ ANTs registraion\n")
                progress_bar.add_desc(f"Use existing file: {wrp_f}\n")

        if progress_bar is not None:
            progress_bar.set_msgTxt("ANTs registraion ... done.")

        # --- 4. Apply warp and resample to ROIs defined in the template space
        for roi_template in ('WM_template', 'Vent_template', 'ROI_template'):
            roi_template_f = getattr(self, roi_template)
            if roi_template_f is None or not Path(roi_template_f).is_file():
                continue

            out_f = work_dir / \
                Path(roi_template_f).name.replace('.nii', '_inOrigFunc.nii')
            if not out_f.is_file() or overwrite:
                st = time.time()

                # -- Apply warp --
                out_f0 = out_f.parent / ('rm.' + out_f.name)
                proc = self._apply_warp2ROI(
                        work_dir, roi_template_f, alAnat_nii, out_f0,
                        res_mode='NearestNeighbor')
                if proc is None:
                    if progress_bar is not None and progress_bar.isVisible():
                        progress_bar.close()
                    if hasattr(self, 'ui_procAnat_btn'):
                        self.ui_procAnat_btn.setEnabled(True)
                    return
                ret = self._show_proc_progress(
                    proc, progress_bar,
                    msgTxt=f"Warp ands resample {roi_template} ...",
                    desc=f"Apply warp to {roi_template}")
                if ret != 0:
                    return

                # -- Erode the warped mask ---
                if roi_template in ('WM_template', 'Vent_template'):
                    out_f1 = out_f0.parent / ('rm.1.' + out_f0.name)
                    if roi_template == 'WM_template':
                        erode = 2
                    elif roi_template == 'Vent_template':
                        erode = 1

                    proc = self._erode_ROI(work_dir, out_f0, out_f1,
                                           erode=erode, ask_cmd=ask_cmd)
                    if proc is None:
                        if progress_bar is not None \
                                and progress_bar.isVisible():
                            progress_bar.close()
                        if hasattr(self, 'ui_procAnat_btn'):
                            self.ui_procAnat_btn.setEnabled(True)
                        return

                    ret = self._show_proc_progress(
                        proc, progress_bar,
                        desc=f"Erode warped {roi_template}")
                    if ret != 0:
                        return

                elif roi_template == 'ROI_template':
                    out_f1 = out_f0

                # -- Resample in func_orig ---
                proc = self._resample_ROI(
                        work_dir, out_f1, self.func_orig, out_f, res_mode='NN',
                        ask_cmd=ask_cmd, progress_bar=progress_bar)
                if proc is None:
                    if progress_bar is not None and progress_bar.isVisible():
                        progress_bar.close()
                    if hasattr(self, 'ui_procAnat_btn'):
                        self.ui_procAnat_btn.setEnabled(True)
                    return
                ret = self._show_proc_progress(
                    proc, progress_bar,
                    msgTxt=f"Warp ands resample {roi_template} ...",
                    desc=f"Resample warped {roi_template}",
                    ETA=self.proc_times[f"ApplyWarp_{roi_template}"],
                    total_ETA=total_ETA)
                if ret != 0:
                    return

                self.proc_times[f"ApplyWarp_{roi_template}"] = \
                    np.ceil(time.time() - st)

            else:
                if progress_bar is not None:
                    bar_inc = (self.proc_times[f"ApplyWarp_{roi_template}"]
                               / total_ETA) * 100
                    progress_bar.set_value(bar_inc)
                    progress_bar.add_desc(
                        f"+++ Apply warp to {roi_template}\n")
                    progress_bar.add_desc(f"Use existing file: {out_f}\n")

            # Set roi_orig attribute
            roi_orig = roi_template.replace('template', 'orig')
            self.set_param(roi_orig, out_f)

            if progress_bar is not None:
                progress_bar.set_msgTxt(
                    f"Apply warp to {roi_template} ... done.")

            # Clean rm.* files
            for rmf in work_dir.glob('rm*'):
                rmf.unlink()

        # --- End -------------------------------------------------------------
        if progress_bar is not None:
            etstr = str(datetime.timedelta(seconds=time.time()-st0))
            etstr = ':'.join(etstr.split('.')[0].split(':')[1:])
            progress_bar.add_desc("Done (took {})".format(etstr))
            progress_bar.set_value(100)
            progress_bar.btnCancel.setText('Close')
            progress_bar.setWindowTitle('Process anatomy image, Done.')

        # Enable warp button
        if hasattr(self, 'ui_procAnat_btn'):
            self.ui_procAnat_btn.setEnabled(True)

        return

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def make_function_image_mask(self, func_src=None, ref_vi=0,
                                 anat_src=None, ask_cmd=False, overwrite=True):
        """Make function image mask for RTP
        3dAutomask is used to make a function image mask.
        If anatomical image is also provided, union mask of function automask
        and anatomy zero-out mask is made.
        The created files is saved in the same directory as the function image.

        Options
        -------
        func_src: string
            Function image file
        ref_vi : int
            reference volume index for func_orig
        anat_src: string (optional)
            Anatomy image file. This must be aligned to self.func_orig
        ask_cmd: bool (optional)
            Enable command editor
        overwrite: bool (optional)
            Overwrite existing file

        Return
        ------
        RTP_mask : stirng or Path object
            Mask for real-time processing
        GSR_mask : stirng or Path object
            Mask for global signal regression

        """

        cmd = ''

        if func_src is not None:
            fbase = re.sub(r'\+.*', '', func_src.name)
            func_mask = func_src.parent / f"automask_{fbase}.nii.gz"
            cmd += f"3dAutomask -overwrite -prefix {func_mask} {func_src}"
            if len(nib.load(func_src).shape) > 3 and \
                    nib.load(func_src).shape[-1] > 1:
                cmd += f"'[{ref_vi}]'; "
            else:
                cmd += "; "

        if anat_src is not None and anat_src.is_file():
            fbase = re.sub(r'\+.*', '', anat_src.name)
            anat_mask = anat_src.parent / f"anatmask_{fbase}.nii.gz"
            temp_out = anat_src.parent / 'rm.anat_mask_tmp.nii.gz'
            cmd += f"3dmask_tool -overwrite -input {anat_src}"
            cmd += f" -prefix {temp_out} -frac 0.0 -fill_holes; "
            cmd += f"3dresample -overwrite -rmode NN -master {func_src}"
            cmd += f" -prefix {anat_mask} -input {temp_out}; rm {temp_out}; "

        if anat_src is not None and anat_src.is_file():
            if func_src is not None:
                RTP_mask = func_src.parent / f"RTP_mask.nii.gz"
                cmd += f"3dmask_tool -overwrite"
                cmd += f" -input {anat_mask} {func_mask}"
                cmd += f" -prefix {RTP_mask} -frac 0.0; "
            else:
                RTP_mask = anat_mask
        elif func_src is not None:
            RTP_mask = func_mask
        else:
            RTP_mask = None

        # Make GSR_mask
        if anat_src is not None and anat_src.is_file() \
                and func_src is not None:
            GSR_mask = func_src.parent / 'GSR_mask.nii.gz'
            cmd += f"3dmask_tool -overwrite -input {func_mask} {anat_mask}"
            cmd += f" -prefix {GSR_mask} -frac 1.0"
        else:
            GSR_mask = RTP_mask

        # Check existing file
        if RTP_mask.is_file() and GSR_mask.is_file() and not overwrite:
            return RTP_mask, GSR_mask

        if ask_cmd:
            labelTxt = 'Commdand line: (see '
            labelTxt += 'https://afni.nimh.nih.gov/pub/dist/doc/program_help/'
            labelTxt += '3dAutomask.html)'
            cmd, okflag = self._edit_command(labelTxt=labelTxt, cmdTxt=cmd)
            if not okflag:
                return -1, -1

        try:
            ostr = subprocess.check_output(cmd, stderr=subprocess.STDOUT,
                                           shell=True)
            self.logmsg(ostr.decode())
        except Exception as e:
            self.errmsg(str(e)+'\n')
            self.errmsg("'{}' failed.".format(cmd))
            return None, None

        return RTP_mask, GSR_mask

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def make_RTP_mask_ui(self):

        # --- Set parameters --------------------------------------------------
        # check file
        func_src = Path(self.func_orig)
        if not func_src.is_file():
            errmsg = f"No function source file: {func_src}"
            self.errmsg(errmsg)
            return

        # anat_src should be skull-stripped image aligned to func_src
        anat_src = Path(self.alAnat)

        # Disable create mask button
        self.ui_makeRTPMask_btn.setEnabled(False)
        QtWidgets.QApplication.processEvents()

        # Disable create mask button
        self.ui_makeRTPMask_btn.setEnabled(False)
        QtWidgets.QApplication.processEvents()

        overwrite = False
        ask_cmd = False

        # Check shift (overwrite) and ctrl (ask_cmd)
        modifiers = QtWidgets.QApplication.keyboardModifiers()
        if modifiers == QtCore.Qt.ShiftModifier:
            overwrite = True
        elif modifiers == QtCore.Qt.ControlModifier:
            ask_cmd = True
        elif modifiers == (QtCore.Qt.ControlModifier |
                           QtCore.Qt.ShiftModifier):
            overwrite = True
            ask_cmd = True

        # --- Run -------------------------------------------------------------
        RTP_mask, GSR_mask = self.make_function_image_mask(
                func_src=func_src, anat_src=anat_src, ask_cmd=ask_cmd,
                overwrite=overwrite)
        if RTP_mask is -1:
            # ask_cmd is canceled
            return
        elif RTP_mask is None:
            self.errmsg("Failed to make RTP mask")
            self.ui_makeRTPMask_btn.setEnabled(True)
            return

        self.set_param('RTP_mask', RTP_mask)
        self.set_param('GSR_mask', GSR_mask)

        # Message dialog
        QtWidgets.QMessageBox.information(
                self.main_win, 'RTP and GSR masks are created',
                f"RTP_mask: {self.RTP_mask}\nGSR_mask: {self.GSR_mask}")

        # Enable create mask button
        self.ui_makeRTPMask_btn.setEnabled(True)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def check_onAFNI(self, base, ovl):
        watch_dir = self.rtp_objs['WATCH'].watch_dir

        # Set underlay and overlay image file
        if base == 'anat':
            base_img = self.alAnat
        elif base == 'func':
            base_img = self.func_orig

        if ovl == 'func':
            ovl_img = self.func_orig
        elif ovl == 'wm':
            ovl_img = self.WM_orig
        elif ovl == 'vent':
            ovl_img = self.Vent_orig
        elif ovl == 'roi':
            ovl_img = self.ROI_orig
        elif ovl == 'RTPmask':
            ovl_img = self.RTP_mask
        elif ovl == 'GSRmask':
            ovl_img = self.GSR_mask

        if not Path(base_img).is_file() or not Path(ovl_img).is_file():
            return

        # Get volume shape to adjust window size
        baseWinSize = 480  # height of axial image window
        bimg = nib.load(base_img)
        vshape = bimg.shape[:3] * np.abs(np.diag(bimg.affine))[:3]
        wh_ax = [int(baseWinSize*vshape[0]//vshape[1]), baseWinSize]
        wh_sg = [baseWinSize, int(baseWinSize*vshape[2]//vshape[1])]
        wh_cr = [int(baseWinSize*vshape[0]//vshape[1]),
                 int(baseWinSize*vshape[2]//vshape[1])]

        # Get cursor position
        if ovl in ['roi', 'wm', 'vent', 'mask']:
            ovl_v = nib.load(ovl_img).get_data()
            if ovl_v.ndim > 3:
                ovl_v = ovl_v[:, :, :, 0]
            ijk = np.mean(np.argwhere(ovl_v != 0), axis=0)
            ijk = np.concatenate([ijk, [1]])
            xyz = np.dot(nib.load(ovl_img).affine, ijk)[:3]

        # Check if afni is ready
        pret = subprocess.check_output("ps ax| grep afni", shell=True)
        procs = [ll for ll in pret.decode().rstrip().split('\n')
                 if 'grep afni' not in ll]
        if len(procs) == 0:
            boot_afni(self.main_win)

        # Run plugout_drive to drive afni
        cmd = 'plugout_drive'
        cmd += " -com 'SWITCH_SESSION {}'".format(os.path.basename(watch_dir))
        cmd += " -com 'RESCAN_THIS'"
        cmd += " -com 'SWITCH_UNDERLAY {}'".format(os.path.basename(base_img))
        cmd += " -com 'SWITCH_OVERLAY {}'" .format(os.path.basename(ovl_img))
        cmd += " -com 'SEE_OVERLAY +'"
        cmd += " -com 'OPEN_WINDOW A.axialimage"
        cmd += " geom={}x{}+0+0 opacity=6'".format(*wh_ax)
        cmd += " -com 'OPEN_WINDOW A.sagittalimage"
        cmd += " geom={}x{}+{}+0 opacity=6'".format(*wh_sg, wh_ax[0])
        cmd += " -com 'OPEN_WINDOW A.coronalimage"
        cmd += " geom={}x{}+{}+0 opacity=6'".format(*wh_cr, wh_ax[0]+wh_sg[0])
        if ovl in ['roi', 'wm', 'vent', 'mask']:
            cmd += " -com 'SET_SPM_XYZ {} {} {}'".format(*xyz)
        cmd += " -quit"
        subprocess.run(cmd, shell=True)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    class PlotROISignal(QtCore.QObject):
        finished = QtCore.pyqtSignal()

        def __init__(self, root, num_ROIs=1, roi_labels=[], main_win=None):
            super().__init__()

            self.root = root
            self.main_win = main_win
            self.abort = False

            # Initialize figure
            plt_winname = 'ROI signal'
            self.plt_win = MatplotlibWindow()
            self.plt_win.setWindowTitle(plt_winname)

            # set position
            if main_win is not None:
                main_geom = main_win.geometry()
                x = main_geom.x() + main_geom.width() + 10
                y = main_geom.y() + 735
            else:
                x, y = (0, 0)
            win_height = 80 + 70 * num_ROIs
            self.plt_win.setGeometry(x, y, 500, win_height)

            # Set axis
            self.roi_labels = roi_labels
            self._axes = self.plt_win.canvas.figure.subplots(num_ROIs, 1)
            if num_ROIs == 1:
                self._axes = [self._axes]

            self.plt_win.canvas.figure.subplots_adjust(
                    left=0.15, bottom=0.18, right=0.95, top=0.96, hspace=0.35)
            self._ln = []
            color_cycle = plt.get_cmap("tab10")

            for ii, ax in enumerate(self._axes):
                if len(self.roi_labels) > ii:
                    ax.set_ylabel(self.roi_labels[ii])
                ax.set_xlim(0, 10)
                self._ln.append(ax.plot(0, 0, color=color_cycle(ii+1)))

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
                    # Plot signal
                    plt_xi = self.root.plt_xi.copy()
                    plt_roi_sig = self.root.plt_roi_sig
                    for ii, ax in enumerate(self._axes):
                        ll = min(len(plt_xi), len(plt_roi_sig[ii]))
                        if ll == 0:
                            continue

                        y = plt_roi_sig[ii][:ll]
                        self._ln[ii][0].set_data(plt_xi[:ll], y)

                        # Adjust y scale
                        if np.sum(~np.isnan(y)) > 2.5:
                            sd = np.nanstd(y)
                            ax.set_ylim([-2.5*sd, 2.5*sd])
                        else:
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

            if hasattr(self.root, 'ui_chbShowROISig'):
                self.root.ui_chbShowROISig.setCheckState(0)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def open_ROISig_plot(self, num_ROIs=1, roi_labels=[]):
        if hasattr(self, 'thPltROISig') and self.thPltROISig.isRunning():
            return

        self.thPltROISig = QtCore.QThread()
        self.pltROISig = RTP_APP.PlotROISignal(self, num_ROIs=num_ROIs,
                                               roi_labels=roi_labels,
                                               main_win=self.main_win)
        self.pltROISig.moveToThread(self.thPltROISig)
        self.thPltROISig.started.connect(self.pltROISig.run)
        self.pltROISig.finished.connect(self.thPltROISig.quit)
        self.thPltROISig.start()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def close_ROISig_plot(self):
        if hasattr(self, 'thPltROISig') and self.thPltROISig.isRunning():
            self.pltROISig.abort = True
            if not self.thPltROISig.wait(1):
                self.pltROISig.finished.emit()
                self.thPltROISig.wait()

            del self.thPltROISig

        if hasattr(self, 'pltROISig'):
            del self.pltROISig

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def show_ROIsig_chk(self, state):
        if state == 2:
            self.open_ROISig_plot()
        else:
            self.close_ROISig_plot()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    class WAIT_ONSET(QtCore.QObject):
        """Send scan start message to an external application
        """

        finished = QtCore.pyqtSignal()

        def __init__(self, onsetObj, com_extApp=None):
            super().__init__()
            self.scanning = False
            self.onsetObj = onsetObj
            self.com_extApp = com_extApp
            self.abort = False

        def run(self):
            while not self.onsetObj.scanning and not self.abort:
                time.sleep(0.001)

            if self.abort:
                self.finished.emit()
                return

            onset_time = self.onsetObj.scan_onset

            if self.com_extApp is not None:
                # Send message to self.com_expWin
                msg = "scan start at {};".format(onset_time)
                self.com_extApp.send(msg.encode())

            self.finished.emit()
            return

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def setup_RTP(self):
        """ Setup RTP modules
        """

        show_progress = (self.main_win is not None)

        if show_progress:
            # progress bar
            progress_bar = DlgProgressBar(title="Setting up experiment ...",
                                          parent=self.main_win)
            progress = 0

        # --- Set real-time proc parameters -----------------------------------
        if self.enable_RTP:
            for proc in (['WATCH', 'TSHIFT', 'VOLREG', 'SMOOTH', 'REGRESS']):
                if show_progress:
                    progress += 100//7
                    progress_bar.set_value(progress)
                    progress_bar.add_desc("Set {} parameters\n".format(proc))

                if proc not in self.rtp_objs:
                    continue

                pobj = self.rtp_objs[proc]
                if pobj.enabled:
                    if proc == 'WATCH':
                        pobj.set_param('clean_files')

                    if proc == 'TSHIFT':
                        if not Path(self.func_orig).is_file():
                            self.errmsg("Not found 'Reference function image'"
                                        f" {self.func_orig}.")
                            if show_progress and progress_bar.isVisible():
                                progress_bar.close()
                                return

                        Nslice = nib.load(self.func_orig).shape[2]
                        if len(pobj.slice_timing) != Nslice:
                            if self.main_win is not None:
                                ret = pobj.set_param(
                                    'slice_timing_from_sample')
                                if ret is not None and ret == -1:
                                    # Canceled
                                    if show_progress and \
                                            progress_bar.isVisible():
                                        progress_bar.close()
                                    return
                            else:
                                self.errmsg(
                                    "slice_timing is not set in TSHIFT.")
                                return

                        self.max_watch_wait = self.rtp_objs['TSHIFT'].TR * 3

                    elif proc == 'VOLREG':
                        if Path(self.func_orig).is_file():
                            pobj.set_param('ref_vol', self.func_orig)
                        else:
                            ret = pobj.set_param('ref_vol', 'external')
                            if ret is not None and ret == -1:
                                # Canceled
                                if show_progress and progress_bar.isVisible():
                                    progress_bar.close()
                                return

                    elif proc == 'SMOOTH':
                        if Path(self.RTP_mask).is_file():
                            pobj.set_param('mask_file', self.RTP_mask)
                        else:
                            ret = pobj.set_param('mask_file', 'external')
                            if ret is not None and ret == -1:
                                # Canceled
                                if show_progress and progress_bar.isVisible():
                                    progress_bar.close()
                                return

                    elif proc == 'REGRESS':
                        pobj.TR = self.rtp_objs['TSHIFT'].TR
                        pobj.tshift = self.rtp_objs['TSHIFT'].ref_time
                        if pobj.mot_reg != 'None':
                            if not self.rtp_objs['VOLREG'].enabled:
                                pobj.ui_motReg_cmbBx.setCurrentText('none')
                                errmsg = 'VOLREG is not enabled.'
                                errmsg += 'Motion regressor is set to none.'
                                self.rtp_objs['VOLREG'].errmsg(errmsg)
                            else:
                                pobj.volreg = self.rtp_objs['VOLREG']

                        if pobj.phys_reg != 'None':
                            pobj.physio = self.rtp_objs['PHYSIO']
                            self.rtp_objs['PHYSIO'].rtp_retrots = \
                                self.rtp_objs['RETROTS']

                        if pobj.GS_reg:
                            if Path(self.GSR_mask).is_file():
                                pobj.set_param('GS_mask', self.GSR_mask)
                            else:
                                ret = pobj.set_param(
                                    'GS_mask', self.watch_dir,
                                    pobj.ui_GS_mask_lnEd.setText)
                                if ret == -1:
                                    if show_progress and \
                                            progress_bar.isVisible():
                                        progress_bar.close()
                                    return

                        if pobj.WM_reg:
                            if Path(self.WM_orig).is_file():
                                pobj.set_param('WM_mask', self.WM_orig)
                            else:
                                ret = pobj.set_param(
                                    'WM_mask', self.watch_dir,
                                    pobj.ui_WM_mask_lnEd.setText)
                                if ret == -1:
                                    if show_progress and \
                                            progress_bar.isVisible():
                                        progress_bar.close()
                                    return

                        if pobj.Vent_reg:
                            if Path(self.Vent_orig).is_file():
                                pobj.set_param('Vent_mask', self.Vent_orig)
                            else:
                                ret = pobj.set_param(
                                    'Vent_mask', self.watch_dir,
                                    pobj.ui_Vent_mask_lnEd.setText)
                                if ret == -1:
                                    if show_progress and \
                                            progress_bar.isVisible():
                                        progress_bar.close()
                                    return

                if show_progress and not progress_bar.isVisible():
                    self.logmsg("Cancel setup experiment")
                    return
        else:
            if show_progress:
                progress += 100//7 * 5
                progress_bar.set_value(progress)

        # --- Check if afni is ready ------------------------------------------
        if show_progress:
            progress_bar.add_desc("Check afni is running\n")

        pret = subprocess.check_output("ps ax| grep afni", shell=True)
        procs = [ll for ll in pret.decode().rstrip().split('\n')
                 if 'grep afni' not in ll and 'RTafni' not in ll]
        if len(procs) == 0:
            boot_afni(self.main_win)

        if show_progress:
            progress += 100//7
            progress_bar.set_value(progress)

        # --- Boot an external application ------------------------------------
        if self.ext_app is not None:
            if show_progress:
                progress_bar.add_desc(
                    f"Setting up external application: {self.ext_app}\n")

            ret = self.boot_external_app()
            if ret != 0:
                if show_progress and progress_bar.isVisible():
                    progress_bar.close()
                return

            if show_progress and not progress_bar.isVisible():
                self.logmsg("Cancel at application setup")
                return

        if show_progress:
            progress += 100//7
            progress_bar.set_value(progress)

        # --- End -------------------------------------------------------------
        if show_progress:
            progress = 100
            progress_bar.set_value(progress)
            progress_bar.add_desc("Done")

        # Set ready button
        self.ui_readyEnd_btn.setEnabled(True)
        self.ui_readyEnd_btn.clicked.disconnect()
        self.ui_readyEnd_btn.setText('Ready to start scan')
        self.ui_readyEnd_btn.clicked.connect(self.ready_to_run)
        self.ui_readyEnd_btn.setStyleSheet(
            "background-color: rgb(246,136,38);"
            "height: 30px;")

        if show_progress:
            progress_bar.close()

        if self.main_win is not None:
            self.main_win.options_tab.setCurrentIndex(2)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def ready_to_run(self):
        """ Ready to start scan
        """

        # --- Ready RTP -------------------------------------------------------
        if self.enable_RTP > 0:
            # connect RTP modules
            proc_chain = self.rtp_objs['WATCH']
            last_proc = proc_chain
            for proc in (['TSHIFT', 'VOLREG', 'SMOOTH', 'REGRESS']):
                if proc not in self.rtp_objs:
                    continue

                pobj = self.rtp_objs[proc]
                if pobj.enabled:
                    last_proc.next_proc = pobj
                    if proc == 'REGRESS':
                        if pobj.GS_reg or pobj.WM_reg or pobj.Vent_reg:
                            if self.rtp_objs['VOLREG'].enabled:
                                pobj.mask_src_proc = self.rtp_objs['VOLREG']
                            elif self.rtp_objs['TSHIFT'].enabled:
                                pobj.mask_src_proc = self.rtp_objs['TSHIFT']
                            else:
                                pobj.mask_src_proc = self.rtp_objs['WATCH']

                    last_proc = pobj

            last_proc.save_proc = True
            last_proc.next_proc = self  # self is the last process

            # Show plot windows
            if self.main_win is not None:
                if self.rtp_objs['VOLREG'].enabled:
                    self.main_win.chbShowMotion.setCheckState(2)

                if self.rtp_objs['REGRESS'].enabled and \
                        self.rtp_objs['REGRESS'].phys_reg != 'None':
                    # Start physio recording
                    self.main_win.chbRecPhysio.setCheckState(2)
                    self.main_win.chbShowPhysio.setCheckState(2)

                if hasattr(self, 'ui_showROISig_cbx'):
                    # Open ROI signal plot
                    self.ui_showROISig_cbx.setCheckState(2)

            else:
                if self.rtp_objs['REGRESS'].enabled and \
                        self.rtp_objs['REGRESS'].phys_reg != 'None':
                    self.rtp_objs['PHYSIO'].start_recording()

            # Reset process chain status
            proc_chain.reset()

            # Ready process sequence: proc_ready calls its child's proc_ready
            if not proc_chain.proc_ready:
                return

            # Disable ui
            if self.main_win is not None:
                objs = list(self.main_win.rtp_objs.values())
                objs += list(self.main_win.rtp_apps.values())
                for pobj in objs:
                    if not hasattr(pobj, 'ui_objs'):
                        continue

                    for ui in pobj.ui_objs:
                        ui.setEnabled(False)

                    if hasattr(pobj, 'ui_enabled_rdb'):
                        pobj.ui_enabled_rdb.setEnabled(False)

            # Set physio.wait_scan
            if 'PHYSIO' in self.rtp_objs:
                self.rtp_objs['PHYSIO'].wait_scan = True
                self.rtp_objs['PHYSIO'].scanning = False

            # Start watchdog observer
            self.rtp_objs['WATCH'].start_watching(self.rtp_objs['SCANONSET'])
            # self.rtp_objs['WATCH'].start_watching()

        # --- Ready common ----------------------------------------------------
        # Ready application
        if self.extApp_proc is not None:
            # Send 'Ready' to extApp_com
            msg = "Ready;"
            self.extApp_com.send(msg.encode())
        else:
            open(self.sig_save_file, 'w').write('Time,Index,Value\n')
            self.logmsg(f'Make the value saving file {self.sig_save_file}')

        # Start running-status checking timer
        self.chk_run_timer.start(1000)

        # Standby scan onset monitor
        self.rtp_objs['SCANONSET'].wait_scan_onset()

        # Run wait_onset thread
        if self.extApp_com is not None:
            self.th_wait_onset = QtCore.QThread()
            self.wait_onset = RTP_APP.WAIT_ONSET(self.rtp_objs['SCANONSET'],
                                                 self.extApp_com)
            self.wait_onset.moveToThread(self.th_wait_onset)
            self.th_wait_onset.started.connect(self.wait_onset.run)
            self.wait_onset.finished.connect(self.th_wait_onset.quit)
            self.th_wait_onset.start()

        # Change button text to 'End'
        self.ui_readyEnd_btn.setText(
            'Waitng for scan start ... (press to quit)')
        self.ui_readyEnd_btn.clicked.disconnect(self.ready_to_run)
        self.ui_readyEnd_btn.clicked.connect(self.end_run)
        self.ui_readyEnd_btn.setStyleSheet(
            "background-color: rgb(246,136,38);"
            "height: 30px;")

        if self.simEnabled and self.main_win is not None:
            self.main_win.options_tab.setCurrentIndex(0)
            self.ui_top_tabs.setCurrentIndex(
                self.ui_top_tabs.indexOf(self.ui_simulationTab))

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def end_run(self):

        # Abort SCANONSET if it is waiting
        self.rtp_objs['SCANONSET'].abort_waiting()
        self.rtp_objs['SCANONSET'].reset()

        # Abort WAIT_ONSET thread if it is running
        if hasattr(self, 'th_wait_onset') and \
                self.th_wait_onset.isRunning():
            self.wait_onset.abort = True
            if not self.th_wait_onset.wait(1):
                self.wait_onset.finished.emit()
                self.th_wait_onset.wait()

        # End MRI simulation if it is running
        if self.mri_sim is not None:
            self.mri_sim.run_MRI('stop')
            self.mri_sim.run_Physio('stop')
            self.ui_startSim_btn.setEnabled(True)

        # End PHYSIO scanning
        if 'PHYSIO' in self.rtp_objs:
            self.rtp_objs['PHYSIO'].wait_scan = False
            self.rtp_objs['PHYSIO'].scanning = False

            # Save data
            vol_num = self.rtp_objs['WATCH'].vol_num + 1
            if vol_num > 0:
                if self.rtp_objs['TSHIFT'].enabled:
                    TR = self.rtp_objs['TSHIFT'].TR
                elif self.rtp_objs['REGRESS'].enabled:
                    TR = self.rtp_objs['TR']
                len_sec = vol_num * TR

                scan_name = self.rtp_objs['WATCH'].scan_name
                watch_dir = self.rtp_objs['WATCH'].watch_dir
                prefix = str(Path(watch_dir) / ('{}_' + f'{scan_name}.1D'))
                self.rtp_objs['PHYSIO'].save_data(prefix=prefix,
                                                  len_sec=len_sec)

        # Send 'End' message to an external application
        if self.extApp_com is not None and self.extApp_proc.poll() is None:
            try:
                self.extApp_com.send('Abort;'.encode())
                self.extApp_com.recv(6)
            except socket.timeout:
                pass

        # Reset all process chain from WATCH
        self.rtp_objs['WATCH'].reset()

        # Enable ui
        if self.main_win is not None:
            objs = list(self.main_win.rtp_objs.values())
            objs += list(self.main_win.rtp_apps.values())
            for pobj in objs:
                if not hasattr(pobj, 'ui_objs'):
                    continue

                for ui in pobj.ui_objs:
                    ui.setEnabled(True)

                if hasattr(pobj, 'ui_enabled_rdb'):
                    pobj.ui_enabled_rdb.setEnabled(True)

        self.ui_readyEnd_btn.setText('Ready to start scan')
        self.ui_readyEnd_btn.clicked.disconnect()
        self.ui_readyEnd_btn.clicked.connect(self.ready_to_run)
        self.ui_readyEnd_btn.setEnabled(False)
        self.ui_readyEnd_btn.setStyleSheet(
            "background-color: rgb(255,201,32);"
            "height: 30px;")

        if self.main_win is not None:
            self.main_win.options_tab.setCurrentIndex(0)
            self.ui_top_tabs.setCurrentIndex(0)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def chkRunTimerEvent(self):
        """Check the running process
        """

        if self.extApp_com is not None:
            # Check a message from an external application
            self.extApp_com.settimeout(0.01)
            try:
                msg = self.extApp_com.recv(2024)
                if (msg == 0 or 'END_EXP' in msg.decode()):
                    if 'End' in self.ui_readyEnd_btn.text():
                        self.end_run()
                        return
                    else:
                        self.ui_readyEnd_btn.setEnabled(False)

                    return
            except socket.timeout:
                pass

            self.extApp_com.settimeout(None)

        # Check delay in WATCH
        if len(self.rtp_objs['WATCH'].proc_time):
            delay = time.time() - self.rtp_objs['WATCH'].proc_time[-1]
            if delay > self.max_watch_wait:
                self.end_run()
                return

        # Check scan start
        if hasattr(self, 'ui_readyEnd_btn') \
                and 'Waitng' in self.ui_readyEnd_btn.text():
            if self.rtp_objs['SCANONSET'].scanning:
                self.ui_readyEnd_btn.setText(
                    'Quit running process')
                self.ui_readyEnd_btn.setStyleSheet(
                    "background-color: rgb(237,45,135);"
                    "height: 10px;")

        # schedule the next check
        self.chk_run_timer.start(1000)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def start_scan_simulation(self):
        """Start scan simulation by copying a volume to watch_dir
        """

        if not self.simEnabled:
            return

        # --- Set simulation parameters ---------------------------------------
        # MRI data
        mri_src = self.simfMRIData
        dst_dir = self.rtp_objs['WATCH'].watch_dir
        suffix_pat = self.rtp_objs['WATCH'].watch_file_pattern
        suffix = re.sub(r'\\d\+', '{num:04d}', suffix_pat)
        if 'BRIK' in suffix_pat or 'HEAD' in suffix_pat:
            suffix = 'sim_fMRI_' + re.sub(r'\.\+\\', '+orig', suffix)
        else:
            suffix = 'sim_fMRI_' + re.sub(r'\.\+\\', '_', suffix)

        self.mri_sim = rtMRISim(mri_src, dst_dir, suffix)
        self.mri_sim._std_out = self._std_out
        self.mri_sim._err_out = self._err_out

        # Physio data
        if self.simPhysPort == 'None':
            # Disable regPhysio on rtp_regress
            if self.rtp_objs['REGRESS'].phys_reg != 'None':
                self.errmsg("Disable physio regression in rtp_regress")
                self.rtp_objs['REGRESS'].set_param('phys_reg', 'None')
            run_physio = False
        else:
            ecg_src = self.simECGData
            resp_src = self.simRespData
            physio_port = self.simPhysPort.split()[0]
            recording_rate_ms = \
                1000 / self.rtp_objs['PHYSIO'].effective_sample_freq
            samples_to_average = self.rtp_objs['PHYSIO'].samples_to_average

            recv_physio_port = re.search(r'slave:(.+)\)',
                                         self.simPhysPort).groups()[0]

            # Stop physio recording
            if self.main_win is not None:
                self.main_win.chbRecPhysio.setCheckState(0)
            else:
                self.rtp_objs['PHYSIO'].stop_recording()

            # Change port
            self.rtp_objs['PHYSIO'].update_port_list()
            self.rtp_objs['PHYSIO'].set_param('_ser_port', recv_physio_port)

            self.mri_sim.set_physio(ecg_src, resp_src, physio_port,
                                    recording_rate_ms, samples_to_average)

            # Start physio recording
            if self.main_win is not None:
                self.main_win.chbRecPhysio.setCheckState(2)
                self.main_win.chbShowPhysio.setCheckState(2)
            else:
                self.rtp_objs['PHYSIO'].start_recording()
                self.rtp_objs['PHYSIO'].open_signal_plot()

            run_physio = True

        # --- Start ---
        if run_physio:
            self.mri_sim.run_Physio('start')

        self.rtp_objs['SCANONSET'].manual_start()
        self.mri_sim.run_MRI('start')
        self.ui_startSim_btn.setEnabled(False)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_param(self, attr, val=None, reset_fn=None, echo=False):
        """
        When reset_fn is None, set_param is considered to be called from
        load_parameters function.
        """

        if attr == 'enable_RTP':
            if reset_fn is None and hasattr(self, 'ui_enabled_rdb'):
                self.ui_enableRTP_chb.setCheckState(val)

            if hasattr(self, 'ui_chbShowROISig'):
                if val == 0:
                    self.ui_chbShowROISig.setCheckState(0)

        elif attr == 'watch_dir':
            pass

        elif attr == 'save_dir':
            self.sig_save_file = Path(val) / self.sig_save_file.name

        elif attr in ('Template', 'ROI_template', 'WM_template',
                      'Vent_template', 'anat_orig', 'alAnat', 'func_orig',
                      'ROI_orig', 'WM_orig', 'Vent_orig',
                      'RTP_mask', 'GSR_mask', 'simfMRIData', 'simECGData',
                      'simRespData'):
            msglab = {'Template': 'template image',
                      'ROI_template': 'ROI mask on template',
                      'WM_template': 'white matter mask on template',
                      'Vent_template': 'ventricle mask on template',
                      'anat_orig': 'anatomy image in original space',
                      'alAnat': 'aligned anatomy image in original space',
                      'func_orig': 'function image in original space',
                      'ROI_orig': 'ROI mask in original space',
                      'WM_orig': 'white matter mask in original space',
                      'Vent_orig': 'ventricle mask in original space',
                      'RTP_mask': 'mask for real-time processing',
                      'GSR_mask': 'mask for global signal regression',
                      'simfMRIData': 'fMRI data for the simulation',
                      'simECGData': 'ECG data for the simulation',
                      'simRespData': 'Respiration data for the simulation',
                      }
            if reset_fn is not None:
                if os.path.isdir(val):
                    startdir = val
                else:
                    if 'Template' in attr:
                        startdir = os.path.dirname(__file__)
                    else:
                        startdir = self.rtp_objs['WATCH'].watch_dir

                dlgMdg = "RTP_APP: Select {}".format(msglab[attr])
                if attr in ('simECGData', 'simRespData'):
                    filt = '*.*;;*.txt;;*.1D'
                else:
                    filt = "*.BRIK* *.nii*"
                fname = self.select_file_dlg(dlgMdg, startdir, filt)
                if fname[0] == '':
                    return -1

                val = fname[0]
                if reset_fn:
                    reset_fn(val)

            else:
                if val is None or not Path(val).is_file():
                    val = ''

                if hasattr(self, "ui_{}_lnEd".format(attr)):
                    obj = getattr(self, "ui_{}_lnEd".format(attr))
                    obj.setText(str(val))

        elif attr == 'simEnabled':
            if hasattr(self, 'ui_startSim_btn'):
                self.ui_startSim_btn.setEnabled(val)

            if reset_fn is None and hasattr(self, 'ui_simEnabled_rdb'):
                self.ui_simEnabled_rdb.setChecked(val)

        elif attr == 'simPhysPort':
            if 'ptmx' in val:
                val = [desc for desc in self.simCom_descs if 'ptmx' in desc][0]

            if reset_fn is None and hasattr(self, 'ui_simPhysPort_cmbBx'):
                self.ui_simPhysPort_cmbBx.setCurrentText(val)

        elif attr == 'proc_times':
            pass

        else:
            return

        setattr(self, attr, val)
        if echo:
            print("{}.".format(self.__class__.__name__) + attr, '=',
                  getattr(self, attr))

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def ui_set_param(self):
        ui_rows = []
        self.ui_objs = []

        watch_dir = self.rtp_objs['WATCH'].watch_dir

        # enabled
        self.ui_enableRTP_chb = QtWidgets.QCheckBox("Enable RTP")
        self.ui_enableRTP_chb.setChecked(self.enable_RTP)
        self.ui_enableRTP_chb.stateChanged.connect(
                lambda x: self.set_param('enable_RTP', x,
                                         self.ui_enableRTP_chb.setCheckState))
        ui_rows.append((self.ui_enableRTP_chb, None))

        # tab
        self.ui_top_tabs = QtWidgets.QTabWidget()
        ui_rows.append((self.ui_top_tabs,))

        # --- Mask creation tab -----------------------------------------------
        maskCreationTab = QtWidgets.QWidget()
        self.ui_top_tabs.addTab(maskCreationTab, 'Mask creation')
        maskCreate_fLayout = QtWidgets.QFormLayout(maskCreationTab)

        # --- Template-Orig alignment ---
        self.ui_OrigImg_grpBx = QtWidgets.QGroupBox(
            "Alignment and warping template")
        OrigImg_gLayout = QtWidgets.QGridLayout(self.ui_OrigImg_grpBx)
        maskCreate_fLayout.addRow(self.ui_OrigImg_grpBx)

        # -- Anatomy orig image --
        ri = 0
        var_lb = QtWidgets.QLabel("Anatomy image :")
        OrigImg_gLayout.addWidget(var_lb, ri, 0)

        self.ui_anat_orig_lnEd = QtWidgets.QLineEdit()
        self.ui_anat_orig_lnEd.setReadOnly(True)
        OrigImg_gLayout.addWidget(self.ui_anat_orig_lnEd, ri, 1)

        self.ui_anat_orig_btn = QtWidgets.QPushButton('Set')
        self.ui_anat_orig_btn.clicked.connect(
                lambda: self.set_param('anat_orig', watch_dir,
                                       self.ui_anat_orig_lnEd.setText))
        OrigImg_gLayout.addWidget(self.ui_anat_orig_btn, ri, 2)

        self.ui_objs.extend([var_lb, self.ui_anat_orig_lnEd,
                             self.ui_anat_orig_btn])

        # -- function orig image --
        ri += 1
        var_lb = QtWidgets.QLabel(
                "Reference function image : ")
        OrigImg_gLayout.addWidget(var_lb, ri, 0)

        self.ui_func_orig_lnEd = QtWidgets.QLineEdit()
        OrigImg_gLayout.addWidget(self.ui_func_orig_lnEd, ri, 1)

        self.ui_func_orig_btn = QtWidgets.QPushButton('Set')
        self.ui_func_orig_btn.clicked.connect(
                lambda: self.set_param('func_orig', watch_dir,
                                       self.ui_func_orig_lnEd.setText))
        OrigImg_gLayout.addWidget(self.ui_func_orig_btn, ri, 2)

        self.ui_objs.extend([var_lb, self.ui_func_orig_lnEd,
                             self.ui_func_orig_btn])

        # -- Process anatomy button --
        self.ui_procAnat_btn = QtWidgets.QPushButton(
                'Process anatomy image\n'
                + '(+shift to overwrite; +ctrl to edit command)')
        self.ui_procAnat_btn.clicked.connect(
                partial(self.proc_anat, progress_dlg=True))
        maskCreate_fLayout.addRow(self.ui_procAnat_btn)
        self.ui_objs.append(self.ui_procAnat_btn)

        # --- check ROI buttons ---
        ROICheckWidget = QtWidgets.QWidget()
        hBoxROICheck = QtWidgets.QHBoxLayout(ROICheckWidget)
        maskCreate_fLayout.addRow(ROICheckWidget)

        self.ui_chkFuncAnat_btn = \
            QtWidgets.QPushButton('Check func-anat align')
        hBoxROICheck.addWidget(self.ui_chkFuncAnat_btn)
        self.ui_chkFuncAnat_btn.clicked.connect(
                lambda: self.check_onAFNI('anat', 'func'))
        self.ui_objs.append(self.ui_chkFuncAnat_btn)

        self.ui_chkROIFunc_btn = QtWidgets.QPushButton('Check ROI on function')
        hBoxROICheck.addWidget(self.ui_chkROIFunc_btn)
        self.ui_chkROIFunc_btn.clicked.connect(
                lambda: self.check_onAFNI('func', 'roi'))
        self.ui_objs.append(self.ui_chkROIFunc_btn)

        self.ui_chkROIAnat_btn = QtWidgets.QPushButton('Check ROI on anatomy')
        hBoxROICheck.addWidget(self.ui_chkROIAnat_btn)
        self.ui_chkROIAnat_btn.clicked.connect(
                lambda: self.check_onAFNI('anat', 'roi'))
        self.ui_objs.append(self.ui_chkROIAnat_btn)

        self.ui_chkWMFunc_btn = QtWidgets.QPushButton('Check WM on anatomy')
        hBoxROICheck.addWidget(self.ui_chkWMFunc_btn)
        self.ui_chkWMFunc_btn.clicked.connect(
                lambda: self.check_onAFNI('anat', 'wm'))
        self.ui_objs.append(self.ui_chkWMFunc_btn)

        self.ui_chkVentFunc_btn = \
            QtWidgets.QPushButton('Check Vent on anatomy')
        hBoxROICheck.addWidget(self.ui_chkVentFunc_btn)
        self.ui_chkVentFunc_btn.clicked.connect(
                lambda: self.check_onAFNI('anat', 'vent'))
        self.ui_objs.append(self.ui_chkVentFunc_btn)

        # --- Create RTP mask button ---
        self.ui_makeRTPMask_btn = QtWidgets.QPushButton(
                'Create real-time processing mask\n'
                + '(+shift to overwrite; +ctrl to edit command)')
        self.ui_makeRTPMask_btn.clicked.connect(self.make_RTP_mask_ui)
        maskCreate_fLayout.addRow(self.ui_makeRTPMask_btn)
        self.ui_objs.append(self.ui_makeRTPMask_btn)

        # --- Check mask ---
        MaskCheckWidget = QtWidgets.QWidget()
        hBoxMaskCheck = QtWidgets.QHBoxLayout(MaskCheckWidget)
        maskCreate_fLayout.addRow(MaskCheckWidget)

        self.ui_chkRTPmask_btn = QtWidgets.QPushButton('Check RTP mask')
        self.ui_chkRTPmask_btn.clicked.connect(
                lambda: self.check_onAFNI('func', 'RTPmask'))
        self.ui_objs.append(self.ui_chkRTPmask_btn)
        hBoxMaskCheck.addWidget(self.ui_chkRTPmask_btn)

        self.ui_chkGSRmask_btn = QtWidgets.QPushButton('Check GSR mask')
        self.ui_chkGSRmask_btn.clicked.connect(
                lambda: self.check_onAFNI('func', 'GSRmask'))
        self.ui_objs.append(self.ui_chkGSRmask_btn)
        hBoxMaskCheck.addWidget(self.ui_chkGSRmask_btn)

        # --- Template tab ----------------------------------------------------
        templateTab = QtWidgets.QWidget()
        self.ui_top_tabs.addTab(templateTab, 'Template')
        template_fLayout = QtWidgets.QFormLayout(templateTab)

        # -- Template group box --
        self.ui_Template_grpBx = QtWidgets.QGroupBox("Template images")
        Template_gLayout = QtWidgets.QGridLayout(self.ui_Template_grpBx)
        template_fLayout.addWidget(self.ui_Template_grpBx)

        ri = 0

        # -- Templete image --
        var_lb = QtWidgets.QLabel("Template brain :")
        Template_gLayout.addWidget(var_lb, ri, 0)

        self.ui_Template_lnEd = QtWidgets.QLineEdit()
        self.ui_Template_lnEd.setText(str(self.Template))
        self.ui_Template_lnEd.setReadOnly(True)
        Template_gLayout.addWidget(self.ui_Template_lnEd, ri, 1)

        self.ui_Template_btn = QtWidgets.QPushButton('Set')
        self.ui_Template_btn.clicked.connect(
                lambda: self.set_param(
                        'Template',
                        os.path.dirname(self.ui_Template_lnEd.text()),
                        self.ui_Template_lnEd.setText))
        Template_gLayout.addWidget(self.ui_Template_btn, ri, 2)

        self.ui_objs.extend([var_lb, self.ui_Template_lnEd,
                             self.ui_Template_btn])

        # -- WM on template --
        ri += 1
        var_lb = QtWidgets.QLabel("WM mask on template :")
        Template_gLayout.addWidget(var_lb, ri, 0)

        self.ui_WM_template_lnEd = QtWidgets.QLineEdit()
        self.ui_WM_template_lnEd.setText(str(self.WM_template))
        self.ui_WM_template_lnEd.setReadOnly(True)
        Template_gLayout.addWidget(self.ui_WM_template_lnEd, ri, 1)

        self.ui_WM_template_btn = QtWidgets.QPushButton('Set')
        self.ui_WM_template_btn.clicked.connect(
                lambda: self.set_param(
                        'WM_template',
                        os.path.dirname(self.ui_WM_template_lnEd.text()),
                        self.ui_WM_template_lnEd.setText))
        Template_gLayout.addWidget(self.ui_WM_template_btn, ri, 2)

        self.ui_objs.extend([var_lb, self.ui_WM_template_lnEd,
                             self.ui_WM_template_btn])

        # -- Vent on template --
        ri += 1
        var_lb = QtWidgets.QLabel("Vent mask on template :")
        Template_gLayout.addWidget(var_lb, ri, 0)

        self.ui_Vent_template_lnEd = QtWidgets.QLineEdit()
        self.ui_Vent_template_lnEd.setText(str(self.Vent_template))
        self.ui_Vent_template_lnEd.setReadOnly(True)
        Template_gLayout.addWidget(self.ui_Vent_template_lnEd, ri, 1)

        self.ui_Vent_template_btn = QtWidgets.QPushButton('Set')
        self.ui_Vent_template_btn.clicked.connect(
                lambda: self.set_param(
                    'Vent_template',
                    os.path.dirname(self.ui_Vent_template_lnEd.text()),
                    self.ui_Vent_template_lnEd.setText))
        Template_gLayout.addWidget(self.ui_Vent_template_btn, ri, 2)

        self.ui_objs.extend([var_lb, self.ui_Vent_template_lnEd,
                             self.ui_Vent_template_btn])

        # -- ROI on template --
        ri += 1
        var_lb = QtWidgets.QLabel("ROI on template :")
        Template_gLayout.addWidget(var_lb, ri, 0)

        self.ui_ROI_template_lnEd = QtWidgets.QLineEdit()
        self.ui_ROI_template_lnEd.setText(str(self.ROI_template))
        self.ui_ROI_template_lnEd.setReadOnly(True)
        Template_gLayout.addWidget(self.ui_ROI_template_lnEd, ri, 1)

        self.ui_ROI_template_btn = QtWidgets.QPushButton('Set')
        self.ui_ROI_template_btn.clicked.connect(
                lambda: self.set_param(
                        'ROI_template',
                        os.path.dirname(self.ui_ROI_template_lnEd.text()),
                        self.ui_ROI_template_lnEd.setText))
        Template_gLayout.addWidget(self.ui_ROI_template_btn, ri, 2)

        self.ui_objs.extend([var_lb, self.ui_ROI_template_lnEd,
                             self.ui_ROI_template_btn])

        # --- Processed image tab ---------------------------------------------
        procImgTab = QtWidgets.QWidget()
        self.ui_top_tabs.addTab(procImgTab, 'Processed images')
        procImg_gLayout = QtWidgets.QGridLayout(procImgTab)

        # -- aligned anatomy --
        ri0 = 0
        var_lb = QtWidgets.QLabel("Aligned anatomy image :")
        procImg_gLayout.addWidget(var_lb, ri0, 0)

        self.ui_alAnat_lnEd = QtWidgets.QLineEdit()
        self.ui_alAnat_lnEd.setText(str(self.alAnat))
        self.ui_alAnat_lnEd.setReadOnly(True)
        procImg_gLayout.addWidget(self.ui_alAnat_lnEd, ri0, 1)

        self.ui_alAnat_btn = QtWidgets.QPushButton('Set')
        self.ui_alAnat_btn.clicked.connect(
                lambda: self.set_param('alAnat', watch_dir,
                                       self.ui_alAnat_lnEd.setText))
        procImg_gLayout.addWidget(self.ui_alAnat_btn, ri0, 2)
        self.ui_objs.extend([var_lb, self.ui_alAnat_lnEd,
                             self.ui_alAnat_btn])

        # -- warped images --
        ri0 += 1
        self.ui_wrpImg_grpBx = QtWidgets.QGroupBox('Warped images')
        wrpImg_gLayout = QtWidgets.QGridLayout(self.ui_wrpImg_grpBx)
        procImg_gLayout.addWidget(self.ui_wrpImg_grpBx, ri0, 0, 1, 3)

        # -- WM in the original space --
        ri = 0
        var_lb = QtWidgets.QLabel("WM mask in original :")
        wrpImg_gLayout.addWidget(var_lb, ri, 0)

        self.ui_WM_orig_lnEd = QtWidgets.QLineEdit()
        self.ui_WM_orig_lnEd.setText(str(self.WM_orig))
        self.ui_WM_orig_lnEd.setReadOnly(True)
        wrpImg_gLayout.addWidget(self.ui_WM_orig_lnEd, ri, 1)

        self.ui_WM_orig_btn = QtWidgets.QPushButton('Set')
        self.ui_WM_orig_btn.clicked.connect(
                lambda: self.set_param('WM_orig', watch_dir,
                                       self.ui_WM_orig_lnEd.setText))
        wrpImg_gLayout.addWidget(self.ui_WM_orig_btn, ri, 2)

        self.ui_objs.extend([var_lb, self.ui_WM_orig_lnEd,
                             self.ui_WM_orig_btn])

        # -- Vent in the original space --
        ri += 1
        var_lb = QtWidgets.QLabel("Vent mask in original :")
        wrpImg_gLayout.addWidget(var_lb, ri, 0)

        self.ui_Vent_orig_lnEd = QtWidgets.QLineEdit()
        self.ui_Vent_orig_lnEd.setText(str(self.Vent_orig))
        self.ui_Vent_orig_lnEd.setReadOnly(True)
        wrpImg_gLayout.addWidget(self.ui_Vent_orig_lnEd, ri, 1)

        self.ui_Vent_orig_btn = QtWidgets.QPushButton('Set')
        self.ui_Vent_orig_btn.clicked.connect(
                lambda: self.set_param('Vent_orig', watch_dir,
                                       self.ui_Vent_orig_lnEd.setText))
        wrpImg_gLayout.addWidget(self.ui_Vent_orig_btn, ri, 2)

        self.ui_objs.extend([var_lb, self.ui_Vent_orig_lnEd,
                             self.ui_Vent_orig_btn])

        # -- ROI in the original space --
        ri += 1
        var_lb = QtWidgets.QLabel("ROI mask in original :")
        wrpImg_gLayout.addWidget(var_lb, ri, 0)

        self.ui_ROI_orig_lnEd = QtWidgets.QLineEdit()
        self.ui_ROI_orig_lnEd.setText(str(self.ROI_orig))
        self.ui_ROI_orig_lnEd.setReadOnly(True)
        wrpImg_gLayout.addWidget(self.ui_ROI_orig_lnEd, ri, 1)

        self.ui_ROI_orig_btn = QtWidgets.QPushButton('Set')
        self.ui_ROI_orig_btn.clicked.connect(
                lambda: self.set_param('ROI_orig', watch_dir,
                                       self.ui_ROI_orig_lnEd.setText))
        wrpImg_gLayout.addWidget(self.ui_ROI_orig_btn, ri, 2)

        self.ui_objs.extend([var_lb, self.ui_ROI_orig_lnEd,
                             self.ui_ROI_orig_btn])

        # --- RTP_mask ---
        ri0 += 1
        var_lb = QtWidgets.QLabel("RTP mask :")
        procImg_gLayout.addWidget(var_lb, ri0, 0)

        self.ui_RTP_mask_lnEd = QtWidgets.QLineEdit()
        self.ui_RTP_mask_lnEd.setText(str(self.RTP_mask))
        self.ui_RTP_mask_lnEd.setReadOnly(True)
        procImg_gLayout.addWidget(self.ui_RTP_mask_lnEd, ri0, 1)

        self.ui_RTP_mask_btn = QtWidgets.QPushButton('Set')
        self.ui_RTP_mask_btn.clicked.connect(
                lambda: self.set_param('RTP_mask', watch_dir,
                                       self.ui_RTP_mask_lnEd.setText))
        procImg_gLayout.addWidget(self.ui_RTP_mask_btn, ri0, 2)

        self.ui_objs.extend([var_lb, self.ui_RTP_mask_lnEd,
                             self.ui_RTP_mask_btn])

        # --- GSR mask ---
        ri0 += 1
        var_lb = QtWidgets.QLabel("GSR mask :")
        procImg_gLayout.addWidget(var_lb, ri0, 0)

        self.ui_GSR_mask_lnEd = QtWidgets.QLineEdit()
        self.ui_GSR_mask_lnEd.setText(str(self.GSR_mask))
        self.ui_GSR_mask_lnEd.setReadOnly(True)
        procImg_gLayout.addWidget(self.ui_GSR_mask_lnEd, ri0, 1)

        self.ui_GSR_mask_btn = QtWidgets.QPushButton('Set')
        self.ui_GSR_mask_btn.clicked.connect(
                lambda: self.set_param('GSR_mask', watch_dir,
                                       self.ui_GSR_mask_lnEd.setText))
        procImg_gLayout.addWidget(self.ui_GSR_mask_btn, ri0, 2)

        self.ui_objs.extend([var_lb, self.ui_GSR_mask_lnEd,
                             self.ui_GSR_mask_btn])

        # --- Simulation tab -------------------------------------------------
        self.ui_simulationTab = QtWidgets.QWidget()
        self.ui_top_tabs.addTab(self.ui_simulationTab, 'Simulation')
        simulation_gLayout = QtWidgets.QGridLayout(self.ui_simulationTab)

        # enabled
        self.ui_simEnabled_rdb = QtWidgets.QRadioButton("Enable simulation")
        self.ui_simEnabled_rdb.setChecked(self.simEnabled)
        self.ui_simEnabled_rdb.toggled.connect(
                lambda checked: self.set_param(
                    'simEnabled', checked, self.ui_simEnabled_rdb.setChecked))
        simulation_gLayout.addWidget(self.ui_simEnabled_rdb, 0, 0)

        # --- Simulation MRI data ---
        var_lb = QtWidgets.QLabel("fMRI data :")
        simulation_gLayout.addWidget(var_lb, 1, 0)

        self.ui_simfMRIData_lnEd = QtWidgets.QLineEdit(self.simfMRIData)
        self.ui_simfMRIData_lnEd.setReadOnly(True)
        simulation_gLayout.addWidget(self.ui_simfMRIData_lnEd, 1, 1)

        self.ui_simfMRIData_btn = QtWidgets.QPushButton('Set')
        self.ui_simfMRIData_btn.clicked.connect(
                lambda: self.set_param('simfMRIData', watch_dir,
                                       self.ui_simfMRIData_lnEd.setText))
        simulation_gLayout.addWidget(self.ui_simfMRIData_btn, 1, 2)

        # --- COM port list ---
        var_lb = QtWidgets.QLabel("Simulated physio signal port :")
        simulation_gLayout.addWidget(var_lb, 2, 0)

        self.ui_simPhysPort_cmbBx = QtWidgets.QComboBox()
        self.ui_simPhysPort_cmbBx.addItems(self.simCom_descs)
        self.ui_simPhysPort_cmbBx.activated.connect(
                lambda idx:
                self.set_param('simPhysPort',
                               self.ui_simPhysPort_cmbBx.currentText(),
                               self.ui_simPhysPort_cmbBx.setCurrentText))
        simulation_gLayout.addWidget(self.ui_simPhysPort_cmbBx, 2, 1)

        # --- ECG data ---
        var_lb = QtWidgets.QLabel("ECG data :")
        simulation_gLayout.addWidget(var_lb, 3, 0)

        self.ui_simECGData_lnEd = QtWidgets.QLineEdit(self.simECGData)
        self.ui_simECGData_lnEd.setReadOnly(True)
        simulation_gLayout.addWidget(self.ui_simECGData_lnEd, 3, 1)

        self.ui_simECGData_btn = QtWidgets.QPushButton('Set')
        self.ui_simECGData_btn.clicked.connect(
            lambda: self.set_param('simECGData', watch_dir,
                                   self.ui_simECGData_lnEd.setText))
        simulation_gLayout.addWidget(self.ui_simECGData_btn, 3, 2)

        # --- Resp data ---
        var_lb = QtWidgets.QLabel("Respiration data :")
        simulation_gLayout.addWidget(var_lb, 4, 0)

        self.ui_simRespData_lnEd = QtWidgets.QLineEdit(self.simRespData)
        self.ui_simRespData_lnEd.setReadOnly(True)
        simulation_gLayout.addWidget(self.ui_simRespData_lnEd, 4, 1)

        self.ui_simRespData_btn = QtWidgets.QPushButton('Set')
        self.ui_simRespData_btn.clicked.connect(
            lambda: self.set_param('simRespData', watch_dir,
                                   self.ui_simRespData_lnEd.setText))
        simulation_gLayout.addWidget(self.ui_simRespData_btn, 4, 2)

        # --- Start simulation button ---
        self.ui_startSim_btn = QtWidgets.QPushButton('Start scan simulation')
        self.ui_startSim_btn.clicked.connect(self.start_scan_simulation)
        self.ui_startSim_btn.setStyleSheet(
            "background-color: rgb(94,63,153);"
            "height: 25px;")
        simulation_gLayout.addWidget(self.ui_startSim_btn, 5, 0, 1, 3)

        # --- Setup experiment button -----------------------------------------
        self.ui_setRTP_btn = QtWidgets.QPushButton(
            'Setup RTP parameters')
        self.ui_setRTP_btn.setStyleSheet("background-color: rgb(151,217,235);"
                                         "height: 30px;")

        self.ui_setRTP_btn.clicked.connect(self.setup_RTP)
        ui_rows.append((self.ui_setRTP_btn,))
        self.ui_objs.append(self.ui_setRTP_btn)

        # --- Show ROI signal checkbox ----------------------------------------
        self.ui_showROISig_cbx = QtWidgets.QCheckBox('Show ROI signal')
        self.ui_showROISig_cbx.setCheckState(0)
        self.ui_showROISig_cbx.stateChanged.connect(
                        lambda x: self.show_ROIsig_chk(x))
        ui_rows.append((self.ui_showROISig_cbx,))

        # --- Ready/End button ------------------------------------------------
        self.ui_readyEnd_btn = QtWidgets.QPushButton('Ready to start scan')
        self.ui_readyEnd_btn.clicked.connect(self.ready_to_run)
        self.ui_readyEnd_btn.setEnabled(False)
        self.ui_readyEnd_btn.setStyleSheet(
            "background-color: rgb(255,201,32);"
            "height: 30px;")

        self.main_win.hBoxExpCtrls.addWidget(self.ui_readyEnd_btn)

        return ui_rows

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def get_options(self):
        opts = super().get_options()

        excld_opts = ('save_dir', 'watch_dir', 'anat_orig_seg', 'ROI_mask',
                      'extApp_proc', 'extApp_com', 'app_com_port',
                      'ext_com_sock', 'chk_run_timer', 'simCom_descs',
                      'max_watch_wait', 'mri_sim', 'ext_app', 'sig_save_file',
                      'brain_anat_orig',  'plt_roi_sig', 'plt_xi', 'num_ROIs',
                      'roi_labels', 'enable_RTP')
        for k in excld_opts:
            if k in opts:
                del opts[k]

        return opts

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __del__(self):
        # Kill self.ext_app
        if self.ext_app is not None:
            pret = subprocess.check_output(f"ps ax| grep {self.ext_app}",
                                           shell=True)
            procs = [ll for ll in pret.decode().rstrip().split('\n')
                     if f'grep {self.ext_app}' not in ll]
            for ll in procs:
                llsp = ll.split()
                pid = int(llsp[0])
                cmd = "kill -KILL {}".format(pid)
                subprocess.check_call(cmd, shell=True)


# %% main (for test and debug) ================================================
if __name__ == '__main__':

    # --- Setup ---------------------------------------------------------------
    from rtpfmri.rtp_common import load_parameters
    rtp_app = RTP_APP()
    rtp_objs = rtp_app.rtp_objs
    rtp_apps = {'RTP_APP': rtp_app}

    allObjs = rtp_objs
    allObjs.update(rtp_apps)

    # Parameters should be set on GUI using main_run_app.py
    load_parameters(allObjs)

    # Set watch_dir
    watch_dir = Path('/data/rt/test')
    for rtp in ('WATCH', 'TSHIFT', 'VOLREG', 'SMOOTH', 'REGRESS'):
        if rtp in rtp_app.rtp_objs:
            rtp_app.rtp_objs[rtp].set_param('watch_dir', watch_dir)
            rtp_app.rtp_objs[rtp].set_param(
                'save_dir', (Path(rtp_app.rtp_objs[rtp].watch_dir) / 'RTP'))

    rtp_app.set_param('watch_dir', watch_dir)
    rtp_app.set_param('save_dir',  (rtp_app.watch_dir / 'RTP'))

    # --- Test functions ------------------------------------------------------
    rtp_app.setup_RTP()

    # start_scan_simulation
    rtp_app.simPhysPort = rtp_app.simCom_descs[1]
    rtp_app.start_scan_simulation()
