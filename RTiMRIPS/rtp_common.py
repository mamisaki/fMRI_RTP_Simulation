#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 16:52:01 2018

@author: mmisaki@laureateinstitute.org
"""


# %% import ===================================================================
from pathlib import Path
import nibabel as nib
import numpy as np
import os
import sys
import re
import datetime
from datetime import timedelta
import subprocess
import matplotlib
import matplotlib.pyplot as plt
from multiprocessing import Value, Array, Lock
from PyQt5 import QtWidgets, QtCore, QtGui
import serial
import pickle
import time
import pandas as pd
import pexpect

import matplotlib.backends.qt_compat as qt_compat
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure


# %% RTP class ================================================================
class RTP(object):
    """ Base class for real-time processing """

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __init__(self, next_proc=None, save_proc=False, save_delay=False,
                 watch_dir='./', max_scan_length=300, main_win=None, verb=True,
                 **kwargs):

        # Set arguments
        self.next_proc = next_proc
        self.save_proc = save_proc
        self.save_delay = save_delay
        self.watch_dir = watch_dir
        self.max_scan_length = max_scan_length
        self.main_win = main_win
        self.verb = verb

        # Initialize parameters
        self.vol_num = -1
        self.proc_time = []
        self.proc_delay = []

        self.saved_files = []
        self.saved_data = None
        self.save_dir = os.path.join(self.watch_dir, 'RTP')
        self.enabled = True

        self._proc_ready = False

        self._std_out = sys.stdout
        self._err_out = sys.stderr

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    @property
    def proc_ready(self):
        return self._proc_ready

    @property
    def verb(self):
        return self._verb

    @verb.setter
    def verb(self, verb):
        self._verb = verb

    @property
    def std_out(self):
        return self._std_out

    @std_out.setter
    def std_out(self, std_out):
        self._std_out = std_out

    @property
    def err_out(self):
        return self._err_out

    @err_out.setter
    def err_out(self, err_out):
        self._err_out = err_out

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def do_proc(self, mri_data, vol_idx=None, pre_proc_time=None):
        """
        Process function

        Parameters
        ----------
        mri_data: MRI_data object
            input MRI data
        vol_idx: int
            index of MRI volume
        pre_proc_time: float
            time of complete in the previous process
        """

        pass

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_param(self, attr, val, echo=True):
        setattr(self, attr, val)
        if echo:
            print("{}.".format(self.__class__.__name__) + attr, '=',
                  getattr(self, attr))

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def ui_set_param(self):
        """
        Retrun the list of usrer interfaces for setting parameters
        Each element is placed in QFormLayout.
        """

        pass

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def select_file_dlg(self, caption, directory, filt):
        """
        If the user presses Cancel, it returns a tuple of empty string.
        """
        fname = QtWidgets.QFileDialog.getOpenFileName(
            None, caption, str(directory), filt)
        return fname

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def keep_processed_image(self, mri_data):
        savefname = self.save_data(mri_data)
        self.saved_files.append(savefname)

        if self.saved_data is None:
            self.saved_data = np.zeros(
                [*mri_data.img_data.shape, self.max_scan_length],
                dtype=mri_data.img_data.dtype)
        self.saved_data[:, :, :, self.vol_num] = mri_data.img_data

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def save_data(self, mri_data):
        savefname = mri_data.save_nii(self.save_dir)

        if self._verb:
            msg = "Save data as {}".format(savefname)
            self.logmsg(msg)

        return savefname

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def save_proc_delay(self):
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)

        clname = self.__class__.__name__
        save_fname = os.path.join(
                self.save_dir, clname + '_proc_delay_' +
                time.strftime('%Y%m%d_%H%M%S') + '.txt')
        np.savetxt(save_fname, self.proc_delay, delimiter='\n')

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def reset(self):
        """Reset process parameters
        """

        # Save proc_delay
        if self.save_delay and len(self.proc_delay) > 0:
            self.save_proc_delay()

        # concatenate saved volumes
        if len(self.saved_files):
            self.concat_saved_volumes(self.saved_data, self.saved_files)

        # Reset running variables
        self.vol_num = -1
        self.done_proc = -1
        self.proc_time = []
        self.proc_delay = []
        self.saved_files = []
        self.saved_data = None

        # Reset child process
        if self.next_proc:
            self.next_proc.reset()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def get_options(self):
        opts = dict()
        excld_opts = ('watch_dir', 'main_win', '_std_out', '_err_out', '_verb',
                      'saved_files', 'proc_time', 'next_proc',
                      'vol_num', 'enabled', 'save_proc', '_proc_ready',
                      'save_delay', 'proc_delay', 'done_proc:', 'proc_data',
                      'saved_data', 'max_scan_length', 'done_proc', '_verb')

        for var_name, var_val in self.__dict__.items():
            if var_name in excld_opts or 'ui_' in var_name or \
                    isinstance(var_val, RTP) or \
                    isinstance(var_val, serial.Serial) or \
                    isinstance(var_val, QtCore.QThread):
                continue

            if isinstance(var_val, Path):
                var_val = str(var_val)

            try:
                pickle.dumps(var_val)
                opts[var_name] = str(var_val)
            except Exception:
                continue

        return opts

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def concat_saved_volumes(self, saved_data, saved_files):
        """Seve concat data
        """

        # --- Start message ---
        if self._verb:
            self.logmsg("Saving concatenated data ...")

        if self.main_win is not None:
            progress_bar = DlgProgressBar(
                    title="Concatenating saved files ...",
                    parent=self.main_win)
            # Disable cancel button
            progress_bar.btnCancel.setVisible(False)
            QtWidgets.QApplication.processEvents()

        # --- Set filename ---
        commstr = re.sub(r'\.nii.*', '', saved_files[0])
        for ff in saved_files[1:]:
            commstr = ''.join([c for ii, c in enumerate(str(ff)[:len(commstr)])
                               if c == commstr[ii]])
        savefname0 = re.sub(r'_nr_\d*', '', commstr)
        savefname0 = re.sub(r'_ch\d*', '', savefname0) + '_all'
        savefname = savefname0

        # Add file number postfix to avoid overwriting existing files
        fn = 1
        while os.path.isfile(savefname + '.nii.gz'):
            savefname = savefname0 + '_{}'.format(fn)
            fn += 1
        savefname += '.nii.gz'
        savefname = Path(savefname)

        # --- Save data ---
        if self.main_win is not None:
            progress_bar.add_desc(
                f"Saving RTP data in {savefname} ...\n")
            QtWidgets.QApplication.processEvents()
        else:
            print(f"Saving RTP data in {savefname} ...")

        # Preapre data array
        time.sleep(0.1)
        img_data = saved_data[:, :, :, :self.vol_num+1]
        affine = nib.load(saved_files[0]).affine
        simg = nib.Nifti1Image(img_data, affine)
        simg.to_filename(savefname)

        if self.main_win is not None:
            progress_bar.set_value(95)
            progress_bar.add_desc("Removing individual files ...\n")
            QtWidgets.QApplication.processEvents()
        else:
            print(" Done")
            print("Removing individual files ...", end='')

        for ff in saved_files:
            if Path(ff).is_file():
                Path(ff).unlink()

        if self.main_win is not None:
            progress_bar.add_desc('Done')
            progress_bar.set_value(100)
            QtWidgets.QApplication.processEvents()
            progress_bar.close()
        else:
            print(" Done")

        if self._verb:
            self.logmsg("Complete saving the concatenated file.")

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def logmsg(self, msg, ret_str=False):
        tstr = datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S.%f')
        msg = "{}:[{}]: {}".format(tstr, self.__class__.__name__, msg)
        if ret_str:
            return msg

        self._std_out.write(msg + '\n')
        if hasattr(self._std_out, 'flush'):
            self._std_out.flush()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def errmsg(self, errmsg, ret_str=False, no_pop=False):
        tstr = datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S.%f')
        msg = "{}:[{}]: !!! {}".format(tstr, self.__class__.__name__, errmsg)
        if ret_str:
            return msg

        self._err_out.write(msg + '\n')
        if hasattr(self._err_out, 'flush'):
            self._err_out.flush()

        if self.main_win is not None and not no_pop:
            # 'parent' cannot be self.main_win since this could be called by
            # other thread.
            msgBox = QtWidgets.QMessageBox()
            msgBox.setIcon(QtWidgets.QMessageBox.Critical)
            msgBox.setText(errmsg)
            msgBox.setWindowTitle(self.__class__.__name__)
            msgBox.exec()


# %% MRI_data class ===========================================================
class MRI_data:
    """ MRI data class """

    def __init__(self, img_data, affine, img_header, save_name='./temp'):
        self.img_data = img_data  # numpy array image data
        self.affine = affine
        self.img_header = img_header  # nibabel image header
        self.save_name = save_name  # filename when the data is saved as nii
        self.data_order = 'x-first'  # x-first | z-first

    @property
    def save_name(self):
        return self._save_name

    @save_name.setter
    def save_name(self, fname):
        if Path(fname).suffix == '.gz':
            fname = Path(fname).stem

        if Path(fname).suffix in ('.nii', '.BRIK', '.HEAD'):
            fname = Path(fname).stem

        self._save_name = str(fname)

    def save_nii(self, save_dir='./'):
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        fname = os.path.join(save_dir, self.save_name+'.nii.gz')
        simg = nib.Nifti1Image(self.img_data, self.affine)
        # NOTE: self.img_header is a nibabel.brikhead.AFNIHeader object , which
        # does not have self.img_header.affine property.

        simg.to_filename(fname)

        return fname


# %% plot_pause ===============================================================
def plot_pause(interval):
    """
    Pause function for matplotlib to update a plot in background
    https://stackoverflow.com/questions/45729092/make-interactive-matplotlib-window-not-pop-to-front-on-each-update-windows-7/45734500#45734500
    """

    backend = plt.rcParams['backend']
    if backend in matplotlib.rcsetup.interactive_bk:
        figManager = matplotlib._pylab_helpers.Gcf.get_active()
        if figManager is not None:
            canvas = figManager.canvas
            if canvas.figure.stale:
                canvas.draw()

            canvas.start_event_loop(interval)
            return


# %% RingBuffer ===============================================================
class RingBuffer:
    """ Ring buffer with shared memory Array """

    def __init__(self, max_size):
        self.cur = Value('i', 0)
        self.max = max_size
        self.data = Array('d', np.ones(max_size)*np.nan)
        self._lock = Lock()

    def append(self, x):
        """ Append an element overwriting the oldest one. """
        with self._lock:
            self.data[self.cur.value] = x
            self.cur.value = (self.cur.value+1) % self.max

    def get(self):
        """ return list of elements in correct order """
        return np.concatenate([self.data[self.cur.value:],
                               self.data[:self.cur.value]])


# %% MatplotlibWindow =========================================================
class MatplotlibWindow(qt_compat.QtWidgets.QMainWindow):
    def __init__(self, figsize=[5, 3]):
        super().__init__()

        self.canvas = FigureCanvas(Figure(figsize=figsize))
        self.setCentralWidget(self.canvas)


# %% boot_afni ================================================================
def boot_afni(main_win=None, work_dir='./'):
    # Check afni process
    pret = subprocess.check_output("ps ax| grep afni", shell=True)
    procs = [ll for ll in pret.decode().rstrip().split('\n')
             if 'grep afni' not in ll and 'RTafni' not in ll]
    if len(procs) > 0:
        for ll in procs:
            llsp = ll.split()
            pid = int(llsp[0])
            cmdl = ' '.join(llsp[4:])
            if 'RTafni' in cmdl:
                continue

            # Warning dialog
            msgBox = QtWidgets.QMessageBox()
            msgBox.setIcon(QtWidgets.QMessageBox.Warning)
            msgBox.setText("Kill existing afni process ?")
            msgBox.setInformativeText("pid={}: {}".format(pid, cmdl))
            msgBox.setWindowTitle("Existing afni process")
            msgBox.setStandardButtons(QtWidgets.QMessageBox.Yes |
                                      QtWidgets.QMessageBox.Ignore |
                                      QtWidgets.QMessageBox.Cancel)
            msgBox.setDefaultButton(QtWidgets.QMessageBox.Cancel)
            ret = msgBox.exec()
            if ret == QtWidgets.QMessageBox.Yes:
                cmd = "kill -KILL {}".format(pid)
                subprocess.check_call(cmd, shell=True)
            elif ret == QtWidgets.QMessageBox.Cancel:
                return

    # Boot afni at watching directory
    if main_win is not None:
        work_dir = main_win.lineEditWdir.text()
        xpos = 0
        ypos = main_win.frameGeometry().height()+25  # 25 => titlebar height
    else:
        xpos = 0
        ypos = 0

    cmd = "cd {}; ".format(work_dir)
    cmd += "afni -rt -yesplugouts -DAFNI_REALTIME_WRITEMODE=1"
    cmd += " -DAFNI_TRUSTHOST=10.124"
    cmd += " -com 'OPEN_WINDOW A geom=+{}+{}'".format(xpos, ypos)

    subprocess.call(cmd, shell=True)


# %% save_parameters ==========================================================
def save_parameters(objs, fname='RTPfMRI_params.pkl'):

    props = dict()
    for k in objs.keys():
        props[k] = dict()
        for var_name, var_val in objs[k].__dict__.items():
            if var_name in ('main_win', '_std_out', '_err_out') or \
                    'ui_' in var_name or \
                    isinstance(var_val, RTP) or \
                    isinstance(var_val, serial.Serial) or \
                    isinstance(var_val, QtCore.QThread):
                continue

            try:
                pickle.dumps(var_val)
                props[k][var_name] = var_val
            except Exception:
                continue

    with open(fname, 'wb') as fd:
        pickle.dump(props, fd)


# %% load_parameters ==========================================================
def load_parameters(objs, fname='RTPfMRI_params.pkl'):

    if not os.path.isfile(fname):
        sys.stderr.write("Not found parameter file: {}".format(fname))
        return False

    try:
        with open(fname, 'rb') as fd:
            props = pickle.load(fd)

        for mod in props.keys():
            if mod in ('RETROTS',):
                continue

            if mod in objs:
                obj = objs[mod]
                for var_name, var_val in props[mod].items():
                    if var_name == 'watch_dir':
                        continue
                    obj.set_param(var_name, var_val)
    except Exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        errmsg = f'{exc_type}, {exc_tb.tb_frame.f_code.co_filename}' + \
            f':{exc_tb.tb_lineno}'
        sys.stderr.write(f"Failed to load {fname}")
        sys.stderr.write(errmsg + '\n')


# %% DlgProgressBar ===========================================================
class DlgProgressBar(QtWidgets.QDialog):
    """
    Progress bar dialog
    """

    def __init__(self, title='Progress', modal=True, parent=None,
                 win_size=(640, 320), show_button=True, st_time=None):
        super().__init__(parent)

        self.resize(*win_size)
        self.setWindowTitle(title)
        self.setModal(modal)
        vBoxLayout = QtWidgets.QVBoxLayout(self)

        # progress bar
        self.progBar = QtWidgets.QProgressBar(self)
        vBoxLayout.addWidget(self.progBar)

        # message text
        self.msgTxt = QtWidgets.QLabel()
        vBoxLayout.addWidget(self.msgTxt)

        # console output
        self.desc_pTxtEd = QtWidgets.QPlainTextEdit(self)
        self.desc_pTxtEd.setReadOnly(True)
        self.desc_pTxtEd.setLineWrapMode(QtWidgets.QPlainTextEdit.WidgetWidth)
        """
        fmt = QtGui.QTextBlockFormat()
        fmt.setLineHeight(0, QtGui.QTextBlockFormat.SingleHeight)
        self.desc_pTxtEd.textCursor().setBlockFormat(fmt)
        """

        vBoxLayout.addWidget(self.desc_pTxtEd)

        # Cancel button
        self.btnCancel = QtWidgets.QPushButton('Cancel')
        self.btnCancel.clicked.connect(self.close)
        vBoxLayout.addWidget(self.btnCancel)
        self.btnCancel.setVisible(show_button)

        self.st_time = st_time
        self.title = title

        self.show()

    def set_value(self, val):
        self.progBar.setValue(val)
        if self.st_time is not None:
            ep = self.progBar.value()
            if ep > 0:
                et = time.time() - self.st_time
                last_t = (et/ep) * (100-ep)
                last_t_str = str(timedelta(seconds=last_t))
                last_t_str = last_t_str.split('.')[0]
                tstr = f"{self.title} (ETA {last_t_str})"
                self.setWindowTitle(tstr)

        self.repaint()

    def set_msgTxt(self, msg):
        self.msgTxt.setText(msg)
        self.repaint()

    def add_msgTxt(self, msg):
        self.msgTxt.setText(self.msgTxt.text()+msg)
        self.repaint()

    def add_desc(self, txt):
        self.desc_pTxtEd.moveCursor(QtGui.QTextCursor.End)
        self.desc_pTxtEd.insertPlainText(txt)

        sb = self.desc_pTxtEd.verticalScrollBar()
        sb.setValue(sb.maximum())

        self.repaint()

    def proc_print_progress(self, proc, bar_inc=None, ETA=None):
        if bar_inc is not None:
            bar_val0 = self.progBar.value()

        self.running_proc = proc
        st = time.time()
        while proc.isalive():
            try:
                out = proc.read_nonblocking(size=100, timeout=0).decode()
                out = '\n'.join(out.splitlines())
                self.add_desc(out)

                if bar_inc is not None:
                    bar_inc0 = min((time.time() - st) / ETA * bar_inc,
                                   bar_inc)
                    if np.floor(bar_inc0+bar_val0) != self.progBar.value():
                        self.set_value(bar_inc0+bar_val0)

                QtWidgets.QApplication.processEvents()
            except pexpect.TIMEOUT:
                pass
            except pexpect.EOF:
                break
            if not self.isVisible():
                break

        try:
            out = proc.read_nonblocking(size=10000, timeout=0).decode()
            out = '\n'.join(out.splitlines()) + '\n\n'
            self.add_desc(out)
        except pexpect.EOF:
            pass

        if bar_inc is not None:
            self.set_value(bar_inc+bar_val0)
            QtWidgets.QApplication.processEvents()

        self.running_proc = None
        return proc.exitstatus

    def closeEvent(self, event):
        if hasattr(self, 'running_proc') and self.running_proc is not None:
            if self.running_proc.isalive():
                self.running_proc.terminate()


# %% LogDev ===================================================================
class LogDev(QtCore.QObject):
    write_log = QtCore.pyqtSignal(str)

    def __init__(self, fname=None, ui_obj=None):
        super().__init__()

        self.fname = fname
        if fname is not None:
            self.fd = open(fname, 'w')
        else:
            self.fd = None

        self.ui_obj = ui_obj
        if self.ui_obj is not None:
            self.write_log.connect(self.ui_obj.print_log)

    def write(self, txt):
        if self.ui_obj is not None:
            # GUI handling across threads is not allowed so logging from other
            # threads (e.g., rtp thread by watchdog) must be called via signal
            self.write_log.emit(txt)

        if self.fd is not None:
            self.fd.write(txt)

    def flush(self):
        if self.fd is not None:
            self.fd.flush()

    def __del__(self):
        if self.fd is not None:
            self.fd.close()


# %% make_design_matrix =======================================================
def make_design_matrix(stim_times, scan_len, TR, out_f):

    NT = int(scan_len//TR)
    cmd = "3dDeconvolve -x1D_stop -nodata {} {}".format(NT, TR)
    cmd += " -local_times -x1D stdout:"
    cmd += " -num_stimts {}".format(len(stim_times))
    ii = 0
    names = []
    for optstr in stim_times:
        ii += 1
        times, opt, name = optstr.split(';')
        names.append(name)
        cmd += " -stim_times {} '{}' '{}'".format(ii, times, opt)
        cmd += " -stim_label {} {}".format(ii, name)

    pr = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE)
    out = pr.stdout.decode()

    ma = re.search(r'\# \>\n(.+)\n# \<', out, re.MULTILINE | re.DOTALL)
    mtx = ma.groups()[0]
    mtx = np.array([[float(v) for v in ll.split()] for ll in mtx.split('\n')])
    mtx = mtx[:, -len(stim_times):]

    desMtx = pd.DataFrame(mtx, columns=names)
    desMtx.to_csv(out_f, index=False)
