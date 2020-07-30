#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Watching file creation for real-time processing

@author: mmisaki@laureateinstitute.org
"""


# %% import ==================================================================#
from pathlib import Path
import os
import sys
import time
import re
import numpy as np
import datetime
import traceback

from watchdog.observers.polling import PollingObserverVFS
from watchdog.events import FileSystemEventHandler
from PyQt5 import QtWidgets

from .rtp_common import MRI_data, RTP

import nibabel as nib


# %% class RTP_WATCH ==========================================================
class RTP_WATCH(RTP):
    """
    Watching new file creation, reading data from the file, and sending data to
    a next process.

    e.g.
    # Make an instance
    rtp_watch = RTP_WATCH('watching/path', 'nr_\d+.+\.BRIK', ignore_init=0,
                          save_dir='./RTP')

    # Start wathing
    rtp_watch.start_watching()

    # When a new file with a filename mathing to r'nr_\d+.+\.BRIK' is created,
    # the file is read by nib.load, coverted in MRI_data object, and saved nii
    # in ./RTP
    """

    # --- Class for handling new found file -----------------------------------
    class NewFileHandler(FileSystemEventHandler):

        def __init__(self, watch_file_pattern, data_proc, scan_onset=None):
            super().__init__()
            self.watch_file_pattern = watch_file_pattern
            self.data_proc = data_proc
            self.scan_onset = scan_onset

        def on_created(self, event):
            if event.is_directory:
                return

            if self.scan_onset is not None and \
                    not self.scan_onset.is_scan_on():
                return

            if re.search(self.watch_file_pattern, Path(event.src_path).name):
                self.data_proc(event.src_path)

    # --- Class for observer with custom thread functions ---------------------
    class RTP_Observer(PollingObserverVFS):
        def __init__(self, stat, listdir, polling_interval=1, verb=False,
                     std_out=sys.stdout):
            super().__init__(stat, listdir, polling_interval)
            self._verb = verb
            self._std_out = std_out

        def logmsg(self, msg, ret_str=False):
            tstr = datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S.%f')
            msg = "{}:[{}]: {}".format(tstr, self.__class__.__name__, msg)
            if ret_str:
                return msg

            print(msg, file=self._std_out)
            if hasattr(self._std_out, 'flush'):
                self._std_out.flush()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __init__(self, watch_dir='', watch_file_pattern=r'nr_\d+.+\.BRIK',
                 polling_interval=0.001, **kwargs):
        """
        Parameters
        ----------
        watch_dir: str
            Watching directory
        watch_file_pattern: str
            Regular expression for watching filename
        polling_interval: float
            interval to check new file creation in seconds
        """

        super().__init__(**kwargs)  # call __init__() in RTP class

        # Set instance parameters
        self.watch_dir = watch_dir
        self.watch_file_pattern = watch_file_pattern
        self.polling_interval = polling_interval
        self.scan_name = None
        self.last_proc_f = ''
        self.done_proc = -1  # processed volume index for checking the progress

        # if watch_dir does not exist, proc_ready is False
        if not Path(self.watch_dir).is_dir():
            self._proc_ready = False

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    @property
    def proc_ready(self):
        self._proc_ready = Path(self.watch_dir).is_dir()
        if not self._proc_ready:
            self.errmsg("watch_dir ({}) is not set properly.".format(
                    self.watch_dir))

        if self.next_proc:
            self._proc_ready &= self.next_proc.proc_ready

        return self._proc_ready

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def start_watching(self, scan_onset=None):
        """
        Start watchdog observer process. The oberver will run on another
        thread.
        """

        if not self.proc_ready:
            return

        # set event_handler
        self.event_handler = \
            RTP_WATCH.NewFileHandler(self.watch_file_pattern, self.do_proc,
                                     scan_onset=scan_onset)

        # self.observer = Observer(timeout=0.001)
        self.observer = RTP_WATCH.RTP_Observer(
                os.stat, os.listdir, polling_interval=self.polling_interval,
                verb=self._verb, std_out=self._std_out)

        self.observer.schedule(self.event_handler, self.watch_dir)

        self.observer.start()

        if self._verb:
            self.logmsg("Start watchdog observer on {}.".format(
                    self.watch_dir))

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def do_proc(self, fpath):
        try:
            # Avoid proceeing the same file multiple times
            if self.last_proc_f == fpath:
                return

            self.last_proc_f = fpath

            # Increment the number of received volume
            self.vol_num += 1

            while True:  # Wait for creation complete
                try:
                    # Load file
                    img = nib.load(fpath)
                    img_data = np.squeeze(img.get_fdata()).astype(np.float32)
                    break
                except Exception:
                    continue

            affine = img.affine
            img_header = img.header

            # Set save_name
            if Path(fpath).suffix == '.gz':
                save_name = Path(Path(fpath).stem).stem
            else:
                save_name = Path(fpath).stem
            save_name = re.sub(r'\+orig.*', '', save_name)

            if self.scan_name is None:
                ma = re.search(r'.+scan_\d+__\d+', save_name)
                if ma:
                    self.scan_name = ma.group()

            # Make MRI_data object instance
            mri_data = MRI_data(img_data, affine, img_header, save_name)

            # Save the processed data
            self.proc_data = mri_data.img_data.copy()

            # Record process time
            self.proc_time.append(time.time())
            proc_delay = self.proc_time[-1] - Path(fpath).stat().st_ctime
            if self.save_delay:
                self.proc_delay.append(proc_delay)

            # log message
            if self._verb:
                f = Path(fpath).name
                if len(self.proc_time) > 1:
                    t_interval = self.proc_time[-1] - self.proc_time[-2]
                else:
                    t_interval = -1
                msg = '{}, Read {}'.format(self.vol_num, f)
                msg += ' (process time {:.4f}s, interval {:.4f}s).'.format(
                        proc_delay, t_interval)
                self.logmsg(msg)

            # Run the next process
            if self.next_proc:
                self.next_proc.do_proc(mri_data, self.vol_num,
                                       self.proc_time[-1])

            # Mark end proc
            self.done_proc = self.vol_num

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
    def stop_watching(self):
        if hasattr(self, 'observer'):
            if self.observer.isAlive():
                self.observer.stop()
                self.observer.join()
            del self.observer

        if self._verb:
            self.logmsg("Stop watchdog observer.")

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def reset(self):
        """Reset process parameters
        """

        if self.verb:
            self.logmsg(f"Reset {self.__class__.__name__} module.")

        # Wait for process in watch thread finish.
        self.stop_watching()
        self.scan_name = None
        super().reset()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_param(self, attr, val=None, reset_fn=None, echo=False):
        """
        When reset_fn is None, set_param is considered to be called from
        load_parameters function.
        """

        # -- Check value --
        if attr == 'watch_dir':
            if not Path(val).is_dir():
                return

            if hasattr(self, 'ui_wdir_lb'):
                self.ui_wdir_lb.setText(str(val))

            if self.main_win is not None:
                self.main_win.set_watch_dir_all(val)

        elif attr == 'save_dir':
            pass

        elif attr == 'polling_interval' and reset_fn is None:
            if hasattr(self, 'ui_pollingIntv_dSpBx'):
                self.ui_pollingIntv_dSpBx.setValue(val)

        elif attr == 'watch_file_pattern':
            if len(val) == 0:
                if reset_fn:
                    reset_fn(str(self.watch_file_pattern))
                return
            if reset_fn is None:
                if hasattr(self, 'ui_watchPat_lnEd'):
                    self.ui_watchPat_lnEd.setText(str(val))

        elif attr == 'clean_files':
            if Path(self.watch_dir).is_dir():
                fs = [ff for ff in Path(self.watch_dir).glob('*')
                      if re.search(self.watch_file_pattern, str(ff))]
            else:
                fs = []
            if len(fs) > 0:
                # Warning dialog
                msgBox = QtWidgets.QMessageBox()
                msgBox.setIcon(QtWidgets.QMessageBox.Question)
                msgBox.setText("Delete {} temporary files?".format(len(fs)))
                msgBox.setDetailedText('\n'.join([str(ff) for ff in fs]))
                msgBox.setWindowTitle("Delete temporary files")
                msgBox.setStandardButtons(QtWidgets.QMessageBox.Yes |
                                          QtWidgets.QMessageBox.No)
                msgBox.setDefaultButton(QtWidgets.QMessageBox.Yes)
                ret = msgBox.exec()
                if ret == QtWidgets.QMessageBox.Yes:
                    # If pattern is BRIK file (.BRIK|.HEAD), remove paried file
                    if re.search('BRIK', self.watch_file_pattern):
                        pat = re.sub('BRIK.*', 'HEAD', self.watch_file_pattern)
                        fs += [ff for ff in os.listdir(self.watch_dir)
                               if re.search(pat, ff)]
                    elif re.search('HEAD', self.watch_file_pattern):
                        pat = re.sub('HEAD', r'BRIK.*',
                                     self.watch_file_pattern)
                        fs += [ff for ff in os.listdir(self.watch_dir)
                               if re.search(pat, ff)]

                    for fbase in fs:
                        (Path(self.watch_dir) / fbase).unlink()

            return

        elif attr == 'save_proc':
            if hasattr(self, 'ui_saveProc_chb'):
                self.ui_saveProc_chb.setChecked(val)

        elif attr == '_verb':
            if hasattr(self, 'ui_verb_chb'):
                self.ui_verb_chb.setChecked(val)

        elif reset_fn is None:
            return

        # -- Set value--
        setattr(self, attr, val)
        if echo:
            print("{}.".format(self.__class__.__name__) + attr, '=',
                  getattr(self, attr))

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def ui_set_param(self):

        ui_rows = []
        self.ui_objs = []

        # watch_dir
        var_lb = QtWidgets.QLabel("Watch directory :")
        self.ui_wdir_lb = QtWidgets.QLabel(self.watch_dir)
        ui_rows.append((var_lb, self.ui_wdir_lb))
        self.ui_objs.extend([var_lb, self.ui_wdir_lb])

        # polling_interval
        var_lb = QtWidgets.QLabel("Polling interval :")
        self.ui_pollingIntv_dSpBx = QtWidgets.QDoubleSpinBox()
        self.ui_pollingIntv_dSpBx.setMinimum(0.0001)
        self.ui_pollingIntv_dSpBx.setSingleStep(0.001)
        self.ui_pollingIntv_dSpBx.setDecimals(4)
        self.ui_pollingIntv_dSpBx.setSuffix(" seconds")
        self.ui_pollingIntv_dSpBx.setValue(self.polling_interval)
        self.ui_pollingIntv_dSpBx.valueChanged.connect(
                lambda x: self.set_param('polling_interval', x,
                                         self.ui_pollingIntv_dSpBx.setValue))
        ui_rows.append((var_lb, self.ui_pollingIntv_dSpBx))
        self.ui_objs.extend([var_lb, self.ui_pollingIntv_dSpBx])

        # watch_file_pattern
        var_lb = QtWidgets.QLabel("Watch pattern :")
        self.ui_watchPat_lnEd = QtWidgets.QLineEdit()
        self.ui_watchPat_lnEd.setText(str(self.watch_file_pattern))
        self.ui_watchPat_lnEd.editingFinished.connect(
                lambda: self.set_param('watch_file_pattern',
                                       self.ui_watchPat_lnEd.text(),
                                       self.ui_watchPat_lnEd.setText))
        ui_rows.append((var_lb, self.ui_watchPat_lnEd))
        self.ui_objs.extend([var_lb, self.ui_watchPat_lnEd])

        # Clean_files
        self.ui_cleanFilest_btn = QtWidgets.QPushButton()
        self.ui_cleanFilest_btn.setText('Clean up existing watch files')
        self.ui_cleanFilest_btn.clicked.connect(
                lambda: self.set_param('clean_files'))
        ui_rows.append((None, self.ui_cleanFilest_btn))
        self.ui_objs.append(self.ui_cleanFilest_btn)

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
        excld_opts = ('watch_dir', 'save_dir', 'scan_name', 'done_proc',
                      'last_proc_f')
        sel_opts = {}
        for k, v in all_opts.items():
            if k in excld_opts:
                continue
            if isinstance(v, Path):
                v = str(v)
            sel_opts[k] = v

        sel_opts['watch_dir'] = self.watch_dir

        return sel_opts

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __del__(self):
        # Kill observer process
        if hasattr(self, 'observer') and self.observer.isAlive():
            self.observer.stop()
            self.observer.join()


# %% main =====================================================================
if __name__ == '__main__':
    # --- Test ---
    import subprocess

    # Sample data directory
    src_dir = Path.home() / 'Development' / 'RTPfMRI' / 'test_data' / 'E13270'
    sample_file = src_dir / 'epiRTeeg_scan_7__004+orig.HEAD'
    assert sample_file.is_file()

    # Watch dir
    watch_dir = Path('/dev/shm/sim_tmp')

    watch_file_pattern = r'nr_\d+.+\.BRIK'

    rtp_wacth = RTP_WATCH(watch_dir, watch_file_pattern)
    rtp_wacth.ignore_init = 0
    rtp_wacth.verb = True

    rtp_wacth.save_proc = True  # save result of the last process
    rtp_wacth.save_dir = watch_dir / 'RTP'

    if not watch_dir.is_dir():
        watch_dir.mkdir()

    rtp_wacth.start_watching()

    img = nib.load(sample_file)
    N_vols = img.shape[-1]
    cmd_tmp = "3dTcat -overwrite -prefix {}/tmp_nr_{:04d}_ch0 {}'[{}]'"
    for ii in range(N_vols):
        cmd = cmd_tmp.format(watch_dir, ii, sample_file, ii)
        subprocess.call(cmd, shell=True)
        time.sleep(1)

    # Clean up watch_dir
    for ff in watch_dir.glob('*'):
        if ff.is_dir():
            for fff in ff.glob('*'):
                fff.unlink()
            ff.rmdir()
        else:
            ff.unlink()

    watch_dir.rmdir()
