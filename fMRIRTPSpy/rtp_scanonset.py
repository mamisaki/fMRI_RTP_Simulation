#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 13:20:48 2018

@author: mmisaki@laureateinstitute.org
"""


# %% import ===================================================================
import os
from pathlib import Path
import re
import serial
from serial.tools.list_ports import comports
import parallel
import socket
import numpy as np
import time
from PyQt5 import QtWidgets, QtCore
import matplotlib as mpl

from .rtp_common import RTP, RingBuffer, MatplotlibWindow

mpl.rcParams['font.size'] = 8


# %% RTP_SCANONSET class ======================================================
class RTP_SCANONSET(RTP):
    """
    Scan onset monitor
    """

    def __init__(self, onsig_port=None, verb=True):
        """
        Options
        -------
        onsig_port: str, 'parport'|'0'
            Port to monitor scan onset signal.
            'parport': monitor parallel port busy (pin 11)
             '0': Unix domain socket at socket file, /tmp/rtp_uds_socket
             'GPIO': general purpose IO (not implemented)
        verb: bool
            verbose flag to print log message
        """

        super().__init__()  # call __init__() in RTP class

        # --- Set parameters ---
        self.not_available = False
        self._verb = verb

        # Set ports
        # port list
        self.dict_onsig_port = {}
        for pt in comports():
            if 'CDC RS-232 Emulation Demo' in pt.description:
                self.dict_onsig_port[pt.device] = pt.description
            elif 'Numato Lab 8 Channel USB GPIO M' in pt.description:
                self.dict_onsig_port[pt.device] = pt.description

        for pp in Path('/dev').glob('parport*'):
            self.dict_onsig_port[str(pp)] = pp.name

        # Add Unix domain socket
        self.uds_sock_file = '/tmp/rtp_uds_socket'
        self.dict_onsig_port['UDS'] = \
            f"Unix domain socket ({self.uds_sock_file})"

        self._onsig_port = None

        if len(self.dict_onsig_port) > 0:
            if onsig_port is None:
                onsig_ports = list(self.dict_onsig_port.keys())
                if len(onsig_ports):
                    onsig_port = sorted(onsig_ports)[0]
            self.init_onsig_port(onsig_port)
        else:
            self.not_available = True
            return

        # --- recording status ---
        self.wait_scan = False
        self.scanning = False
        self.scan_onset = -1.0

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def init_onsig_port(self, onsig_port):
        if onsig_port is None:
            return

        if '(' in onsig_port:
            onsig_port = onsig_port.split('(')[0].rstrip()

        if onsig_port not in self.dict_onsig_port.keys():
            self.errmsg("No DIO port {} exists".format(onsig_port))
            return

        if '/dev/parport' in onsig_port:
            # Parallel port
            if self._onsig_port == onsig_port:
                del self._pport
                time.sleep(1)

            self._onsig_port = onsig_port
            try:
                self._pport = parallel.Parallel(onsig_port)
            except Exception as e:
                self.errmsg(e)
                errmsg = "'sudo modprobe ppdev parport_pc parprot' and "
                errmsg += "'sudo modprobe -r lp' might solve the problem"
                self.errmsg(errmsg)
                errmsg = "Scan onset signal cannot be received"
                errmsg += " at {}".format(onsig_port)
                self.errmsg(errmsg)
                self._pport = None
                return

        elif '/dev/ttyACM' in onsig_port:
            # Numato Lab 8 Channel USB GPIO Module
            if self._onsig_port == onsig_port:
                del self._onsig_port_ser
                time.sleep(1)

            self._onsig_port = onsig_port

            try:
                self._onsig_port_ser = serial.Serial(onsig_port, 19200,
                                                     timeout=0.0005)
                self._onsig_port_ser.flushOutput()
                self._onsig_port_ser.write(b"gpio clear 0\r")
            except Exception as e:
                self.errmsg(e)
                errmsg = "Failed to open {}".format(onsig_port)
                self.errmsg(errmsg)
                errmsg = "Scan onset signal cannot be received"
                errmsg += " at {}".format(onsig_port)
                self.errmsg(errmsg)
                self._onsig_port = None
                return

        elif onsig_port == 'UDS':
            # Unix domain socket
            if self._onsig_port == onsig_port:
                del self._onsig_sock
                time.sleep(1)

            self._onsig_port = onsig_port

            # Create a UDS socket
            self._onsig_sock = socket.socket(socket.AF_UNIX,
                                             socket.SOCK_DGRAM)
            self._onsig_sock.settimeout(0.01)
            try:
                os.unlink(self.uds_sock_file)
            except Exception:
                if os.path.isfile(self.uds_sock_file):
                    raise

            self._onsig_sock.bind(self.uds_sock_file)
        else:
            self.errmsg("{} is not implemented".format(onsig_port) +
                        " for receiving scan onset signal.\n")
            return

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def read_onsig_port(self):
        if '/dev/parport' in self._onsig_port:
            if self._pport is None:
                return

            # scan Busy (pin 11)
            # wait a while (0.1 ms)
            busy = False
            st = time.time()
            while time.time()-st < 0.0001 and not busy:
                busy |= self._pport.getInBusy()

            return int(busy)

        elif '/dev/ttyACM' in self._onsig_port:
            self._onsig_port_ser.reset_output_buffer()
            self._onsig_port_ser.reset_input_buffer()
            self._onsig_port_ser.write(b"gpio read 0\r")
            resp = self._onsig_port_ser.read(1024)
            ma = re.search(r'gpio read 0\n\r(\d)\n', resp.decode())
            if ma:
                sig = ma.groups()[0]
                return int(sig == '1')
            else:
                return

        elif self._onsig_port == '0':
            try:
                sig = self._onsig_sock.recvfrom(1)[0]
                return int(sig == b'1')

            except socket.timeout:
                return

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    class Monitoring(QtCore.QObject):
        finished = QtCore.pyqtSignal()

        # ---------------------------------------------------------------------
        def __init__(self, root, main_win=None):
            super().__init__()
            self.root = root
            self.abort = False

        # ---------------------------------------------------------------------
        def run(self):
            while not self.abort:
                try:
                    if self.root.read_onsig_port() == 1:
                        self.root.scan_onset = time.time()
                        if self.root._verb:
                            self.root.logmsg("Received scan onset signal.")
                        self.root.scanning = True
                        break

                except Exception as e:
                    print(e)
                    break

                time.sleep(0.0001)

            # -- end loop --
            self.finished.emit()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def wait_scan_onset(self):
        if hasattr(self, 'thMonitor') and self.thMonitor.isRunning():
            return

        self.scanning = False
        self.scan_onset = -1.0

        self.thMonitor = QtCore.QThread()
        self.monitor = RTP_SCANONSET.Monitoring(self, main_win=self.main_win)
        self.monitor.moveToThread(self.thMonitor)
        self.thMonitor.started.connect(self.monitor.run)
        self.monitor.finished.connect(self.thMonitor.quit)
        self.thMonitor.start()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def is_waiting(self):
        return hasattr(self, 'thMonitor') and self.thMonitor.isRunning()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def abort_waiting(self):
        if self.is_waiting():
            self.monitor.abort = True
            if not self.thMonitor.wait(1):
                # self.monitor.finished is not emitted with sone reason
                self.monitor.finished.emit()
                self.thMonitor.wait()

            del self.thMonitor
            del self.monitor

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def is_scan_on(self):
        return self.scanning

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def manual_start(self):
        if not self.is_waiting():
            return

        self.scan_onset = time.time()
        if self._verb:
            self.logmsg("Manual start")
        self.scanning = True

        self.abort_waiting()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def reset(self):
        self.scanning = False
        self.scan_onset = -1.0

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    class PlotOnsigPort(QtCore.QObject):
        finished = QtCore.pyqtSignal()

        def __init__(self, root, sample_freq=40, show_length_sec=5,
                     main_win=None):
            """
            Options
            -------
            sample_freq: float
                sampling frequency, Hz
            show_length_sec: float
                plot window length, seconds
            """
            super().__init__()

            # Set variables
            self.root = root
            self.sample_freq = sample_freq
            self.main_win = main_win
            self.abort = False

            self.sig_rbuf = RingBuffer(sample_freq * show_length_sec)

            # Initialize figure
            plt_winname = 'Monitor {}'.format(self.root._onsig_port)
            self.plt_win = MatplotlibWindow()
            self.plt_win.setWindowTitle(plt_winname)

            # set position
            if main_win is not None:
                main_geom = main_win.geometry()
                x = main_geom.x() + main_geom.width()
                y = main_geom.y()
            else:
                x, y = (0, 0)
            self.plt_win.setGeometry(x, y, 360, 180)

            # Set axis
            self.ax = self.plt_win.canvas.figure.subplots()
            xi = np.arange(0, show_length_sec, 1.0/sample_freq) + \
                1.0/sample_freq
            self.ln = self.ax.plot(xi, self.sig_rbuf.get())

            self.ax.set_xlim([0, show_length_sec])
            self.ax.set_xlabel('seconds')
            self.ax.set_ylim([-0.1, 1.1])
            self.ax.set_yticks([0, 1])
            self.ax.set_ylabel('TTL')
            self.ax.set_position([0.15, 0.25, 0.8, 0.7])

            # show window
            self.plt_win.show()

        # ---------------------------------------------------------------------
        def run(self):
            interval = 1.0/self.sample_freq
            nt = time.time()+interval
            while self.plt_win.isVisible() and not self.abort:
                while time.time() < nt:
                    time.sleep(interval/10)

                val = self.root.read_onsig_port()
                if val is None:
                    pass
                    # self.sig_rbuf.append(np.nan)
                else:
                    self.sig_rbuf.append(val)
                    self.ln[0].set_ydata(self.sig_rbuf.get())

                self.ax.figure.canvas.draw()
                self.ax.figure.canvas.start_event_loop(interval/100.0)

                nt += interval

                if self.main_win is not None and \
                        not self.main_win.isVisible():
                    break

            self.end_thread()

        def end_thread(self):
            if self.plt_win.isVisible():
                self.plt_win.close()
            self.finished.emit()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_param(self, attr, val=None, reset_fn=None, echo=False):
        """
        When reset_fn is None, set_param is considered to be called from
        load_parameters function.
        """

        # -- check value --
        if attr == 'onsig_port':
            if hasattr(self, 'thMonOnsigPort') and \
                    self.thMonOnsigPort.isRunning():
                self.monOnsigPort.abort = True
                self.thMonOnsigPort.wait()
                del self.thMonOnsigPort

            if val is not None:
                self.init_onsig_port(val)
            return

        elif attr == '_onsig_port' and reset_fn is None:
            if self._onsig_port == val:
                return

            idx = self.ui_onSigPort_cmbBx.findText(val,
                                                   QtCore.Qt.MatchContains)
            if idx == -1:
                return

            if hasattr(self, 'thMonOnsigPort') and \
                    self.thMonOnsigPort.isRunning():
                self.monOnsigPort.abort = True
                self.thMonOnsigPort.wait()
                del self.thMonOnsigPort

            if hasattr(self, 'ui_onSigPort_cmbBx'):
                self.ui_onSigPort_cmbBx.setCurrentIndex(idx)

            if val is not None:
                self.init_onsig_port(val)

            return

        elif attr == 'monitor_onsig_port':
            if hasattr(self, 'thPltOnsigPort') and \
                    self.thPltOnsigPort.isRunning():
                return

            self.thPltOnsigPort = QtCore.QThread()
            self.pltOnsigPort = \
                RTP_SCANONSET.PlotOnsigPort(self, main_win=self.main_win)
            self.pltOnsigPort.moveToThread(self.thPltOnsigPort)
            self.thPltOnsigPort.started.connect(self.pltOnsigPort.run)
            self.pltOnsigPort.finished.connect(self.thPltOnsigPort.quit)
            self.thPltOnsigPort.start()
            return

        elif attr == 'plot_len_sec' and reset_fn is None:
            if hasattr(self, 'ui_pltLen_dSpBx'):
                self.ui_pltLen_dSpBx.setValue(val)

        elif isinstance(val, serial.Serial) or \
                isinstance(val, QtCore.QThread):
            return

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
    def ui_set_param(self):

        ui_rows = []
        self.ui_objs = []

        # get list of comports
        dev_list = []
        for pinfo in comports():
            dev_list.append(pinfo.device)

        # onsig_port
        var_lb = QtWidgets.QLabel("Port to receive scan onset signal :")
        self.ui_onSigPort_cmbBx = QtWidgets.QComboBox()
        devlist = sorted(['{} ({})'.format(dev, desc)
                          for dev, desc in self.dict_onsig_port.items()])
        self.ui_onSigPort_cmbBx.addItems(devlist)
        selIdx = np.argwhere([self._onsig_port in lst for lst in devlist])
        if len(selIdx):
            self.ui_onSigPort_cmbBx.setCurrentIndex(selIdx.ravel()[0])
        self.ui_onSigPort_cmbBx.activated.connect(
                lambda idx: self.set_param(
                        'onsig_port', self.ui_onSigPort_cmbBx.currentText(),
                        self.ui_onSigPort_cmbBx.setCurrentIndex))
        ui_rows.append((var_lb, self.ui_onSigPort_cmbBx))
        self.ui_objs.extend([var_lb, self.ui_onSigPort_cmbBx])

        # monitor onsig_port
        self.ui_monitorOnSigPort_btn = QtWidgets.QPushButton()
        self.ui_monitorOnSigPort_btn.setText('Show port status')
        self.ui_monitorOnSigPort_btn.clicked.connect(
                lambda: self.set_param('monitor_onsig_port'))
        ui_rows.append((None, self.ui_monitorOnSigPort_btn))
        self.ui_objs.append(self.ui_monitorOnSigPort_btn)

        # manual start button
        self.ui_manualStart_btn = QtWidgets.QPushButton()
        self.ui_manualStart_btn.setText('Manual start')
        self.ui_manualStart_btn.setStyleSheet("background-color: rgb(255,0,0)")
        self.ui_manualStart_btn.clicked.connect(self.manual_start)
        ui_rows.append((None, self.ui_manualStart_btn))

        # verb
        self.ui_verb_chb = QtWidgets.QCheckBox("Verbose logging")
        self.ui_verb_chb.setChecked(self.verb)
        self.ui_verb_chb.stateChanged.connect(
                lambda state: setattr(self, 'polling_interval', state > 0))
        ui_rows.append((self.ui_verb_chb, None))
        self.ui_objs.append(self.ui_verb_chb)

        return ui_rows

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def get_options(self):
        all_opts = super().get_options()
        excld_opts = ('save_dir', 'scanning', 'scan_onset', 'wait_scan',
                      'dict_onsig_port', 'not_available')
        sel_opts = {}
        for k, v in all_opts.items():
            if k in excld_opts:
                continue
            sel_opts[k] = v

        return sel_opts

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __del__(self):
        if hasattr(self, 'thMonOnsigPort') and \
                self.thMonOnsigPort.isRunning():
            self.monOnsigPort.abort = True
            self.thMonOnsigPort.wait()
            del self.thMonOnsigPort


# %% Dummy SCANONSET class for debug
class DUMMY_SCANONSET(RTP):
    def __init__(self):
        super().__init__()  # call __init__() in RTP class
        self.scanning = False

    def is_scan_on(self):
        return self.scanning
