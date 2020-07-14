#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mmisaki@laureateinstitute.org
"""


# %% import ===================================================================
import sys
import os
from pathlib import Path
import subprocess
import numpy as np
import re
import time
import nibabel as nib
import serial
import struct
import socket
from multiprocessing import Process, Pipe, Value, Lock

from .rtp_common import RTP


# %% MRIFeeder class ==========================================================
class MRIFeeder:
    """
    Class for MRI data feeder process
    """

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __init__(self, mri_src, dst_dir, TR=0.0, suffix='sim_fMRI_{num}.nii',
                 parent=None):
        """ Initialize MRI feed
        cmd_pipe: tuple. pair of parent's and child's end of pipe
        """

        self.mri_src = mri_src
        self.dst_dir = dst_dir
        self.suffix = suffix
        self.parent = parent

        # Set MRI data parameters
        img = nib.load(mri_src)
        self._mri_feed_num = img.shape[-1]
        if TR == 0.0:
            header = img.header
            self.TR = header.get_zooms()[-1]
        else:
            self.TR = TR

        # -- Create the process --
        cmd_pipe = Pipe()
        self.p_cmd_pipe = cmd_pipe[0]  # Parent end of pipe
        self.proc = Process(target=self.run, args=(cmd_pipe[1],))

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def run(self, cmd_pipe):

        # Run file feed
        try:
            cmdtmp = f"3dTcat -prefix {Path(self.dst_dir) / self.suffix}"
            cmdtmp += f" {self.mri_src}"
            cmdtmp += "'[{num}]' 2>/dev/null"

            st = time.time()
            num = 0
            cpcmd = cmdtmp.format(**{'num': num})
            dst_f = re.search(r' -prefix (\S+)', cpcmd).groups()[0]
            feed_delay = 0.2
            while True:
                if time.time() >= st + (num+1)*self.TR-feed_delay:
                    # Feed file
                    feed_st = time.time()
                    subprocess.call(cpcmd, shell=True)
                    feed_delay = time.time() - feed_st
                    if self.parent is not None:
                        self.parent.logmsg(f"Write {dst_f}")
                    else:
                        print(f"Write {dst_f}")
                    num += 1
                    if num == self._mri_feed_num:
                        break

                    cpcmd = cmdtmp.format(**{'num': num})
                    dst_f = re.search(r' -prefix (\S+)', cpcmd).groups()[0]

                if cmd_pipe.poll():
                    cmd = cmd_pipe.recv()
                    if cmd == 'STOP':
                        if self.parent is not None:
                            self.parent.logmsg(f"+++ Stop MRI feeding")
                        else:
                            print(f"+++ Stop MRI feeding")
                        break

        finally:
            if self.parent is not None:
                self.parent.logmsg(f"+++ Finish feeding: wrote {num} volumes.")
            else:
                print(f"+++ Finish feeding: wrote {num} volumes.")
            cmd_pipe.send('FINISH')

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def start(self):
        self.proc.start()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def stop(self):
        try:
            self.p_cmd_pipe.send('STOP')
            self.proc.join(1)
        except Exception:
            pass

        self.proc.terminate()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __del__(self):
        self.stop()


# %% PhysioFeeder class =======================================================
class PhysioFeeder:

    def __init__(self, ecg_f, resp_f, recording_rate_ms=25,
                 samples_to_average=5, sport='/dev/ptmx', OnsetSig=None,
                 parent=None):
        """
        Options
        -------
        ecg_f: string
            ECG filename
        resp_f: string
            Resp filename
        recording_rate_ms: int
            Sampling interval of ECG and Resp file data
        samples_to_average: int
            Number of samples averaged at making ECG and Resp file data.Thus,
            the actual signal frequency was
            1000 * samples_to_average/recording_rate_ms Hz
        sport: string
            output serial port device name (default is pseudo serial port
            /dev/ptmx)
        OnsetSig : OnsetSignal object
        parent:
        """

        self._cname = self.__class__.__name__
        self.parent = parent

        # -- Read data from file --
        self.read_data_files(ecg_f, resp_f)

        # -- set signal timing --
        self._recording_rate_ms = recording_rate_ms
        self._samples_to_average = samples_to_average

        # -- Set output port --
        self._sport = sport
        self.set_out_ports(sport)

        self.OnsetSig = OnsetSig

        self._count = Value('L', 0)  # data counter as shared memory
        self._count_lock = Lock()  # counter lock
        try:
            # -- Create process --
            cmd_pipe = Pipe()
            self.p_cmd_pipe = cmd_pipe[0]
            self.proc = Process(target=self.run, args=(cmd_pipe[1],))
        except Exception:
            if self.parent is not None:
                self.parent.errmsg('Failed to create Physiofeeder.')
            else:
                sys.stderr.write(
                    f"{time.ctime()}: Failed to create Physiofeeder.\n")
            # traceback.print_exc(file=sys.stdout)
            return

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def read_data_files(self, ecg_f, resp_f):
        """
        Read ECG, Resp data files

        Set variables
        -------------
        self._ecg: array
            ECG data vector
        self._resp: array
            Resp data vector
        self._siglen: int
            data length
        """

        # -- Initialize --
        ecg = []
        resp = []
        minlen = 0

        # -- Read data --
        # ECG
        if len(ecg_f) and os.path.isfile(ecg_f):
            # Read text
            ecg = open(ecg_f).read()
            # Split text and convert to int
            ecg = [int(round(float(v))) for v in ecg.rstrip().split('\n')]
            # Convert to short int array
            ecg = np.array(ecg, dtype=np.int16)
            minlen = len(ecg)
            log_msg = f"Read {len(ecg)} values from {ecg_f}"
            self._log(log_msg)

        # Respiration
        if os.path.isfile(resp_f):
            # Read text
            resp = open(resp_f).read()
            # Split text and convert to int
            resp = [int(round(float(v))) for v in resp.rstrip().split('\n')]
            # Convert to short int array
            resp = np.array(resp, dtype=np.int16)
            minlen = min([minlen, len(resp)])
            log_msg = f"Read {len(resp)} values from {resp_f}"
            self._log(log_msg)

        # -- adjust signal length
        if minlen > 0:
            adjusted = False
            if len(ecg) > minlen:
                ecg = ecg[:minlen]
                adjusted = True
            elif len(ecg) == 0:
                ecg = np.zeros(minlen, dtype=np.int16)

            if len(resp) > minlen:
                resp = resp[:minlen]
                adjusted = True
            elif len(resp) == 0:
                resp = np.zeros(minlen, dtype=np.int16)

            if adjusted:
                log_msg = f"Data length is adjusted to {minlen}"
                self._log(log_msg)

        # -- Save data as class variable --
        self._ecg = ecg
        self._resp = resp
        self._siglen = minlen

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_out_ports(self, sport):

        # -- Open a serial port --
        if 'ptmx' in sport:
            sport, fd = sport.split(',')
            self.sport_fd = int(fd)

        self._ser = serial.Serial(sport, 115200)
        self._ser.flushOutput()

        log_msg = f"Open serial port {sport}"
        self._log(log_msg)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def data_packet(self, seqn, ecg, resp, ecg2, ecg3):

        # packing data into packet
        seqn = seqn % (np.iinfo(np.uint16).max+1)
        byte_pack = struct.pack('H', np.array(seqn, dtype=np.uint16))
        byte_pack += struct.pack('h', ecg2)
        byte_pack += struct.pack('h', ecg3)
        byte_pack += struct.pack('h', resp)
        byte_pack += struct.pack('h', ecg)

        # checksum
        chsum = np.array(sum(struct.unpack('B' * 10, byte_pack)),
                         dtype=np.int16)

        # Add error
        # if seqn > 0 and seqn%1000 == 0:
        #    chsum += 1

        byte_pack += struct.pack('h', chsum)

        return byte_pack

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def run(self, cmd_pipe):

        # -- Set signal interval --
        interval_sec = self._recording_rate_ms/self._samples_to_average/1000

        # Start loop for sending signal
        cmd_pipe.send('STARTED')
        log_msg = "Start physio signal feeding"
        self._log(log_msg)

        t = 0
        with self._count_lock:
            self._count.value = 0

        if self.OnsetSig is not None:
            self.OnsetSig.send_scan_start()

        t0 = time.time()  # start time
        runSend = True
        while runSend:
            # Repeat _samples_to_average times to simulate raw data
            for rep in range(self._samples_to_average):
                # packet number
                t = t % 65536
                ni = self._count.value % self._siglen

                # Prepare packet
                byte_pack = self.data_packet(t, self._ecg[ni],
                                             self._resp[ni], 0, 0)

                # wait for feed time
                feed_sec = (self._count.value * self._samples_to_average + rep)
                feed_sec *= interval_sec

                while (time.time() - t0) < feed_sec:
                    time.sleep(interval_sec/10000)

                # Send packet
                if 'ptmx' in self._ser.name:
                    os.write(self.sport_fd, byte_pack)
                else:
                    self._ser.write(byte_pack)
                t += 1  # increment packet number

                if cmd_pipe.poll():
                    cmd = cmd_pipe.recv()
                    if cmd == 'STOP':
                        runSend = False
                        break

            with self._count_lock:
                self._count.value += 1  # increment data index

        log_msg = "Finish physio signal feeding."
        n_sent = (self._count.value + 1) * self._samples_to_average
        log_msg += f" {n_sent} points were sent."
        self._log(log_msg)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _log(self, msg):
        if self.parent:
            self.parent.logmsg(msg)
        else:
            print(msg)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def start(self):
        self.proc.start()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def stop(self):
        self.proc.terminate()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __del__(self):
        self.proc.terminate()


# %% OnsetSignal class ========================================================
class OnsetSignal():
    """ Onset signal sender class """

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __init__(self, onsig_port, onsig_sock_file='/tmp/rtp_uds_socket',
                 logfd=sys.stdout):

        self._onsig_port = onsig_port
        self._logfd = logfd
        self._cname = self.__class__.__name__

        if onsig_port == '0':
            self._onsig_sock_file = onsig_sock_file

        # -- Set onset signal sending port --
        if '/dev/ttyACM' in onsig_port:
            # Numato 8 Channel USB GPIO Module
            self._onsig_port_ser = serial.Serial(onsig_port, 19200,
                                                 timeout=0.1)
            self._onsig_port_ser.flushOutput()
            self._onsig_port_ser.write(b"gpio clear 0\r")

        elif onsig_port == '0':
            # Use UDS socket to send trigger signal
            self.onsig_sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
            try:
                if self._logfd:
                    log_msg = "UDS socket {} will be used".format(
                            self._onsig_sock_file)
                    log_msg += " for sending scan start signal"
                    self._log(log_msg)

            except socket.error:
                if self._logfd:
                    log_msg = "Cannot connect UDS socket "
                    log_msg += "{}".format(self._onsig_sock_file)
                    self._log(log_msg)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def send_scan_start(self):
        """ send pulse to the port """
        if '/dev/ttyACM' in self._onsig_port:
            ont = time.time()
            self._onsig_port_ser.write(b"gpio set 0\r")
            time.sleep(0.1)
            self._onsig_port_ser.write(b"gpio clear 0\r")
            print("Scan onset {}".format(ont))
            sys.stdout.flush()

        elif self._onsig_port == '0':
            if not os.path.exists(self._onsig_sock_file):
                errmsg = 'No UDS file'
                errmsg += ' {}\n'.format(self._onsig_sock_file)
                errmsg += 'UDS socket file must be prepared by the receiver.'
                if self._logfd:
                    self._log(errmsg)
            else:
                self.onsig_sock.sendto(b'1', self._onsig_sock_file)
                print("Scan onset {}".format(time.time()))
                sys.stdout.flush()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _log(self, msg):
        wmsg = "{} [{}] {}\n".format(time.ctime(), self._cname, msg)
        self._logfd.write(wmsg)


# %% MRISim class =============================================================
class rtMRISim(RTP):
    """Real-time MRI simulation
    """

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __init__(self, mri_src='', dst_dir='', suffix='', onsig_port=None):
        super().__init__()  # call __init__() in RTP class

        # Initialize variables
        self.mri_src = mri_src
        self.dst_dir = dst_dir
        self.suffix = suffix
        self.mri_feeder = None

        if onsig_port is not None:
            self.OnsetSig = OnsetSignal(onsig_port)
        else:
            self.OnsetSig = None

        self.physio_kwargs = {}
        self.phyio_feeder = None

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_physio(self, ecg_src, resp_src, physio_port, recording_rate_ms=25,
                   samples_to_average=5):
        self.physio_kwargs = {
            'ecg_f': ecg_src,
            'resp_f': resp_src,
            'recording_rate_ms': recording_rate_ms,
            'samples_to_average': samples_to_average,
            'sport': physio_port,
            'OnsetSig': self.OnsetSig,
            'parent': self}

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def run_MRI(self, mode):

        if mode == 'start':
            if self.mri_feeder is not None and self.mri_feeder.proc.is_alive():
                # mri_feeder is running
                return

            if not Path(self.dst_dir).is_dir():
                self.errmsg("Not found destination directory, {self.dst_dir}.")

            self.mri_feeder = MRIFeeder(self.mri_src, self.dst_dir,
                                        suffix=self.suffix, parent=self)

            self.mri_feeder.start()

        elif mode == 'stop':
            if self.mri_feeder is not None and self.mri_feeder.proc.is_alive():
                self.mri_feeder.stop()
                del self.mri_feeder
                self.mri_feeder = None

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def run_Physio(self, mode):
        if mode == 'start':
            if self.phyio_feeder is not None \
                    and self.phyio_feeder.proc.is_alive():
                # phyio_feeder is running
                return

            if not ('ecg_f' in self.physio_kwargs and
                    'resp_f' in self.physio_kwargs and
                    'sport' in self.physio_kwargs):
                sys.stderr.write("!!! Physio parameters have not been set.")
                return

            self.phyio_feeder = PhysioFeeder(**self.physio_kwargs)
            self.phyio_feeder.start()

        elif mode == 'stop':
            if self.phyio_feeder is not None and \
                    self.phyio_feeder.proc.is_alive():
                self.phyio_feeder.stop()
                del self.phyio_feeder
                self.phyio_feeder = None
