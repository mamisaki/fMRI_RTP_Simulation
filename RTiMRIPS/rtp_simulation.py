#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mmisaki@laureateinstitute.org
"""


# %% import ===================================================================
from pathlib import Path
import nibabel as nib
import shutil
import re
import subprocess
import numpy as np
import time
import sys
from datetime import timedelta

from .rtp_common import MRI_data
from .rtp_watch import RTP_WATCH
from .rtp_volreg import RTP_VOLREG
from .rtp_smooth import RTP_SMOOTH
from .rtp_regress import RTP_REGRESS
from .rtp_retrots import RTP_RETROTS
from .rtp_physio_dummy import RTP_PHYSIO_DUMMY
from .rtp_app import RTP_APP


# %% RTP_SIM class ============================================================
class RTP_SIM(RTP_APP):
    """RTP simulation class
    """

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __init__(self, work_dir='./', verb=True):
        super().__init__(verb=verb)
        self.rtp_physio = None
        self.work_dir = Path(work_dir)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def setup_rtp_watch(self, ftype='nii', polling_interval=0.001,
                        watch_ignore_init=0, max_scan_length=300,
                        save_delay=False, verb=True, **kwargs):
        """Setup RTP_WATCH module
        """

        watch_dir = self.work_dir / 'RT_DATA'
        if watch_dir.is_dir():
            shutil.rmtree(watch_dir)
        watch_dir.mkdir()

        # Set options
        self.rtp_watch.watch_dir = watch_dir
        if ftype == 'HEAD':
            ftype = 'BRIK*'
        watch_file_pattern = r'nr_\d+.*\.' + ftype
        self.rtp_watch.watch_file_pattern = watch_file_pattern
        self.rtp_watch.polling_interval = polling_interval
        self.rtp_watch.ignore_init = watch_ignore_init
        self.rtp_watch.max_scan_length = max_scan_length
        self.rtp_watch.save_delay = save_delay
        self.rtp_watch.verb = verb

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def setup_rtp_tshift(self, TR=2.0, tshift_sample_func=None,
                         slice_timing=[], slice_dim=2, tshift_ignore_init=3,
                         tshift_method='cubic', tshift_ref_time=0,
                         max_scan_length=300, save_delay=False, verb=True,
                         **kwargs):
        """Setup RTP_TSHIFT module
        """

        self.rtp_tshift.TR = TR
        if tshift_sample_func is not None and \
                Path(tshift_sample_func).is_file():
            self.rtp_tshift.slice_timing_from_sample(tshift_sample_func)
        else:
            self.rtp_tshift.slice_timing = slice_timing
        self.rtp_tshift.slice_dim = slice_dim
        self.rtp_tshift.ignore_init = tshift_ignore_init
        self.rtp_tshift.method = tshift_method
        self.rtp_tshift.ref_time = tshift_ref_time
        self.rtp_tshift.max_scan_length = max_scan_length
        self.rtp_tshift.save_delay = save_delay
        self.rtp_tshift.verb = verb

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def setup_rtp_volreg(self, volreg_refvol, volreg_ref_vi=0,
                         volreg_regmode=2, max_scan_length=300,
                         save_delay=False, verb=True, **kwargs):
        """Setup RTP_VOLREG module
        """

        self.rtp_volreg.regmode = volreg_regmode
        self.rtp_volreg.set_ref_vol(volreg_refvol, volreg_ref_vi)
        self.rtp_volreg.max_scan_length = max_scan_length
        self.rtp_volreg.save_delay = save_delay
        self.rtp_volreg.verb = verb

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def setup_rtp_smooth(self, blur_fwhm, smooth_mask, max_scan_length=300,
                         save_delay=False, verb=True, **kwargs):
        """Setup RTP_REGRESS module
        """

        self.rtp_smooth.blur_fwhm = blur_fwhm
        self.rtp_smooth.set_mask(smooth_mask)
        self.rtp_smooth.max_scan_length = max_scan_length
        self.rtp_smooth.save_delay = save_delay
        self.rtp_smooth.verb = verb

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def setup_rtp_regress(
            self, max_scan_length=300, wait_num=0, rtp_volreg=None,
            mot_reg='mot12', rtp_physio=None, phys_reg='RVT+RICOR13', TR=2.0,
            tshift_ref_time=0.0, mask_src_proc=None, GS_mask=None,
            GS_reg=False, WM_mask=None, WM_reg=False, Vent_mask=None,
            Vent_reg=False, max_poly_order=np.inf, onGPU=False, desMtx_f=None,
            save_delay=False, verb=True, **kwargs):
        """Setup RTP_REGRESS module
        """

        self.rtp_regress.onGPU = onGPU
        self.rtp_regress.set_wait_num(wait_num)

        self.rtp_regress.mask_file = 0  # Receive from a previous proc
        # Need for making RETROICOR regressor
        self.rtp_regress.TR = TR
        self.rtp_regress.tshift = tshift_ref_time
        self.rtp_regress.max_scan_length = max_scan_length

        if rtp_volreg is not None:
            self.rtp_regress.volreg = rtp_volreg
            self.rtp_regress.mot_reg = mot_reg
        else:
            self.rtp_regress.mot_reg = 'None'

        if rtp_physio is not None:
            self.rtp_regress.physio = rtp_physio
            self.rtp_regress.phys_reg = phys_reg
        else:
            self.rtp_regress.phys_reg = 'None'  # 'RVT+RICOR13'

        if mask_src_proc is not None:
            self.rtp_regress.mask_src_proc = mask_src_proc

        if mask_src_proc is not None and GS_mask is not None:
            if Path(GS_mask).is_file():
                self.rtp_regress.GS_mask = GS_mask
                self.rtp_regress.GS_reg = GS_reg
            else:
                print("{GS_mask} was not found.")
                sys.stdout.flush()
                self.rtp_regress.GS_reg = False
        else:
            self.rtp_regress.GS_reg = False

        if mask_src_proc is not None and WM_mask is not None:
            if Path(WM_mask).is_file():
                self.rtp_regress.WM_mask = WM_mask
                self.rtp_regress.WM_reg = WM_reg
            else:
                print("{WM_mask} was not found.")
                sys.stdout.flush()
                self.rtp_regress.WM_reg = False
        else:
            self.rtp_regress.WM_reg = False

        if mask_src_proc is not None and Vent_mask is not None:
            if Path(Vent_mask).is_file():
                self.rtp_regress.Vent_mask = Vent_mask
                self.rtp_regress.Vent_reg = Vent_reg
            else:
                print("{Vent_mask} was not found.")
                sys.stdout.flush()
                self.rtp_regress.Vent_reg = False
        else:
            self.rtp_regress.Vent_reg = False

        if desMtx_f is not None:
            self.rtp_regress.set_param('desMtx_f', desMtx_f)

        self.rtp_regress.save_delay = save_delay
        self.rtp_regress.verb = verb

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def process_chain(self, Procs, out_dir, **kwargs):
        """Make process chain
        """

        print("Process chain setup:")
        sys.stdout.flush()

        # Reset chain
        for rtp in ('rtp_watch', 'rtp_tshift', 'rtp_volreg', 'rtp_smooth',
                    'rtp_regress'):
            setattr(getattr(self, rtp), 'next_proc', None)
        initial_proc = None
        last_proc = None

        for proc in Procs:
            print('+' * 20, f"\n++ {proc}")
            if proc == 'watch':
                self.setup_rtp_watch(**kwargs)
                proc_obj = self.rtp_watch

            elif proc == 'tshift':
                self.setup_rtp_tshift(**kwargs)
                proc_obj = self.rtp_tshift

            elif proc == 'volreg':
                self.setup_rtp_volreg(**kwargs)
                proc_obj = self.rtp_volreg

            elif proc == 'smooth':
                self.setup_rtp_smooth(**kwargs)
                proc_obj = self.rtp_smooth

            elif proc == 'regress':
                # --- regress: Set volreg module for mot_reg ---
                if ('mot_reg' in kwargs and kwargs['mot_reg'] != 'None') or \
                        ('mot_reg' not in kwargs and
                         self.rtp_regress.mot_reg != 'None'):
                    # Search rtp_volreg in the chain
                    rtp_volreg = None
                    obj = initial_proc
                    while True:
                        if isinstance(obj, RTP_VOLREG):
                            rtp_volreg = obj
                            break

                        obj = obj.next_proc
                        if obj is None:
                            break

                    if rtp_volreg is not None:
                        kwargs['rtp_volreg'] = rtp_volreg
                    else:
                        self.errmsg("No VOLREG in the process chain. " +
                                    "Change mot_reg='None'")
                        kwargs['mot_reg'] = 'None'

                # --- regress: Set unsmoothed module for GS/WM/Vent_reg ---
                if ('GS_reg' in kwargs and kwargs['GS_reg']) or \
                        ('GS_reg' not in kwargs and
                         self.rtp_regress.GS_reg) or \
                        ('WM_reg' in kwargs and kwargs['WM_reg']) or \
                        ('WM_reg' not in kwargs and
                         self.rtp_regress.WM_reg) or \
                        ('Vent_reg' in kwargs and kwargs['Vent_reg']) or \
                        ('Vent_reg' not in kwargs and
                         self.rtp_regress.Vent_reg):
                    # Get unsmoothed module
                    mask_src_proc = None
                    obj = initial_proc
                    while True:
                        if not isinstance(obj, RTP_SMOOTH) and \
                                not isinstance(obj, RTP_REGRESS):
                            mask_src_proc = obj

                        obj = obj.next_proc
                        if obj is None:
                            break
                    kwargs['mask_src_proc'] = mask_src_proc

                # --- regress: Set physio module for phys_reg ---
                if ('phys_reg' in kwargs and kwargs['phys_reg'] != 'None') or \
                        ('phys_reg' not in kwargs and
                         self.rtp_regress.phys_reg != 'None'):
                    # Check ecg, resp files
                    if 'ecg_f' not in kwargs or 'resp_f' not in kwargs:
                        self.errmsg("No ecg/resp file is given. " +
                                    "Change phys_reg='None'")
                        kwargs['phys_reg'] = 'None'
                    else:
                        rtp_retrots = RTP_RETROTS()
                        rtp_physio = RTP_PHYSIO_DUMMY(
                            kwargs['ecg_f'], kwargs['resp_f'],
                            kwargs['sample_freq'], rtp_retrots)
                        kwargs['rtp_physio'] = rtp_physio

                self.setup_rtp_regress(**kwargs)
                proc_obj = self.rtp_regress
            else:
                continue

            if initial_proc is None:
                initial_proc = proc_obj

            if last_proc is not None:
                last_proc.next_proc = proc_obj

            proc_obj.save_proc = False
            last_proc = proc_obj

        last_proc.save_proc = True  # save the result of the last process
        last_proc.save_dir = out_dir

        if 'save_delay' in kwargs and kwargs['save_delay']:
            proc = initial_proc
            while proc is not None:
                proc.save_dir = out_dir
                proc = proc.next_proc

        print('done.')
        print('-' * 20)
        sys.stdout.flush()

        return initial_proc

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def print_module_options(self, mod):
        class_name = mod.__class__.__name__
        opts = mod.get_options()

        print('-' * 80)
        print(f"{class_name} parameters")
        for k, v in opts.items():
            print(f"    {k}: {v}")

        sys.stdout.flush()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def run_RTP(self, Procs, imput_img_f, out_dir, verb=True, **kwargs):

        # --- Make a process chain --------------------------------------------
        proc_chain = self.process_chain(Procs, out_dir, **kwargs)
        proc_chain.reset()
        assert proc_chain.proc_ready

        # Print options
        mod = proc_chain
        while True:
            self.print_module_options(mod)
            mod = mod.next_proc
            if mod is None:
                break
        print('-' * 80)
        sys.stdout.flush()

        # --- Load test data --------------------------------------------------
        img = nib.load(imput_img_f)
        N_vols = img.shape[-1]
        fname_stem = imput_img_f.stem
        ext = imput_img_f.suffix
        if ext == '.gz':
            fname_stem = Path(imput_img_f.stem).stem
            ext = Path(imput_img_f.stem).suffix
        ftype = ext[1:]
        fname_stem = fname_stem.replace('+orig', '')

        # --- Run RTP simulation ----------------------------------------------
        if isinstance(proc_chain, RTP_WATCH):
            # --- Simulation with rtp_watch -----------------------------------
            watch_dir = proc_chain.watch_dir
            if watch_dir.is_dir():
                shutil.rmtree(watch_dir)
            watch_dir.mkdir()

            rt_file_temp = proc_chain.watch_file_pattern
            rt_file_temp = re.sub(r'\\d\+', '{:04d}', rt_file_temp)
            rt_file_temp = re.sub(r'\.\*', '', rt_file_temp)
            rt_file_temp = fname_stem + re.sub(r'\\', '', rt_file_temp)

            if ftype == 'nii':
                print("Loading test data ...", end='')
                Vall = np.squeeze(img.get_fdata()).astype(np.float32)
                print(' done')

            proc_chain.start_watching()
            sys.stdout.flush()

            print("Start watchdog thread")
            proc_chain.start_watching()
            sys.stdout.flush()

            print('-' * 80)
            print("Start simulating real-time data processing")
            sys.stdout.flush()
            st0 = time.time()

            for ii in range(N_vols):
                st1 = time.time()

                rt_fname = watch_dir / rt_file_temp.format(ii)
                if ftype == 'nii':
                    V = Vall[:, :, :, ii]
                    rt_img = nib.Nifti1Image(V, img.affine)
                    rt_img.to_filename(rt_fname)
                elif ftype in ('BRIK', 'HEAD'):
                    rt_fname = rt_fname.parent / rt_fname.stem
                    cmd = f"3dTcat -overwrite -prefix {rt_fname}"
                    cmd += f" {imput_img_f}'[{ii}]' >/dev/null 2>&1"
                    subprocess.check_call(cmd, shell=True)

                st2 = time.time()

                # Wait for the end of the process
                while proc_chain.done_proc < ii:
                    time.sleep(0.001)

                pr_t = time.time() - st2
                cp_t = st2 - st1
                print(f"Process volume {ii}"
                      f" (took: total {cp_t+pr_t:.4f}s,"
                      f" data copy {cp_t:.4f}s, process {pr_t:.4f}s)")
                sys.stdout.flush()

            # Clean up watch_dir
            for ff in watch_dir.glob('*'):
                if ff.is_dir():
                    for fff in ff.glob('*'):
                        fff.unlink()
                    ff.rmdir()
                else:
                    ff.unlink()

            watch_dir.rmdir()
        else:
            # --- Simulation without rtp_watch: direct data feeding -----------
            print("Loadig test data ...")
            sys.stdout.flush()
            st0 = time.time()

            img_data = np.squeeze(img.get_fdata()).astype(np.float32)
            affine = img.affine
            img_header = img.header

            print('-' * 80)
            print("Start simulating real-time data processing")
            sys.stdout.flush()

            rt_file_temp = fname_stem + '_nr_{:04d}'
            for ii in range(N_vols):
                st = time.time()
                mri_data = MRI_data(img_data[:, :, :, ii], affine, img_header,
                                    rt_file_temp.format(ii))
                proc_chain.do_proc(mri_data, ii, st)

                if verb:
                    print(f"+++ Processed volume {ii}"
                          f" (took: process {time.time()-st:.4f}s) +++\n")
                    sys.stdout.flush()

        proc_chain.reset()
        etstr = str(timedelta(seconds=time.time()-st0)).split('.')[0]
        print(f"End. (took {etstr}s)\n")
        sys.stdout.flush()
        sys.stderr.flush()
