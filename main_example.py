#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mmisaki@laureateinstitute.org
"""


# %% import ===================================================================
from pathlib import Path
import pandas as pd
import numpy as np
import datetime
import sys
import time
import nibabel as nib
import torch
import shutil
import argparse
from RTiMRIPS.rtp_simulation import RTP_SIM

WORK_OUTPUT = Path('sample_RTP_out')
if not WORK_OUTPUT.is_dir():
    WORK_OUTPUT.mkdir()


# %% Set template masks =======================================================
template_dir = Path('MNI_template')
brain_f = template_dir / 'MNI152_2009_template.nii.gz'
WM_f = template_dir / 'MNI152_2009_template_WM.nii.gz'
Vent_f = template_dir / 'MNI152_2009_template_Vent.nii.gz'


# %% Make RTP simulator =======================================================
def init_rtp_sim(brain_f, WM_f, Vent_f, work_dir='./'):
    rtp_sim = RTP_SIM(work_dir=work_dir)

    # Set template images
    rtp_sim.Template = brain_f
    rtp_sim.WM_template = WM_f
    rtp_sim.Vent_template = Vent_f

    rtp_options0 = {'tshift_ignore_init': 3,  # tshift options
                    'tshift_method': 'cubic',
                    'tshift_ref_time': 0,
                    'volreg_regmode': 2,      # cubic, volreg option
                    'blur_fwhm': 6,           # smooth option
                    'max_scan_length': 300,   # regress options
                    'max_poly_order': np.inf,
                    'verb': True,
                    'onGPU': torch.cuda.is_available(),
                    'save_delay': False
                    }

    return rtp_sim, rtp_options0


# %% Check and copy files =====================================================
def check_copy_files(dataList, work_root=WORK_OUTPUT, overwrite=False):
    for si, row in dataList.iterrows():
        work_dir = Path(work_root) / row.Idx
        if not work_dir.is_dir():
            work_dir.mkdir()

        for idx, ff in row.iteritems():
            if idx == 'Idx':
                continue
            assert Path(ff).is_file(), f"Not found {row.Idx}: {ff}\n"

            dst = work_dir / Path(ff).name
            if not dst.is_file() or overwrite:
                shutil.copy(ff, dst)

            dataList.loc[si, idx] = dst

    return dataList


# %% RTP_warp: warp_template_to_orig ==========================================
def RTP_warp(dataList, rtp_sim, work_root, overwrite=False):

    for si, row in dataList.iterrows():
        work_dir = work_root / row.Idx
        if not work_dir.is_dir():
            work_dir.mkdir()

        # Check results
        WM_orig_f = (work_dir /
                     rtp_sim.WM_template.name.replace('.nii',
                                                      '_inOrigFunc.nii')
                     )
        Vent_orig_f = (work_dir /
                       rtp_sim.Vent_template.name.replace('.nii',
                                                          '_inOrigFunc.nii')
                       )
        if WM_orig_f.is_file() and Vent_orig_f.is_file() and not overwrite:
            continue

        print("=" * 80)
        print(f"Warp template to orig {row.Idx} ({si+1}/{len(dataList)})")
        sys.stdout.flush()

        st = time.time()
        rtp_sim.work_dir = work_dir
        rtp_sim.func_orig = row.Func
        rtp_sim.anat_orig = row.Anat

        # Run warping
        rtp_sim.proc_anat(overwrite=overwrite)

        # end message
        et = time.time() - st
        etstr = str(datetime.timedelta(seconds=et)).split('.')[0]
        print(f"Done ({si+1}/{len(dataList)}) took {etstr}\n")
        sys.stdout.flush()


# %% RTP_mask: make_function_image_mask =======================================
def RTP_mask(dataList, rtp_sim, work_root, overwrite=False):
    for si, row in dataList.iterrows():
        work_dir = work_root / row.Idx
        if not work_dir.is_dir():
            work_dir.mkdir()

        # Check results
        RTP_mask_f = work_dir / 'RTP_mask.nii.gz'
        GSR_mask_f = work_dir / 'GSR_mask.nii.gz'
        if RTP_mask_f.is_file() and GSR_mask_f.is_file() and not overwrite:
            continue

        print("=" * 80)
        print(f"Make function mask {row.Idx} ({si+1}/{len(dataList)})")
        st = time.time()
        sys.stdout.flush()

        func_src = row.Func
        anat_src = list(work_dir.glob('brain_*_al_func.nii.gz'))[0]

        # make_function_image_mask
        rtp_sim.make_function_image_mask(func_src, anat_src=anat_src,
                                         overwrite=overwrite)

        # end message
        et = time.time() - st
        etstr = str(datetime.timedelta(seconds=et)).split('.')[0]
        print(f"Done {row.Idx} ({si+1}/{len(dataList)}) took {etstr}\n")
        sys.stdout


# %% Run_RTP ==================================================================
def Run_RTP(dataList, rtp_sim, rtp_options0, proc_type,  work_root=WORK_OUTPUT,
            WATCH=False, phys_sample_freq=40, overwrite=False):
    """ RTP_Run
    Rrun RTP simulation

    Parameters
    ----------
    dataList : pandas DataFrame
        data list.
    rtp_sim : RTP_SIM object
        RTP simulator.
    rtp_options0 : dictionary
        RTP options.
    proc_type : string
        Process type
    work_root : Path
        root of working directory
    WATCH : bool
        Add WATCH at the head of process chain
    phys_sample_freq : float
        sampling frequency of ECG and Resp
    overwrite : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    None.

    """

    # ---  Set Procs ---
    Procs = []
    Regs = []
    if WATCH:
        Procs += ['watch']

    for ii, proc in enumerate(proc_type.split('_')):
        if proc == 'TS':
            Procs.append('tshift')
        elif proc == 'VR':
            Procs.append('volreg')
        elif proc == 'SM':
            Procs.append('smooth')
        elif proc == 'REG':
            Procs.append('regress')
            Regs = proc_type.split('_')[ii+1:]

    # --- run RTP simulation for each data ---
    for si, row in dataList.iterrows():
        work_dir = work_root / row.Idx
        rtp_sim.work_dir = work_dir

        out_dir = work_dir / proc_type
        if not out_dir.is_dir():
            out_dir.mkdir()

        res_fs = list(out_dir.glob('*_all.nii.gz'))
        if len(res_fs) > 0 and not overwrite:
            continue

        if overwrite and len(res_fs):
            for ff in res_fs:
                ff.unlink()

        print("=" * 80)
        print(f"{proc_type} for {row.Idx} ({si+1}/{len(dataList)})")

        # --- Set parameters ---
        # Input image
        imput_img_f = row.Func

        rtp_options = rtp_options0.copy()
        if 'watch' in Procs:
            ftype = imput_img_f.suffix
            if ftype == '.gz':
                ftype = imput_img_f.stem.suffix
            rtp_options['ftype'] = ftype[1:]

        if 'tshift' in Procs or 'regress' in Procs:
            # Set TR: TR must be set for TSHIFT and REGRESS
            # (for setting polyreg number)
            header = nib.load(imput_img_f).header
            if hasattr(header, 'info'):
                TR = header.info['TAXIS_FLOATS'][1]
            elif hasattr(header, 'get_zooms'):
                TR = header.get_zooms()[3]
            else:
                TR = 2.0
            rtp_options['TR'] = TR

        if 'tshift' in Procs:
            # TSHIFT options
            rtp_options['tshift_sample_func'] = imput_img_f

        if 'volreg' in Procs:
            # VOLREG reference
            rtp_options['volreg_refvol'] = imput_img_f
            rtp_options['volreg_ref_vi'] = 0

        if 'smooth' in Procs:
            # SMOOTH options
            RTP_mask_f = work_dir / 'RTP_mask.nii.gz'
            assert RTP_mask_f.is_file()
            rtp_options['smooth_mask'] = RTP_mask_f

        if 'regress' in Procs:
            # --- REGRESS options ---
            rtp_options['max_poly_order'] = np.inf
            rtp_options['mot_reg'] = 'None'
            rtp_options['GS_reg'] = False
            rtp_options['WM_reg'] = False
            rtp_options['Vent_reg'] = False
            rtp_options['phys_reg'] = 'None'

            if 'mot' in Regs:
                if 'dmot' not in Regs:
                    rtp_options['mot_reg'] = 'mot6'
                else:
                    rtp_options['mot_reg'] = 'mot12'
            elif 'dmot' in Regs:
                rtp_options['mot_reg'] = 'dmot6'

            if 'gs' in Regs:
                # Global signal regressor
                GSR_mask_f = work_dir / 'GSR_mask.nii.gz'
                assert GSR_mask_f.is_file()
                rtp_options['GS_reg'] = True
                rtp_options['GS_mask'] = GSR_mask_f

            if 'wmvent' in Regs:
                # WM, Vent regressors
                WM_mask_f = work_dir / \
                    'MNI152_2009_template_WM_inOrigFunc.nii.gz'
                Vent_mask_f = work_dir / \
                    'MNI152_2009_template_Vent_inOrigFunc.nii.gz'
                assert WM_mask_f.is_file()
                assert Vent_mask_f.is_file()
                rtp_options['WM_reg'] = True
                rtp_options['WM_mask'] = WM_mask_f
                rtp_options['Vent_reg'] = True
                rtp_options['Vent_mask'] = Vent_mask_f

            if 'ricor' in Regs or 'rvt' in Regs:
                # Physio regressors
                rtp_options['phys_reg'] = 'None'
                rtp_options['ecg_f'] = Path(row.ECG)
                rtp_options['resp_f'] = Path(row.Resp)
                rtp_options['sample_freq'] = phys_sample_freq
                if 'ricor' in Regs:
                    if 'rvt' in Regs:
                        rtp_options['phys_reg'] = 'RVT+RICOR13'
                    else:
                        rtp_options['phys_reg'] = 'RICOR8'
                else:
                    rtp_options['phys_reg'] = 'RVT5'

        # --- run ---
        rtp_sim.run_RTP(Procs, imput_img_f, out_dir, **rtp_options)


# %% main =====================================================================
if __name__ == '__main__':

    # --- parse augument ------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--WATCH', action='store_true', default=False,
                        help="Add WATCH")
    parser.add_argument(
        '--RTP',
        default='TS_VR_SM_REG_hpf_mot_dmot_gs_wmvent_ricor_rvt',
        help="Process type")
    parser.add_argument('--dataList', default='DataList.csv',
                        help='datalist file (.csv).')
    parser.add_argument("--overwrite", action='store_true', default=False,
                        help="overwrite option")
    args = parser.parse_args()

    WATCH = args.WATCH
    RTP = args.RTP
    dataList_f = args.dataList
    overwrite = args.overwrite

    # --- Set RTP_PROCS -------------------------------------------------------
    # Check proc_type
    proc_comp = ['TS', 'VR', 'SM', 'REG', 'hpf', 'mot', 'dmot', 'gs', 'wmvent',
                 'ricor', 'rvt']
    dproc = np.setdiff1d(RTP.split('_'), proc_comp)
    if len(dproc):
        assert False, f"{RTP} ({dproc}) is not defined.\n"

    print("=" * 80)
    print(f"Real-time processing simulation: {RTP}")
    sys.stdout.flush()

    # --- Read the data_list --------------------------------------------------
    dataList = pd.read_csv(dataList_f)
    print(f"+++ Load data list from {dataList_f}")
    sys.stdout.flush()

    # --- Initialize rtp_sim --------------------------------------------------
    print(f"+++ Initialize RTP_SIM")
    sys.stdout.flush()
    rtp_sim, rtp_options0 = init_rtp_sim(brain_f, WM_f, Vent_f)

    # Prepare wroking files
    print(f"+++ Prepare files")
    sys.stdout.flush()

    print(f"    check and copy data files")
    sys.stdout.flush()
    dataList = check_copy_files(dataList, work_root=WORK_OUTPUT,
                                overwrite=overwrite)

    print(f"    check/warp template masks")
    sys.stdout.flush()
    RTP_warp(dataList, rtp_sim, work_root=WORK_OUTPUT, overwrite=overwrite)

    print(f"    make RTP and GSR masks")
    sys.stdout.flush()
    RTP_mask(dataList, rtp_sim, work_root=WORK_OUTPUT, overwrite=overwrite)

    # --- Run RTP simulations -------------------------------------------------
    print("-" * 80)
    print(f"--- Run {RTP} ({time.ctime()}) ---\n")
    sys.stdout.flush()

    Run_RTP(dataList, rtp_sim, rtp_options0, RTP, WATCH=WATCH,
            overwrite=overwrite)

    print(f"    end. ({time.ctime()})")
    print("-" * 80)
    sys.stdout.flush()
