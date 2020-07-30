#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mmisaki@laureateinstitute.org
"""


# %% import ===================================================================
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import nibabel as nib
import subprocess
import sys
from collections import OrderedDict


# %% Class UnetBlock ==========================================================
class ResidC3dBlock(nn.Module):
    """
    https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """

    def __init__(self, in_channels, features):
        super(ResidC3dBlock, self).__init__()

        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=features,
                               kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(num_features=features)
        self.lrelu1 = nn.LeakyReLU(inplace=True)

        self.conv2 = nn.Conv3d(in_channels=features, out_channels=features,
                               kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(num_features=features)

        self.idmap_conv = nn.Conv3d(in_channels=in_channels,
                                    out_channels=features, kernel_size=1)
        self.lrelu2 = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.lrelu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        identity_map = self.idmap_conv(identity)

        x += identity_map  # add shortcut
        out = self.lrelu2(x)

        return out


# %% Class UNet ===============================================================
class UNet(nn.Module):
    """
    https://pytorch.org/hub/mateuszbuda_brain-segmentation-pytorch_unet/
    """

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __init__(self, in_channels=1, out_channels=1,
                 features=[32, 64, 128, 256, 512]):
        super(UNet, self).__init__()

        # --- Encoder ---
        self.n_encoders = len(features) - 1
        for ii, out_ch in enumerate(features[:-1]):
            blk = ResidC3dBlock(in_channels, out_ch)
            setattr(self, f"encoder{ii+1}", blk)
            setattr(self, f"pool{ii+1}",
                    nn.MaxPool3d(kernel_size=2, stride=2))
            in_channels = out_ch

        # --- Bottleneck ---
        self.bottleneck = ResidC3dBlock(in_channels, features[-1])

        # --- Decoder ---
        in_channels = features[-1]
        for jj, out_ch in enumerate(features[-2::-1]):
            setattr(self, f"upconv{jj+1}",
                    nn.ConvTranspose3d(in_channels, out_ch, kernel_size=2,
                                       stride=2))
            blk = ResidC3dBlock(out_ch*2, out_ch)
            setattr(self, f"decoder{jj+1}", blk)
            in_channels = out_ch

        # --- Output by 1x1x1 convolution ---
        self.out_conv = nn.Conv3d(in_channels=in_channels,
                                  out_channels=out_channels, kernel_size=1)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def forward(self, x):
        """DEBUG
        import numpy as np
        x0 = torch.from_numpy(np.ones((5, 1, 64, 64, 64), dtype=np.float32))
        self = Unet()

        device = torch.device("cuda")
        x0 = x0.to(device)

        x = x0
        """

        # Encoding path
        encs = []
        for ii in range(self.n_encoders):
            encoder = getattr(self, f"encoder{ii+1}")
            pooling = getattr(self, f"pool{ii+1}")

            x = encoder(x)
            encs.append(x)
            x = pooling(x)

        # bottle neck
        bottleneck = self.bottleneck(x)

        # Decoding path
        y = bottleneck
        for jj in range(self.n_encoders):
            upconv = getattr(self, f"upconv{jj+1}")
            decoder = getattr(self, f"decoder{jj+1}")

            y = upconv(y)
            y = torch.cat((y, encs[-(jj+1)]), dim=1)
            y = decoder(y)

        # output
        output = torch.sigmoid(self.out_conv(y))
        return output


# %% dlSS: deep learning skull-strip ==========================================
class dlSS():
    """deep-learning skull-stripping by UNet
    """
    script_dir = Path(__file__).absolute().parent

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __init__(self, trained_model, device=None):
        """ Initialize dlSS

        Parameters
        ----------
        trained_model : Path
            Traiend parameter file of pytorch model for Brain segmentation.
        device : string, optional
            'cpu' or 'cuda'. When device is None, 'cuda' is used if a cuda
            device is available, othewise 'cpu' is used. The default is None.

        Returns
        -------
        dlSeg object.

        """

        assert Path(trained_model).is_file(), f"Not foound {trained_model}.\n"

        self.model_param = trained_model
        if device is None:
            self.device = torch.device("cpu" if not torch.cuda.is_available()
                                       else "cuda")
        else:
            self.device = device

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def cropping(self, img_shape, crop_size, crop_stride=16,
                 equal_stride=True):
        """Make cropping patch mask volumes

        Parameters
        ----------
        img_shape : array like
            Shape of a volume image.
        crop_size : int
            cropping patch size
        crop_stride : int, optional
            Stride of cropping area. The default is 16.
        equal_stride : bool, optional
            Adjust the stride to equally distribute the cropping areas.
            The default is True.

        Returns
        -------
        cropMaskV : array
            4D volume of cropping masks.

        """

        corners = []
        for i_size in img_shape:
            n_crop_inplane = np.ceil(
                (i_size - crop_size)/crop_stride) + 1
            if equal_stride:
                i_stride = int(
                    np.ceil((i_size-crop_size) / (n_crop_inplane-1)))
            else:
                i_stride = crop_stride

            cc = np.arange(n_crop_inplane) * i_stride
            cc = cc[cc + crop_size <= i_size]
            if cc[-1] + crop_size < i_size:
                cc = np.concatenate([cc, [i_size-crop_size]])

            corners.append(cc)

        xx, yy, zz = np.meshgrid(*corners)
        crop_corner = np.concatenate(
            [xx.reshape((-1, 1)), yy.reshape((-1, 1)), zz.reshape((-1, 1))],
            axis=1).astype(np.int)

        cropMaskV = np.zeros((len(crop_corner), *img_shape), dtype=np.uint8)
        for ii, (xc, yc, zc) in enumerate(crop_corner):
            cropMaskV[ii, xc:xc+crop_size, yc:yc+crop_size,
                      zc:zc+crop_size] = 1

        return cropMaskV

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def apply(self, image_f, out_prefix=None, crop_stride=64, batch_size=8,
              thresh=0.5, verb=True):
        """ Apply dlSS

        Parameters
        ----------
        image_f : string or Path object
            Path to the input image file.
        out_prefix : string or Path object, optional
            Output file prefix. The default is None.
            If this is not None, segmentation image is save as nii with
            out_prefix file name. Otherwise, segmentation image is returned as
            an array.
        crop_stride : int, optional
            Stride of cropping. Smaller stride samples image mode densely,
            that could result in better segmentation with the cost of
            computation time. The default is 16.
        batch_size : int, optional
            Batch size of proccessing crops. The larger batch size coud be
            faster in processing, but could hit the memory limitation (on GPU).
            The default is 32.
        thresh : float, optional
            Segmetation probabilty threshold. The default is 0.5.
        verb : bool, optional
            Verbatim. The default is True.

        Returns
        -------
        None : if out_prefix is not None.
        segVol : 3D array
            If out_prefix is None, volume array is returned.

        """

        print('+' * 60)
        print('++ dlSS')
        print(f"Input file: {image_f}")
        sys.stdout.flush()

        image_f = Path(image_f)
        assert image_f.is_file(), f"Not found {image_f}.\n"
        work_dir = image_f.parent

        # --- Preparation -----------------------------------------------------
        # Get voxel size
        img = nib.load(image_f)
        if hasattr(img.header, 'info'):
            dx, dy, dz = np.abs(img.header.info['DELTA'])
        elif hasattr(img.header, 'get_zooms'):
            dx, dy, dz = img.header.get_zooms()[:3]
        else:
            sys.stderr.write("No voxel size information in mri_data header")
            return -1

        # resample
        resample = ~np.all(np.array((dx, dy, dz)) == 1.0)
        if resample:
            res_f = work_dir / ('rm_dss.' + image_f.name)
            cmd = f"3dresample -overwrite -prefix {res_f} -dxyz 1 1 1"
            cmd += f" -rmode Linear -orient LPI -input {image_f}"
            subprocess.check_call(cmd, shell=True)
            inputV = nib.load(str(res_f)).get_fdata().astype(np.float32)
        else:
            inputV = nib.load(str(image_f)).get_fdata().astype(np.float32)

        inputV = inputV.squeeze()

        # Global sacling
        sc = np.percentile(inputV[inputV > 0], 99)
        inputV /= sc
        inputV[inputV > 1.0] = 1.0

        # --- Apply the unet --------------------------------------------------
        print('-' * 50)
        print(f"Skull-stripping")
        sys.stdout.flush()

        # Load model parameters
        model_param = torch.load(self.model_param, map_location=device)

        crop_size = model_param['crop_size']
        features = model_param['features']
        unet_model = UNet(in_channels=1, out_channels=1, features=features)
        unet_model.load_state_dict(model_param['state_dict'])
        unet_model.to(self.device)
        unet_model.eval()

        # Make cropping masks
        cropMaskV = self.cropping(inputV.shape, crop_size,
                                  crop_stride=crop_stride,
                                  equal_stride=True)

        # Initialize output
        segMask = np.zeros(inputV.shape, dtype=np.float32)
        n_eval = np.zeros(inputV.shape, dtype=np.uint)

        # Preapare input batch array
        inputs = np.empty((batch_size, 1, crop_size, crop_size, crop_size),
                          dtype=np.float32)
        crop_areas = np.empty((batch_size, *inputV.shape),
                              dtype=np.uint8)

        # Run segmetation
        n_crop = cropMaskV.shape[0]
        for ii in range(0, cropMaskV.shape[0], batch_size):
            # Collect batch samples
            batch_maskV = cropMaskV[ii:ii+batch_size, :, :, :]
            n_in_batch = batch_maskV.shape[0]
            use_patches = []
            for jj in range(n_in_batch):
                # Get cropped images
                maskedImg = inputV[cropMaskV[ii+jj, :, :, :] > 0]
                maskedImg = maskedImg.reshape([1, *inputs.shape[-3:]])

                if not np.any(maskedImg > 0):
                    continue

                # local scaling
                sc = np.percentile(maskedImg[maskedImg > 0], 99)
                maskedImg /= sc
                maskedImg[maskedImg > 1.0] = 1.0

                # Append to the input array
                inputs[jj, :, :, :, :] = maskedImg
                crop_areas[jj, :, :, :] = cropMaskV[ii, :, :, :]
                use_patches.append(jj)

            # Apply the model to a batch
            if verb:
                print("\rProcessing cropped image ..."
                      f" {ii+batch_size}/{n_crop}", end='')
                sys.stdout.flush()

            X = torch.from_numpy(
                    inputs[use_patches, :, :, :, :]).to(self.device)
            with torch.no_grad():
                outputs = unet_model(X).cpu().numpy()

            # Add outputs to the segMask
            for ll, kk in enumerate(use_patches):
                cropMask = batch_maskV[kk, :, :, :] > 0
                segMask[cropMask] += outputs[ll, 0, :, :, :].ravel()
                n_eval += cropMask

        segMask[n_eval > 0] /= n_eval[n_eval > 0]
        if verb:
            print(' done.')
            sys.stdout.flush()

        # --- Output ----------------------------------------------------------
        if resample:
            # Save image
            out_tmp0 = work_dir / f'rm_dss.tmpSS.nii'
            img = nib.load(res_f)
            simg = nib.Nifti1Image(segMask, img.affine, header=img.header)
            simg.to_filename(out_tmp0)

            # Resample
            out_tmp1 = work_dir / f'rm_dss.res.tmpSS.nii'
            cmd = f"3dresample -overwrite -master {image_f}"
            cmd += f" -input {out_tmp0}"
            cmd += f" -prefix {out_tmp1} -rmode Linear"
            subprocess.check_call(cmd, shell=True)

            segMask = nib.load(out_tmp1).get_fdata()
            segMask = (segMask > thresh).astype(np.uint8)
        else:
            segMask = (segMask > thresh).astype(np.uint8)

        # Save in a file
        if out_prefix is not None:
            out_f = Path(out_prefix)
            ext = out_f.suffix
            if ext == '.gz':
                ext = Path(out_f.stem).suffix

            if ext != '.nii':
                out_f = Path(str(out_f) + 'nii.gz')

            img = nib.load(image_f)
            simg = nib.Nifti1Image(segMask, img.affine)
            simg.to_filename(out_f)

            if verb:
                print(f"Brain mask is saved as {out_f}")

            for delf in work_dir.glob('rm_dss.*'):
                delf.unlink()
        else:
            return segMask


# %% main =====================================================================
if __name__ == '__main__':

    # --- parse argument ------------------------------------------------------
    parser = argparse.ArgumentParser(description='dlSS')

    # input, prefix
    parser.add_argument("--input", help="input file name", required=True)
    parser.add_argument("--prefix", help="output file prefix")

    # parameters
    parser.add_argument("--trained_model", help="trained model parameter file")
    parser.add_argument("--device", metavar='cpu|cuda',
                        help="Processing device; cpu or cuda (GPU)")
    parser.add_argument("--crop_stride", type=int, default=64,
                        help="Stride of cropping.")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size of proccessing crops")
    parser.add_argument("--thresh", type=float, default=0.5,
                        help="Segmentation probability threshold")
    # quiet
    parser.add_argument("--quiet", action='store_true', help="Disable verb")

    # parse
    args = parser.parse_args()

    trained_model = args.trained_model
    if trained_model is None:
        trained_model = Path(__file__).absolute().parent / 'unetSeg_Brain.pt'
    else:
        trained_model = Path(trained_model)

    assert trained_model.is_file(), \
        f"Not found paremeter file, {trained_model}\n"

    image_f = Path(args.input)
    assert image_f.is_file(), f"Not found input file {image_f}\n"

    out_prefix = args.prefix
    if out_prefix is None:
        out_prefix = image_f.parent / ('brainmask.' + image_f.name)
    else:
        out_prefix = Path(out_prefix)

    device = args.device
    crop_stride = args.crop_stride
    batch_size = args.batch_size
    thresh = args.thresh
    if args.quiet is not None and args.quiet:
        verb = False
    else:
        verb = True

    # --- Apply ---
    DSS = dlSS(trained_model=trained_model, device=device)
    DSS.apply(image_f, out_prefix, crop_stride, batch_size, thresh, verb)
