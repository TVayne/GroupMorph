import nibabel as nib
from torch.utils import data
import itertools
import numpy as np
import torch
import torch.utils.data as Data
import torch.nn as nn
import torch.nn.functional as F

def upsample(img, is_flow, scale_factor=2.0, align_corners=True):
    img_resized = nn.functional.interpolate(img, scale_factor=scale_factor, mode='trilinear',
                                            align_corners=align_corners)

    if is_flow:
        img_resized *= scale_factor

    return img_resized


class VecInt(nn.Module):
    def __init__(self, nsteps):
        super().__init__()

        assert nsteps >= 0, "nsteps should be >= 0, found: %d" % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer()

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec

class SpatialTransformer(nn.Module):
    def __init__(self):
        super(SpatialTransformer, self).__init__()

    def forward(self, src, flow, mode='bilinear'):
        shape = flow.shape[2:]

        vectors = [torch.arange(0, s) for s in shape]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)  # y, x, z
        grid = torch.unsqueeze(grid, 0)  # add batch
        grid = grid.type(torch.FloatTensor)
        grid = grid.cuda()
        # grid = grid

        new_locs = grid + flow

        for i in range(len(shape)):
            new_locs[:, i, ...] = 2*(new_locs[:,i,...]/(shape[i]-1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1,0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2,1,0]]

        return F.grid_sample(src, new_locs, mode=mode)


def save_img(I_img,savename):
    affine = np.diag([1, 1, 1, 1])
    new_img = nib.nifti1.Nifti1Image(I_img, affine, header=None)
    nib.save(new_img, savename)


def save_flow(I_img,savename):
    affine = np.diag([1, 1, 1, 1])
    new_img = nib.nifti1.Nifti1Image(I_img, affine, header=None)
    nib.save(new_img, savename)

class Dataset_OASIS(Data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, names, norm=False):
        'Initialization'
        super(Dataset_OASIS, self).__init__()
        self.norm = norm
        self.index_pair = list(names)

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.index_pair)
# todo:check the classification of index_pair
  def __getitem__(self, index):


      moved_img0 = self.index_pair[index]+"/aligned_norm.nii.gz"
      moved_label0 = self.index_pair[index]+"/aligned_seg35.nii.gz"

      moved_img = load_4D_with_crop(moved_img0, cropx=160, cropy=192, cropz=192)
      moved_label = load_4D_with_crop(moved_label0, cropx=160, cropy=192, cropz=192)

      if self.norm:
          moved_img = imgnorm(moved_img)

      moved_img = torch.from_numpy(moved_img)
      moved_label = torch.from_numpy(moved_label)

      output = {'image': moved_img.float(), 'image_label': moved_label.float(), 'index': index}
      return output

def load_4D(name):
    X0 = nib.load(name)
    X1 = X0.get_fdata()
    X2 = np.reshape(X1, (1,) + X1.shape)

    return X2

def load_4D_with_crop(name, cropx, cropy, cropz):
    X = nib.load(name)
    X = X.get_fdata()

    x, y, z = X.shape
    startx = x//2 - cropx//2
    starty = y//2 - cropy//2
    startz = z//2 - cropz//2

    X = X[startx:startx+cropx, starty:starty+cropy, startz:startz+cropz]

    X = np.reshape(X, (1,) + X.shape)
    return X

def imgnorm(img):
    max_v = np.max(img)
    min_v = np.min(img)

    norm_img = (img - min_v) / (max_v - min_v)
    return norm_img

def generate_grid_unit(imgshape):
    x = (np.arange(imgshape[0]) - ((imgshape[0] - 1) / 2)) / (imgshape[0] - 1) * 2
    y = (np.arange(imgshape[1]) - ((imgshape[1] - 1) / 2)) / (imgshape[1] - 1) * 2
    z = (np.arange(imgshape[2]) - ((imgshape[2] - 1) / 2)) / (imgshape[2] - 1) * 2
    grid = np.rollaxis(np.array(np.meshgrid(z, y, x)), 0, 4)
    grid = np.swapaxes(grid, 0, 2)
    grid = np.swapaxes(grid, 1, 2)
    return grid


def jacobian_determinant_gpu(dense_flow):
    _, _, H, W, D = dense_flow.shape

    dense_pix = dense_flow[:, [2, 1, 0], :].to(dense_flow.device)
    gradz = nn.Conv3d(3, 3, (3, 1, 1), padding=(1, 0, 0), bias=False, groups=3)
    gradz.weight.data[:, 0, :, 0, 0] = torch.tensor([-0.5, 0, 0.5]).view(1, 3).repeat(3, 1)
    gradz.to(dense_flow.device)
    grady = nn.Conv3d(3, 3, (1, 3, 1), padding=(0, 1, 0), bias=False, groups=3)
    grady.weight.data[:, 0, 0, :, 0] = torch.tensor([-0.5, 0, 0.5]).view(1, 3).repeat(3, 1)
    grady.to(dense_flow.device)
    gradx = nn.Conv3d(3, 3, (1, 1, 3), padding=(0, 0, 1), bias=False, groups=3)
    gradx.weight.data[:, 0, 0, 0, :] = torch.tensor([-0.5, 0, 0.5]).view(1, 3).repeat(3, 1)
    gradx.to(dense_flow.device)
    with torch.no_grad():
        jacobian = torch.cat((gradz(dense_pix), grady(dense_pix), gradx(dense_pix)), 0) \
                   + torch.eye(3, 3).view(3, 3, 1, 1, 1).to(dense_flow.device)
        jacobian = jacobian[:, :, 2:-2, 2:-2, 2:-2]
        jac_det = jacobian[0, 0, :, :, :] * (jacobian[1, 1, :, :, :] * jacobian[2, 2, :, :, :]
                - jacobian[1, 2, :, :, :] * jacobian[2, 1, :, :,:]) \
                  - jacobian[1, 0, :, :, :] * (jacobian[0, 1, :, :, :] * jacobian[2, 2, :, :, :]
                - jacobian[0, 2, :, :, :] * jacobian[2, 1, :, :, :]) \
                  + jacobian[ 2, 0,  :, :,:] * (jacobian[0, 1, :, :, :] * jacobian[1, 2, :, :, :]
                    - jacobian[0, 2, :, :, :] * jacobian[1,  1, :, :, :])

    return jac_det

import re
def process_label():
    #process labeling information for FreeSurfer
    # seg_table = [0, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 26,
    #                       28, 30, 31, 41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60, 62,
    #                       63, 72, 77, 80, 85, 251, 252, 253, 254, 255]
    seg_table = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                 31, 32, 33, 34, 35]


    file1 = open('seg35_labels.txt', 'r')
    Lines = file1.readlines()
    dict = {}
    seg_i = 0
    seg_look_up = []
    for seg_label in seg_table:
        for line in Lines:
            line = re.sub(' +', ' ',line).split(' ')
            try:
                int(line[0])
            except:
                continue
            if int(line[0]) == seg_label:
                seg_look_up.append([seg_i, int(line[0]), line[1]])
                dict[seg_i] = line[1]
        seg_i += 1
    return dict

def csv_writter(line, name):
    with open(name+'.csv', 'a') as file:
        file.write(line)
        file.write('\n')

def dice_val_substruct(y_pred, y_true, std_idx):
    with torch.no_grad():
        y_pred = nn.functional.one_hot(y_pred, num_classes=36)
        y_pred = torch.squeeze(y_pred, 1)
        y_pred = y_pred.permute(0, 4, 1, 2, 3).contiguous()
        y_true = nn.functional.one_hot(y_true, num_classes=36)
        y_true = torch.squeeze(y_true, 1)
        y_true = y_true.permute(0, 4, 1, 2, 3).contiguous()
    y_pred = y_pred.detach().cpu().numpy()
    y_true = y_true.detach().cpu().numpy()

    line = 'p_{}'.format(std_idx)
    for i in range(36):
        pred_clus = y_pred[0, i, ...]
        true_clus = y_true[0, i, ...]
        intersection = pred_clus * true_clus
        intersection = intersection.sum()
        union = pred_clus.sum() + true_clus.sum()
        dsc = (2.*intersection) / (union + 1e-5)
        line = line+','+str(dsc)
    return line


