import os, glob, datetime,scipy.ndimage
from argparse import ArgumentParser
import numpy as np
import torch.utils.data as Dat
from Model.Net import GruopMorph
from Model.Loss import *
from Model.Function import Dataset_OASIS, SpatialTransformer, jacobian_determinant_gpu
from surface_distance import compute_robust_hausdorff, compute_surface_distances
parser = ArgumentParser()
parser.add_argument("--local_ori", type=float,
                    dest="local_ori", default=0,
                    help="Local Orientation Consistency loss: suggested range 1 to 1000")
parser.add_argument("--bs_ch", type=int,
                    dest="bs_ch", default=8,
                    help="number of basic channels")
parser.add_argument("--modelpath", type=str,
                    dest="modelpath",
                    default='Model/reg104000.pth',
                    help="model's path")
parser.add_argument("--gpu", type=str,
                    dest="gpu",
                    default='1',
                    help="gpus")

opt = parser.parse_args()
bs_ch = opt.bs_ch
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

imgshape = (160, 192, 192)
groups = (4, 2, 2)  # (4,4,4), (4,4,2), (4,2,2) or (2,2,2)
def test():

    model = GruopMorph(1, 8, imgshape, groups).cuda()
    transform = SpatialTransformer().cuda()
    step = 0
    model.eval()
    transform.eval()
    model.load_state_dict(torch.load(opt.modelpath))

    valname = sorted(glob.glob('../neurite-oasis.v1.0/OASIS_OAS1_*_MR1'))[282:414]  # 137
    valid_generator = Dat.DataLoader(Dataset_OASIS(valname, norm=False),
                                                  batch_size=1,
                                                  shuffle=False, num_workers=2)
    fixed_names = sorted(glob.glob('../neurite-oasis.v1.0/OASIS_OAS1_*_MR1'))[277:282]  # 255
    fixed_generator = Dat.DataLoader(Dataset_OASIS(fixed_names, norm=False), batch_size=1,
                                         shuffle=False, num_workers=2)
    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    j_mean = []
    j_std = []
    HD95 = []
    dice_total_reg = []
    MSES = []
    NCCS = []
    MIS = []
    mi = MutualInformation()
    print("\nValiding...")
    for batch_idx, data in enumerate(valid_generator):
        X, X_label = data['image'].to(device), data['label'].to(device)

        for batch_idx_fixed, data_fixed in enumerate(fixed_generator):
            Y, Y_label = data_fixed['image'].to(device), data_fixed['label'].to(device)

            with torch.no_grad():
                flows, warps, _ = model(X, Y)

                MSE = mse(Y, warps)
                MSE = MSE.cpu().numpy()
                MSES.append(MSE)

                ncc = -ncc_loss(Y, warps)
                ncc = ncc.cpu().numpy()
                NCCS.append(ncc)

                MI = -mi(Y, warps)
                MI = MI.cpu().numpy()
                MIS.append(MI)


                X_Y_label_5 = transform(X_label, flows, mode='nearest')
                X_Y_label = X_Y_label_5[0, 0, :, :, :]
                Y_label1 = Y_label[0, 0, :, :, :]
                dice_score_reg = dicegup(X_Y_label, Y_label1)
                dice_score_reg = dice_score_reg.cpu().numpy()
                dice_total_reg.append(dice_score_reg)



                step = step + 1

                # flows = flows.data.cpu().numpy()
                f = flows.permute(0,1,4,3,2)
                j = jacobian_determinant_gpu(f).cpu().numpy()
                j_mean.append(np.mean(j < 0))
                j_std.append(np.std(j))

                X_Y_label = X_Y_label.cpu().numpy()
                Y_label1 = Y_label1.cpu().numpy()

                count = 0
                hd95 = 0
                for i in range(1, 36):
                    if ((Y_label1 == i).sum() == 0) or ((X_Y_label == i).sum() == 0):
                        continue
                    hd95 += compute_robust_hausdorff(
                        compute_surface_distances((Y_label1 == i), (X_Y_label == i), np.ones(3)), 95.)
                    count += 1
                hd95 /= count
                HD95.append(hd95)

                print("step:%d, current reg_dice:%f, current HD:%f, current HD_mean:%f, current MSE:%f"
                      % (step, dice_score_reg, hd95, np.mean(HD95), MSE))



    dice_total_reg = np.array(dice_total_reg)
    j_mean = np.array(j_mean)
    j_std = np.array(j_std)
    HD95 = np.array(HD95)
    MSE = np.array(MSES)
    NCC = np.array(NCCS)
    MI = np.array(MIS)
    print("Registration Dice mean:%f(%f), j_mean:%f, j_std:%f" %
          (dice_total_reg.mean(), dice_total_reg.std(), j_mean.mean(), j_std.mean()))
    print("Dice median:", np.median(dice_total_reg))
    print("HD95:", HD95.mean())
    print("MSE:", MSE.mean())
    log_dir = "log/test.txt"
    with open(log_dir, "a") as log:
        log.write("step:" + str(step) + ". Reg Dice mean:" + str(
            dice_total_reg.mean()) + "("+str(dice_total_reg.std())+")" +
              ".Dice_median:" + str(np.median(dice_total_reg)) + ". HD95 mean:" + str(
            HD95.mean()) + "("+str(HD95.std())+")"+ ". j_mean:" + str(j_mean.mean()) + "(" + str(
            j_mean.std()) + ")" + ". j_std:" + str(j_std.mean()) + "(" + str(j_std.std()) + ")" +
                  ". MSE:" + str(MSE.mean()) + "("+str(MSE.std())+")"
                  + ". NCC:" + str(NCC.mean()) + "("+str(NCC.std())+")"
                  + ". MI:" + str(MI.mean()) + "("+str(MI.std())+")" + "\n")




# ***************计算DICE*********************
def diceval(im1, atlas):
    unique_class = np.unique(atlas)
    dice = 0
    num_count = 0
    for i in unique_class:
        if (i == 0) or ((im1==i).sum()==0) or ((atlas==i).sum()==0):
            continue

        sub_dice = np.sum(atlas[im1 == i] == i) * 2.0 / (np.sum(im1 == i) + np.sum(atlas == i))
        dice += sub_dice
        num_count += 1
    if num_count == 0:
        return dice
    else:
        return dice/num_count
# *************计算DICE***********************

def jacobian_determinant(disp):
    _, _, H, W, D = disp.shape

    # disp = disp[:,[2,1,0],:]

    gradx = np.array([-0.5, 0, 0.5]).reshape(1, 3, 1, 1)
    grady = np.array([-0.5, 0, 0.5]).reshape(1, 1, 3, 1)
    gradz = np.array([-0.5, 0, 0.5]).reshape(1, 1, 1, 3)

    gradx_disp = np.stack([scipy.ndimage.correlate(disp[:, 0, :, :, :], gradx, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 1, :, :, :], gradx, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 2, :, :, :], gradx, mode='constant', cval=0.0)], axis=1)

    grady_disp = np.stack([scipy.ndimage.correlate(disp[:, 0, :, :, :], grady, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 1, :, :, :], grady, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 2, :, :, :], grady, mode='constant', cval=0.0)], axis=1)

    gradz_disp = np.stack([scipy.ndimage.correlate(disp[:, 0, :, :, :], gradz, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 1, :, :, :], gradz, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 2, :, :, :], gradz, mode='constant', cval=0.0)], axis=1)

    grad_disp = np.concatenate([gradx_disp, grady_disp, gradz_disp], 0)

    jacobian = grad_disp + np.eye(3, 3).reshape(3, 3, 1, 1, 1)
    jacobian = jacobian[:, :, 2:-2, 2:-2, 2:-2]
    jacdet = jacobian[0, 0, :, :, :] * (
            jacobian[1, 1, :, :, :] * jacobian[2, 2, :, :, :] - jacobian[1, 2, :, :, :] * jacobian[2, 1, :, :,
                                                                                          :]) - \
             jacobian[1, 0, :, :, :] * (
                     jacobian[0, 1, :, :, :] * jacobian[2, 2, :, :, :] - jacobian[0, 2, :, :, :] * jacobian[2,
                                                                                                   1, :, :,
                                                                                                   :]) + \
             jacobian[2, 0, :, :, :] * (
                     jacobian[0, 1, :, :, :] * jacobian[1, 2, :, :, :] - jacobian[0, 2, :, :, :] * jacobian[1,
                                                                                                   1, :, :, :])

    return jacdet



if __name__ == '__main__':
    start = datetime.datetime.now()
    test()
    end = datetime.datetime.now()
    print("Time used:", end - start)
