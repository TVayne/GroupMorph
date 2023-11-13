from thop import profile
import os
from Loss import *
import torch.nn.functional as F
from Function import SpatialTransformer, VecInt, upsample

def group_corr_volume(features1, features2, max_displacement, group):

    C = features1.shape[1]
    features1_all = torch.split(features1, C//group, dim=1)
    features2_all = torch.split(features2, C//group, dim=1)
    corrs = []
    for i in range(group):
        features1 = features1_all[i]
        features2 = features2_all[i]
        # Set maximum displacement and compute the number of image shifts.
        _, _, height, width, depth = features1.shape
        if max_displacement <= 0 or max_displacement >= height:
            raise ValueError(f'Max displacement of {max_displacement} is too large.')

        max_disp = max_displacement
        num_shifts = 2 * max_disp + 1

        # Pad features2 and shift it while keeping features1 fixed to compute the
        # cost volume through correlation.

        # Pad features2 such that shifts do not go out of bounds.
        features2_padded = torch.nn.functional.pad(
            input=features2,
            pad=[max_disp, max_disp, max_disp, max_disp, max_disp, max_disp],
            mode='constant')
        # print(features2_padded.shape)
        cost_list = []
        for i in range(num_shifts):
            for j in range(num_shifts):
                for k in range(num_shifts):
                    prod = features1 * features2_padded[:, :, i:(height + i), j:(width + j), k:(depth+k)]
                    corr = torch.mean(prod, dim=1, keepdim=True)
                    cost_list.append(corr)
        cost_volume = torch.cat(cost_list, dim=1)
        corrs.append(cost_volume)
    return corrs

class voxelencoder(nn.Module):
    def conv_block(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_ch),
                nn.LeakyReLU(0.2)
            )
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.InstanceNorm3d(out_ch),
                nn.LeakyReLU(0.2)
            )
        return layer

    def __init__(self, in_planes, plans):
        super(voxelencoder, self).__init__()
        self.conv0 = self.conv_block(in_planes, plans)  # 连接
        self.conv1 = self.conv_block(plans, 2 * plans, stride=2)
        self.conv2 = self.conv_block(2 * plans, 2*plans, stride=1)  # 连接
        self.conv3 = self.conv_block(2 * plans, 4*plans, stride=2)
        self.conv4 = self.conv_block(4 * plans, 4*plans, stride=1)  # 连接
        self.conv5 = self.conv_block(4 * plans, 8*plans, stride=2)
        self.conv6 = self.conv_block(8 * plans, 8*plans, stride=1)

    def forward(self, x):
        x1 = self.conv0(x)
        x2 = self.conv1(x1)
        x3 = self.conv2(x2)
        x4 = self.conv3(x3)
        x5 = self.conv4(x4)
        x6 = self.conv5(x5)
        x7 = self.conv6(x6)


        return x1, x3, x5, x7

class group_flow(nn.Module):
    def conv_block(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_ch),
                nn.LeakyReLU(0.2)
            )
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.InstanceNorm3d(out_ch),
                nn.LeakyReLU(0.2)
            )
        return layer
    def __init__(self, in_plans, plans, group=4):
        super(group_flow, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_plans, plans, kernel_size=3, padding=1, stride=1),
            nn.InstanceNorm3d(plans),
            nn.LeakyReLU(0.2),
            nn.Conv3d(plans, plans, kernel_size=3, padding=1, stride=1)
        )
        self.conv2 = nn.Sequential(
            nn.InstanceNorm3d(plans),
            nn.LeakyReLU(0.2),
            nn.Conv3d(plans, plans, kernel_size=3, padding=1, stride=1),
            nn.InstanceNorm3d(plans),
            nn.LeakyReLU(0.2),
            nn.Conv3d(plans, plans, kernel_size=3, padding=1, stride=1)
        )
        self.conv3 = nn.Sequential(
            nn.InstanceNorm3d(plans),
            nn.LeakyReLU(0.2),
            nn.Conv3d(plans, plans, kernel_size=3, padding=1, stride=1)
        )
        self.conv4 = nn.Sequential(
            nn.InstanceNorm3d(plans),
            nn.LeakyReLU(0.2),
            nn.Conv3d(plans, plans, kernel_size=3, padding=1, stride=1)
        )


        self.flowout = nn.Conv3d(plans, 3, kernel_size=3, padding=1, stride=1)
        self.context = self.conv_block(plans, plans//2)
        self.group = group
        self.plans = plans

    def forward(self, costvolumes, f, context=None):
        C = f.shape[1]
        fs = torch.split(f, C//self.group, dim=1)
        flows = []
        contexts = []
        if context == None:
            for i in range(self.group):
                costvolume = costvolumes[i]
                f = fs[i]
                x_in = torch.cat([f, costvolume], dim=1)
                if i >= 0:
                    x = self.conv1(x_in)
                    if i >= 1:
                        x = self.conv2(x)
                        if i >= 2:
                            x = self.conv3(x)
                            if i >= 3:
                                x = self.conv4(x)

                flowi = self.flowout(x)
                # save_flow(flowi.cpu().numpy()[0, :, :, :, :].transpose(1, 2, 3, 0),
                #           'image_GDNet' + '/subsubfield' + str(self.plans) + '_' + str(i) + '.nii.gz')
                contexti = self.context(x)
                flows.append(flowi)
                contexts.append(contexti)
        else:
            c = context.shape[1]
            context = torch.split(context, c//self.group, dim=1)
            for i in range(self.group):
                costvolume = costvolumes[i]
                f = fs[i]
                con_text = context[i]
                x_in = torch.cat([f, costvolume, con_text], dim=1)
                if i >= 0:
                    x = self.conv1(x_in)
                    if i >= 1:
                        x = self.conv2(x)
                        if i >= 2:
                            x = self.conv3(x)
                            if i >= 3:
                                x = self.conv4(x)
                flowi = self.flowout(x)
                # save_flow(flowi.cpu().numpy()[0, :, :, :, :].transpose(1, 2, 3, 0),
                #           'image_GDNet' + '/subsubfield' + str(self.plans) + '_' + str(i) + '.nii.gz')
                contexti = self.context(x)
                flows.append(flowi)
                contexts.append(contexti)
        # flow = self.flowfusion(flows, xs)
        flow = sum(flows) / self.group
        context_out = torch.cat(contexts, dim=1)
        return flow, context_out


class context_attention(nn.Module):
    def __init__(self, plans):
        super(context_attention, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(2 * plans, plans, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(plans),
            nn.LeakyReLU(0.2))
        self.conv2 = nn.Conv3d(plans, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, contextp1, contextp2, context1, context2):
        contexts = torch.cat([contextp1, contextp2], dim=1)
        contexts = self.conv1(contexts)
        wcontext1 = self.conv2(context1) * contexts
        wcontext2 = self.conv2(context2) * contexts
        sum = contexts + wcontext1 + wcontext2
        return sum

class split(nn.Module):
    def __init__(self, plans, group=4):
        super(split, self).__init__()
        self.convs = nn.ModuleList([])
        if group == 4:
            for i in range(group):
                self.convs.append(nn.Conv3d(plans, plans, kernel_size=2*i+1,
                                            stride=1, padding=(2*i+1)//2))
        if group == 2:
            for i in range(group):
                self.convs.append(nn.Conv3d(plans, plans, kernel_size=2*i+3,
                                            stride=1, padding=(2*i+3)//2))
        self.group = group
    def forward(self, m,f):
        C = m.shape[1]
        features1_all = torch.split(m, C // self.group, dim=1)
        features2_all = torch.split(f, C // self.group, dim=1)
        ms = []
        fs = []
        j = 0
        for conv in self.convs:
            ms.append(conv(features1_all[j]))
            fs.append(conv(features2_all[j]))
            j += 1
        featms = torch.cat(ms, dim=1)
        featfs = torch.cat(fs, dim=1)
        return featms, featfs

class GruopMorph(nn.Module):

    def conv_block(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_ch),
                nn.LeakyReLU(0.2)
            )
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_ch, 8, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.InstanceNorm3d(8),
                nn.LeakyReLU(0.2),
                nn.Conv3d(8, 3, kernel_size, stride=stride, padding=padding, bias=bias),
            )
        return layer


    def __init__(self, in_ch, basic_channel=8, imgshape=(160, 192, 192), groups=(4, 2, 2)):
        super(GruopMorph, self).__init__()

        self.transformer = SpatialTransformer().cuda()

        self.mf = voxelencoder(in_ch, basic_channel)
        self.max_displacement1 = 1

        self.group8 = groups[0]
        self.group4 = groups[1]
        self.group2 = groups[2]

        self.conv = self.conv_block(16+8*(self.group2//2), 3)

        self.groupflow8 = group_flow(27 + 64//self.group8, basic_channel*8, group=self.group8)
        self.groupflow4 = group_flow(27 + 32//self.group4 + 32, basic_channel*4, group=self.group4)
        self.groupflow2 = group_flow(27 + 16//self.group2 + 16, basic_channel*2, group=self.group2)


        self.context_attention8 = context_attention(basic_channel*4)
        self.context_attention4 = context_attention(basic_channel*2)
        self.context_attention2 = context_attention(basic_channel)


        self.absolute_pos_embed8 = nn.Parameter(
            torch.zeros(1, 32*self.group8, int(imgshape[0]/8), int(imgshape[1]/8), int(imgshape[2]/8)))
        self.absolute_pos_embed4 = nn.Parameter(
            torch.zeros(1, 16*self.group4, int(imgshape[0]/4), int(imgshape[1]/4), int(imgshape[2]/4)))
        self.absolute_pos_embed2 = nn.Parameter(
            torch.zeros(1, 8*self.group2, int(imgshape[0]/2), int(imgshape[1]/2), int(imgshape[2]/2)))
        self.split8 = split(basic_channel*8//self.group8, self.group8)
        self.split4 = split(basic_channel*4//self.group4, self.group4)
        self.split2 = split(basic_channel*2//self.group2, self.group2)


        self.VecInt = VecInt(7)



    def forward(self, x, y):
        m1, m2, m4, m8 = self.mf(x)
        f1, f2, f4, f8 = self.mf(y)


        smo = 0
        m8, f8 = self.split8(m8, f8)
        costvolum8 = group_corr_volume(m8, f8, max_displacement=self.max_displacement1, group=self.group8)
        vec0, context0 = self.groupflow8(costvolum8, f8)
        smo = smo + smoothloss(vec0)
        flow0 = self.VecInt(vec0)



        contextp0 = context0 + self.absolute_pos_embed8
        contextps = torch.split(contextp0, 32, dim=1)
        contexts = torch.split(context0, 32, dim=1)
        cs8 = []
        for i in range(self.group8//2):
            contexti = self.context_attention8(contextps[2*i], contextps[2*i+1], contexts[2*i], contexts[2*i+1])
            cs8.append(contexti)
        if self.group8 == self.group4:
            for i in range(self.group8 // 2):
                contexti = self.context_attention8(contextps[2 * i + 1], contextps[2 * i], contexts[2*i+1], contexts[2*i])
                cs8.append(contexti)
        context0 = torch.cat(cs8, dim=1)
        context0 = upsample(context0, is_flow=False)
        flow0 = upsample(flow0, is_flow=True)

        m4, f4 = self.split4(m4, f4)
        w4 = self.transformer(m4, flow0)
        costvolum4 = group_corr_volume(w4, f4, max_displacement=self.max_displacement1, group=self.group4)
        det_vec, context0 = self.groupflow4(costvolum4, f4, context0)
        smo = smo + smoothloss(det_vec)
        det_flow = self.VecInt(det_vec)
        flow0 = self.transformer(flow0, det_flow) + det_flow


        contextp0 = context0 + self.absolute_pos_embed4
        contextps = torch.split(contextp0, 16, dim=1)
        contexts = torch.split(context0, 16, dim=1)
        cs4 = []
        for i in range(self.group4//2):
            contexti = self.context_attention4(contextps[2*i], contextps[2*i+1], contexts[2*i], contexts[2*i+1])
            cs4.append(contexti)
        if self.group4 == self.group2:
            for i in range(self.group4 // 2):
                contexti = self.context_attention4(contextps[2 * i + 1], contextps[2 * i], contexts[2*i+1], contexts[2*i])
                cs4.append(contexti)
        context0 = torch.cat(cs4, dim=1)


        flow0 = upsample(flow0, is_flow=True)
        context0 = upsample(context0, is_flow=False)
        m2, f2 = self.split2(m2, f2)
        w2 = self.transformer(m2, flow0)
        costvolum2 = group_corr_volume(w2, f2, max_displacement=self.max_displacement1, group=self.group2)
        det_vec, context0 = self.groupflow2(costvolum2, f2, context0)
        smo = smo + smoothloss(det_vec)
        det_flow = self.VecInt(det_vec)
        flow0 = self.transformer(flow0, det_flow) + det_flow



        contextp0 = context0 + self.absolute_pos_embed2
        contextps = torch.split(contextp0, 8, dim=1)
        contexts = torch.split(context0, 8, dim=1)
        cs2 = []
        for i in range(self.group2//2):
            contexti = self.context_attention2(contextps[2*i], contextps[2*i+1], contexts[2*i], contexts[2*i+1])
            cs2.append(contexti)
        context0 = torch.cat(cs2, dim=1)


        flow0 = upsample(flow0, is_flow=True)
        context0 = upsample(context0, is_flow=False)
        warpedfeature2 = self.transformer(m1, flow0)
        x_in = torch.cat([warpedfeature2, f1, context0], dim=1)

        det_vec = self.conv(x_in)
        smo = smo + smoothloss(det_vec)
        det_flow = self.VecInt(det_vec)
        flow0 = self.transformer(flow0, det_flow) + det_flow

        warped = self.transformer(x, flow0)



        return flow0, warped, smo




if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    y = torch.rand(1, 1, 160, 192, 192)
    y = y.cuda().float()
    model = GruopMorph(1, 8).cuda()
    flop, para = profile(model, inputs=(y,y,))
    print('Flops:', "%.2fM" % (flop / 1e6), 'Params:', "%.2fM" % (para / 1e6))

