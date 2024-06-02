import torch
from tqdm import trange

from diffusers.models.unets.unet_1d import UNet1DModel
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel

def test_1D():
    anchor_num = 400
    anchor_dim = 22
    t_dim = 16

    net = UNet1DModel(
        sample_size=anchor_num,
        in_channels=anchor_dim + t_dim,
        out_channels=anchor_dim,
    )

    x = torch.rand([128, 22, 400])
    t = torch.rand([128])

    y = net(x, t)
    print(y.sample.shape)
    return True

def test_2D():
    anchor_num = 400
    anchor_dim = 22
    condition_dim = 768

    net = UNet2DConditionModel(
        in_channels=1,
        out_channels=1,
        block_out_channels=(32, 64, 128, 128),
        cross_attention_dim=condition_dim,
    ).cuda()

    x = torch.rand([10, 1, anchor_num, anchor_dim]).cuda()
    t = torch.rand([10]).cuda()
    c = torch.rand([10, 1, condition_dim]).cuda()

    for i in trange(100):
        y = net(x, t, c)
    print(y.sample.shape)
    return True

def test():
    #test_1D()
    test_2D()
    return True
