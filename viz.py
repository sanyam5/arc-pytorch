import os
import argparse
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from models import ArcBinaryClassifier
from batcher import Batcher


parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=128, help='input batch size')
parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to ARC')
parser.add_argument('--glimpseSize', type=int, default=4, help='the height / width of glimpse seen by ARC')
parser.add_argument('--numStates', type=int, default=256, help='number of hidden states in ARC controller')
parser.add_argument('--numGlimpses', type=int, default=16, help='the number glimpses of each image in pair seen by ARC')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--name', default=None, help='Custom name for this configuration. Needed for loading model'
                                                 'and saving images')
parser.add_argument('--load', required=True, help='the model to load from.')


opt = parser.parse_args()

if opt.name is None:
    # if no name is given, we generate a name from the parameters.
    # only those parameters are taken, which if changed break torch.load compatibility.
    opt.name = "{}_{}_{}_{}".format(opt.numGlimpses, opt.glimpseSize, opt.numStates,
                                    "cuda" if opt.cuda else "cpu")


def display(image1, mask1, image2, mask2, name="hola.png"):
    _, ax = plt.subplots(1, 2)

    mask1 = (mask1 > 0.15).data.numpy()
    mask1 = np.ma.masked_where(mask1 == 0, mask1)

    mask2 = (mask2 > 0.15).data.numpy()
    mask2 = np.ma.masked_where(mask2 == 0, mask2)

    ax[0].imshow(image1.data.numpy(), cmap=mpl.cm.bone)
    ax[0].imshow(mask1, interpolation="nearest", cmap=mpl.cm.jet_r, alpha=0.7)

    ax[1].imshow(image2.data.numpy(), cmap=mpl.cm.bone)
    ax[1].imshow(mask2, interpolation="nearest", cmap=mpl.cm.ocean, alpha=0.7)

    plt.savefig("visualization/{}/{}".format(opt.name, name))


def visualize():

    # make directory for storing images.
    images_path = os.path.join("visualization", opt.name)
    os.makedirs(images_path, exist_ok=True)

    batcher = Batcher(batch_size=2)

    # initialise the model
    discriminator = ArcBinaryClassifier(num_glimpses=opt.numGlimpses,
                                        glimpse_h=opt.glimpseSize,
                                        glimpse_w=opt.glimpseSize,
                                        controller_out=opt.numStates)
    discriminator.load_state_dict(torch.load("saved_models/{}/{}".format(opt.name, opt.load)))

    arc = discriminator.arc

    sz = 30
    X, Y = batcher.fetch_batch("train", batch_size=sz)
    pred = discriminator(X)
    bce = torch.nn.BCEWithLogitsLoss()
    loss = bce(pred, Y.float())

    print("loss is {}".format(loss))

    same_pred = pred[sz // 2:].data.numpy()[:, 0]
    mx = same_pred.argsort()[len(same_pred) // 2]
    val = same_pred[mx]
    print(mx, val)


    same = mx + sz // 2

    img1, img2 = X[same]
    all_hidden = arc._forward(X)
    my_hidden = all_hidden[same]
    first_h = []
    second_h = []

    for turn in range(len(my_hidden)):
        if turn % 2 == 1:  # the first image outputs the hidden state for the next image
            first_h.append(my_hidden[turn])
        else:
            second_h.append(my_hidden[turn])

    num_glimpses = len(my_hidden) // 2

    for i in range(num_glimpses):
        d1 = img1
        d2 = img2
        gp1 = torch.tanh(arc.glimpser(first_h[i][None, :]))
        gp2 = torch.tanh(arc.glimpser(second_h[i][None, :]))
        o1 = arc.glimpse_window.get_attention_mask(gp1, mask_h=opt.imageSize, mask_w=opt.imageSize)[0]
        o2 = arc.glimpse_window.get_attention_mask(gp2, mask_h=opt.imageSize, mask_w=opt.imageSize)[0]
        display(d1, o1, d2, o2, "img_{}".format(i))


if __name__ == "__main__":
    visualize()
