from models import ARC
import numpy as np
import torch
from torch.autograd import Variable

langs = np.load("data/images_background.npy")


def test_arc() -> None:
    arc = ARC()

    batch = [
        [langs[0][0], langs[0][1]],  # similar
        [langs[0][0], langs[1][0]],  # same language dissimilar
        [langs[0][0], langs[-1][0]],  # different language dissimilar
    ]

    batch = Variable(torch.from_numpy(np.array(batch)))

    h = arc(batch)

    assert h.size()[0] == len(batch)
    assert h.size()[1] == arc.lstm_out


def test_glimpse() -> None:
    arc = ARC(glimpse_h=2, glimpse_w=2, lstm_out=4)

    images_zero = Variable(torch.zeros(10, 8, 8)).float()  # 10 8x8 images of all zeros
    images_one = Variable(torch.ones(10, 8, 8)).float()  # 10 8x8 images of all ones
    Hx_zero = Variable(torch.zeros(10, 4)).float()

    assert arc.glimpse_window.get_glimpse(images_zero, Hx_zero).max().data[0] == 0
    assert arc.glimpse_window.get_glimpse(images_one, Hx_zero).min().data[0] > .9
