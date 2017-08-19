from models import ARC
import numpy as np
import torch
from torch.autograd import Variable

langs = np.load("data/images_background.npy")


def test_arc() -> None:
    arc = ARC()


    batch = [
        [langs[0][0][0], langs[0][0][1]],  # similar
        [langs[0][0][0], langs[0][1][0]],  # same language dissimilar
        [langs[0][0][0], langs[1][1][0]],  # different language dissimilar
    ]

    batch = Variable(torch.from_numpy(np.array(batch)))

    h = arc(batch)

    assert h.size()[0] == len(batch)
    assert h.size()[1] == arc.lstm_out

