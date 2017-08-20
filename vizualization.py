from models import ARC, Discriminator
from batcher import Batcher
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt


batcher = Batcher(batch_size=2)


def display(image, name = "hola.png"):
    plt.imshow(image.data.numpy())
    plt.savefig("images/{}".format(name))
    plt.show()


disc = Discriminator(num_glimpses=6, lstm_out=128)
disc.load_state_dict(torch.load("saved_models/<model-name>"))
arc = disc.arc

X, Y = batcher.fetch_batch("test")
all_hidden = arc._forward(X)

same = 1

img1, img2 = X[same]
my_hidden = all_hidden[same]
first_h = []
second_h = []

for turn in range(len(my_hidden)):
    if turn % 2 == 0:
        first_h.append(my_hidden[turn])
    else:
        second_h.append(my_hidden[turn])

num_glimpses = len(my_hidden) // 2

for i in range(num_glimpses):
    d1 = arc.draw_attention(images=img1[None, :, :], Hx=first_h[i][None, :])[0]
    d2 = arc.draw_attention(images=img2[None, :, :], Hx=second_h[i][None, :])[0]

    display(d1, "img1_g_{}".format(i))
    display(d2, "img2_g_{}".format(i))
