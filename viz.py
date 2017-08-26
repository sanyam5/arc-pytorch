
from models import ARC, ArcBinaryClassifier
from batcher import Batcher
import torch

batcher = Batcher(batch_size=2)


exp_name = "16_4_4_256"

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


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

    plt.savefig("visualization/{}/{}".format(exp_name, name))
    plt.show()


discriminator = ArcBinaryClassifier(num_glimpses=16, glimpse_h=4, glimpse_w=4, controller_out=256)
mod = torch.load("saved_models/{}/{}".format(exp_name, "best"))
discriminator.load_state_dict(mod)
arc = discriminator.arc


sz = 50
X, Y = batcher.fetch_batch("train", batch_size=sz)
pred = discriminator(X)
bce = torch.nn.BCEWithLogitsLoss()
loss = bce(pred, Y.float())

ard_pred = (pred > 0.5).int()

print("loss is {}".format(loss))

all_hidden = arc._forward(X)

same_pred = pred[sz // 2:].data.numpy()[:, 0]
mx = same_pred.argsort()[len(same_pred) // 2]
val = same_pred[mx]
print(mx, val)

# In[140]:


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
    o1 = arc.get_attention_mask(Hx=first_h[i][None, :])[0]
    o2 = arc.get_attention_mask(Hx=second_h[i][None, :])[0]
    display(d1, o1, d2, o2, "img_g_{}".format(i))
