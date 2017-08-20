import torch
import torch.nn as nn
import torch.functional as F
from torch.autograd import Variable
import math
from typing import Tuple


class ARC(nn.Module):

    def __init__(self, num_glimpses: int=8, glimpse_h: int=8, glimpse_w: int=8, lstm_out: int=128) -> None:
        super().__init__()
        self.num_glimpses = num_glimpses
        self.glimpse_h = glimpse_h
        self.glimpse_w = glimpse_w
        self.lstm_out = lstm_out
        self.num_out = self.lstm_out

        self.lstm = nn.LSTM(input_size=(glimpse_h * glimpse_w), hidden_size=self.lstm_out)
        self.glimpser = nn.Linear(in_features=self.lstm_out, out_features=3)  # three glimpse params --> h, w, delta

    def forward(self, image_pairs: Variable) -> Variable:
        """
        The main forward method of the ARC module
        Args:
            image_pairs: (B, 2, h, w) A batch of pairs of images

        Returns:
            (B, lstm_out) A batch of final hidden states after each pair of image has been shown for num_glimpses
            glimpses.

        """
        # convert to images to float.
        image_pairs = image_pairs.float()

        batch_size = image_pairs.size()[0]

        # initial hidden state of the LSTM.
        Hx = Variable(torch.zeros(1, batch_size, self.lstm_out))  # (1, B, lstm_out)

        for turn in range(2*self.num_glimpses):  # we have num_glimpses glimpses for each image in the pair.
            # select image to show, alternate between the first and second image in the pair
            images_to_observe = image_pairs[:,  turn % 2]  # (B, h, w)

            # choose a portion from image to glimpse using attention
            glimpses = self.get_glimpse(images_to_observe, Hx.view(batch_size, -1))  # (B, glimpse_h, glimpse_w)
            flattened_glimpses = glimpses.view(1, batch_size, -1)  # (B, 1, glimpse_h * glimpse_w), one time-step

            # select hidden state to pass
            if turn == 0:  # if this the first turn of the LSTM, we let it initialize its own hidden state.
                last_hidden = None
            else:  # for later turns we supply the hidden state from the previous turn
                last_hidden = Hx, Cx

            # feed the glimpses and the previous hidden state to the LSTM.
            Hx, (_, Cx) = self.lstm(flattened_glimpses, last_hidden)

        # return a batch of last hidden states.
        return Hx.view(batch_size, self.lstm_out)

    def get_filterbank(self, delta_caps: Variable, center_caps: Variable, from_length: int, to_length: int) -> Variable:
        """

        Args:
            delta_caps: (B) A batch of deltas [-1, 1]
            center_caps: (B) A batch of [-1, 1] reals that dictate the location of center of cauchy kernel glimpse.
            from_length: int length of input image along that dimension
            to_length: int length of the filtered image along that dimension

        Returns:
            A Variable of size (B, from_length, to_length)

        """

        # convert dimension sizes to float. lots of math ahead.
        from_length = float(from_length)
        to_length = float(to_length)

        # scale the centers and the deltas to map to the actual size of given image.
        centers = (from_length - 1) * (center_caps + 1) / 2.0  # (B)
        deltas = (float(from_length)/to_length) * (1.0 - torch.abs(delta_caps))

        # calculate gamma for cauchy kernel
        gammas = torch.exp(1.0 - 2 * torch.abs(delta_caps))  # (B)

        # coordinate of pixels on the glimpse
        glimpse_pixels = Variable(torch.arange(0, to_length) - (to_length - 1.0) / 2.0)  # (to_length)
        # space out with delta
        glimpse_pixels = deltas[:, None] * glimpse_pixels[None, :]  # (B, to_length)
        # center around the centers
        glimpse_pixels = centers[:, None] + glimpse_pixels  # (B, to_length)

        # coordinates of pixels on the image
        image_pixels = Variable(torch.arange(0, from_length))  # (from_length)

        fx = image_pixels - glimpse_pixels[:, :, None]  # (B, to_length, from_length)
        fx = fx / gammas[:, None, None]
        fx = fx ** 2.0
        fx = 1.0 + fx
        fx = math.pi * gammas[:, None, None] * fx
        fx = 1.0 / fx
        fx = fx / (torch.sum(fx, dim=2) + 1e-4)[:, :, None]  # we add a small constant in the denominator division by 0.

        return fx.transpose(1, 2)

    def get_glimpse(self, images: Variable, Hx: Variable) -> Variable:
        """

        Args:
            images: (B, h, w) A batch of images
            Hx: (B, hx) A batch of hidden states

        Returns:
            (B, glimpse_h, glimpse_w) A batch of glimpses.

        """

        batch_size, image_h, image_w = images.size()

        glimpse_params = torch.tanh(self.glimpser(Hx))  # (B, 3)  a batch of glimpse params (x, y, delta)

        # (B, image_h, glimpse_h)
        F_h = self.get_filterbank(delta_caps=glimpse_params[:, 2], center_caps=glimpse_params[:, 0],
                                  from_length=image_h, to_length=self.glimpse_h)

        # (B, image_w, glimpse_w)
        F_w = self.get_filterbank(delta_caps=glimpse_params[:, 2], center_caps=glimpse_params[:, 1],
                                  from_length=image_w, to_length=self.glimpse_w)

        # F_h.T * images * F_w
        glimpses = images
        glimpses = torch.bmm(F_h.transpose(1, 2), glimpses)
        glimpses = torch.bmm(glimpses, F_w)

        return glimpses  # (B, glimpse_h, glimpse_w)


class Discriminator(nn.Module):

    def __init__(self):
        super().__init__()
        self.arc = ARC()
        self.dense = nn.Linear(self.arc.num_out, 1)

    def forward(self, image_pairs: Variable) -> Variable:
        arc_out = self.arc(image_pairs)

        # not putting sigmoid here, use sigmoid in the loss function.
        decision = self.dense(arc_out)
        return decision
