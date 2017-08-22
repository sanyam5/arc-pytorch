import torch
import torch.nn as nn
import torch.nn.functional as F
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

        self.lstm = nn.LSTMCell(input_size=(glimpse_h * glimpse_w), hidden_size=self.lstm_out)
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

        all_hidden = self._forward(image_pairs)  # (B, 2*num_glimpses, lstm_out)
        last_hidden = all_hidden[:, -1, :]  # (B, lstm_out)

        return last_hidden

    def _forward(self, image_pairs: Variable) -> Variable:
        """
        Same as as forward except that it returns all hidden states instead of just the last one.

        Args:
            image_pairs: (B, 2, h, w) A batch of pairs of images

        Returns:
            (B, lstm_out) A batch of all hidden states.

        """

        # convert to images to float.
        image_pairs = image_pairs.float()

        batch_size = image_pairs.size()[0]

        all_hidden = []

        # initial hidden state of the LSTM.
        Hx = Variable(torch.zeros(batch_size, self.lstm_out))  # (B, lstm_out)
        Cx = Variable(torch.zeros(batch_size, self.lstm_out))  # (B, lstm_out)

        for turn in range(2*self.num_glimpses):  # we have num_glimpses glimpses for each image in the pair.
            # select image to show, alternate between the first and second image in the pair
            images_to_observe = image_pairs[:,  turn % 2]  # (B, h, w)

            # choose a portion from image to glimpse using attention
            glimpses = self.get_glimpse(images_to_observe, Hx)  # (B, glimpse_h, glimpse_w)
            flattened_glimpses = glimpses.view(batch_size, -1)  # (B, glimpse_h * glimpse_w), one time-step

            # feed the glimpses and the previous hidden state to the LSTM.
            Hx, Cx = self.lstm(flattened_glimpses, (Hx, Cx))  # (B, lstm_out), (B, lstm_out)

            # append this hidden state to all states
            all_hidden.append(Hx)

        all_hidden = torch.stack(all_hidden)  # (2*num_glimpses, B, lstm_out)
        all_hidden = all_hidden.transpose(0, 1)  # (B, 2*num_glimpses, lstm_out)

        # return a batch of all hidden states.
        return all_hidden

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

    def get_attention_mask(self, Hx: Variable, mask_h: int=32, mask_w: int=32) -> Variable:
        """
        Draw (approximately) the area focused by attention for display purposes.

        Args:
            Hx: (B, hx) A batch of hidden states
            mask_h: int Height of the mask to draw
            mask_w: int Width of the mask to draw

        Returns:
            (B, mask_h, mask_w): A batch of masks with attended portions weighted more.

        """

        batch_size, _ = Hx.size()

        glimpse_params = torch.tanh(self.glimpser(Hx))  # (B, 3)  a batch of glimpse params (x, y, delta)

        # (B, image_h, glimpse_h)
        F_h = self.get_filterbank(delta_caps=glimpse_params[:, 2], center_caps=glimpse_params[:, 0],
                                  from_length=mask_h, to_length=self.glimpse_h)

        # (B, image_w, glimpse_w)
        F_w = self.get_filterbank(delta_caps=glimpse_params[:, 2], center_caps=glimpse_params[:, 1],
                                  from_length=mask_w, to_length=self.glimpse_w)

        # (B, glimpse_h, glimpse_w)
        glimpse_proxy = Variable(torch.ones(batch_size, self.glimpse_h, self.glimpse_w))

        # find the attention mask that lead to the glimpse.
        mask = glimpse_proxy
        mask = torch.bmm(F_h, mask)
        mask = torch.bmm(mask, F_w.transpose(1, 2))

        # scale to between 0 and 1.0
        mask = mask - mask.min()
        mask = mask / mask.max()
        mask = mask.float()

        return mask


class Discriminator(nn.Module):

    def __init__(self, num_glimpses: int=8, glimpse_h: int=8, glimpse_w: int=8, lstm_out: int = 128):
        super().__init__()
        self.arc = ARC(
            num_glimpses=num_glimpses,
            glimpse_h=glimpse_h,
            glimpse_w=glimpse_w,
            lstm_out=lstm_out)

        # three dense layers, gradually toning the states down.
        self.dense1 = nn.Linear(lstm_out, 64)
        self.dense2 = nn.Linear(64, 1)

    def forward(self, image_pairs: Variable) -> Variable:
        arc_out = self.arc(image_pairs)

        # not putting sigmoid here, use sigmoid in the loss function.
        d1 = F.elu(self.dense1(arc_out))
        decision = self.dense2(d1)

        return decision

    def save_to_file(self, file_path: str) -> None:
        torch.save(self.state_dict(), file_path)
