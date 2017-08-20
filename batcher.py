"""
taken and modified from https://github.com/pranv/ARC
"""

import numpy as np
from numpy.random import choice
import torch
from torch.autograd import Variable

from scipy.misc import imresize as resize

from image_augmenter import ImageAugmenter


class Omniglot(object):
    def __init__(self, path='data/omniglot.npy', batch_size=128, image_size=32):
        """
        path: path to omniglot.npy file produced by "data/setup_omniglot.py" script
        batch_size: the output is (2 * batch size, 1, image_size, image_size)
                    X[i] & X[i + batch_size] are the pair
        image_size: size of the image
        data_split: in number of alphabets, e.g. [30, 10] means out of 50 Omniglot characters,
                    30 is for training, 10 for validation and the remaining(10) for testing
        within_alphabet: for verfication task, when 2 characters are sampled to form a pair,
                        this flag specifies if should they be from the same alphabet/language
        ---------------------
        Data Augmentation Parameters:
            flip: here flipping both the images in a pair
            scale: x would scale image by + or - x%
            rotation_deg
            shear_deg
            translation_px: in both x and y directions
        """
        chars = np.load(path)

        # resize the images
        resized_chars = np.zeros((1623, 20, image_size, image_size), dtype='uint8')
        for i in range(1623):
            for j in range(20):
                resized_chars[i, j] = resize(chars[i, j], (image_size, image_size))
        chars = resized_chars

        self.mean_pixel = chars.mean() / 255.0  # used later for mean subtraction

        # starting index of each alphabet in a list of chars
        a_start = [0, 20, 49, 75, 116, 156, 180, 226, 240, 266, 300, 333, 355, 381,
                   424, 448, 496, 518, 534, 586, 633, 673, 699, 739, 780, 813,
                   827, 869, 892, 909, 964, 984, 1010, 1036, 1062, 1088, 1114,
                   1159, 1204, 1245, 1271, 1318, 1358, 1388, 1433, 1479, 1507,
                   1530, 1555, 1597]

        # size of each alphabet (num of chars)
        a_size = [20, 29, 26, 41, 40, 24, 46, 14, 26, 34, 33, 22, 26, 43, 24, 48, 22,
                  16, 52, 47, 40, 26, 40, 41, 33, 14, 42, 23, 17, 55, 20, 26, 26, 26,
                  26, 26, 45, 45, 41, 26, 47, 40, 30, 45, 46, 28, 23, 25, 42, 26]

        # each alphabet/language has different number of characters.
        # in order to uniformly sample all characters, we need weigh the probability
        # of sampling a alphabet by its size. p is that probability
        def size2p(size):
            s = np.array(size).astype('float64')
            return s / s.sum()

        self.size2p = size2p

        self.data = chars
        self.a_start = a_start
        self.a_size = a_size
        self.image_size = image_size
        self.batch_size = batch_size

        flip = True
        scale = 0.2
        rotation_deg = 20
        shear_deg = 10
        translation_px = 5
        self.augmentor = ImageAugmenter(image_size, image_size,
                                        hflip=flip, vflip=flip,
                                        scale_to_percent=1.0 + scale, rotation_deg=rotation_deg, shear_deg=shear_deg,
                                        translation_x_px=translation_px, translation_y_px=translation_px)

    def fetch_batch(self, part):
        """
            This outputs batch_size number of pairs
            Thus the actual number of images outputted is 2 * batch_size
            Say A & B form the half of a pair
            The Batch is divided into 4 parts:
                Dissimilar A 		Dissimilar B
                Similar A 			Similar B

            Corresponding images in Similar A and Similar B form the similar pair
            similarly, Dissimilar A and Dissimilar B form the dissimilar pair

            When flattened, the batch has 4 parts with indices:
                Dissimilar A 		0 - batch_size / 2
                Similar A    		batch_size / 2  - batch_size
                Dissimilar B 		batch_size  - 3 * batch_size / 2
                Similar B 			3 * batch_size / 2 - batch_size

        """
        pass


class Batcher(Omniglot):
    def __init__(self, path='data/omniglot.npy', batch_size=128, image_size=32):
        Omniglot.__init__(self, path, batch_size, image_size)

        a_start = self.a_start
        a_size = self.a_size

        # slicing indices for splitting a_start & a_size
        i = 20
        j = 30
        starts = {}
        starts['train'], starts['val'], starts['test'] = a_start[:i], a_start[i:j], a_start[j:]
        sizes = {}
        sizes['train'], sizes['val'], sizes['test'] = a_size[:i], a_size[i:j], a_size[j:]

        size2p = self.size2p

        p = {}
        p['train'], p['val'], p['test'] = size2p(sizes['train']), size2p(sizes['val']), size2p(sizes['test'])

        self.starts = starts
        self.sizes = sizes
        self.p = p

    def fetch_batch(self, part):
        X, Y = self._fetch_batch(part)

        X = Variable(torch.from_numpy(X)).view(2*self.batch_size, self.image_size, self.image_size)

        X1 = X[:self.batch_size]  # (B, h, w)
        X2 = X[self.batch_size:]  # (B, h, w)

        X = torch.stack([X1, X2], dim=1)  # (B, 2, h, w)

        Y = Variable(torch.from_numpy(Y))
        return X, Y

    def _fetch_batch(self, part):
        data = self.data
        starts = self.starts[part]
        sizes = self.sizes[part]
        p = self.p[part]
        image_size = self.image_size
        batch_size = self.batch_size

        num_alphbts = len(starts)

        X = np.zeros((2 * batch_size, image_size, image_size), dtype='uint8')
        for i in range(batch_size // 2):
            # choose similar chars
            same_idx = choice(range(starts[0], starts[-1] + sizes[-1]))

            # choose dissimilar chars within alphabet
            alphbt_idx = choice(num_alphbts, p=p)
            char_offset = choice(sizes[alphbt_idx], 2, replace=False)
            diff_idx = starts[alphbt_idx] + char_offset

            X[i], X[i + batch_size] = data[diff_idx, choice(20, 2)]
            X[i + batch_size // 2], X[i + 3 * batch_size // 2] = data[same_idx, choice(20, 2, replace=False)]

        y = np.zeros((batch_size, 1), dtype='int32')
        y[:batch_size // 2] = 0
        y[batch_size // 2:] = 1

        if part == 'train':
            X = self.augmentor.augment_batch(X)
        else:
            X = X / 255.0

        X = X - self.mean_pixel
        X = X[:, np.newaxis]
        X = X.astype("float64")

        return X, y

