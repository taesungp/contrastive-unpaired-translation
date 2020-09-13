import numpy as np
import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import util.util as util


class SingleImageDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)

        self.dir_A = os.path.join(opt.dataroot, 'trainA')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, 'trainB')  # create a path '/path/to/data/trainB'

        if os.path.exists(self.dir_A) and os.path.exists(self.dir_B):
            self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
            self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B

        assert len(self.A_paths) == 1 and len(self.B_paths) == 1,\
            "SingleImageDataset class should be used with one image in each domain"
        A_img = Image.open(self.A_paths[0]).convert('RGB')
        B_img = Image.open(self.B_paths[0]).convert('RGB')
        print("Image sizes %s and %s" % (str(A_img.size), str(B_img.size)))

        self.A_img = A_img
        self.B_img = B_img

        # In single-image translation, we augment the data loader by applying
        # random scaling. Still, we design the data loader such that the
        # amount of scaling is the same within a minibatch. To do this,
        # we precompute the random scaling values, and repeat them by |batch_size|.
        A_zoom = 1 / self.opt.random_scale_max
        zoom_levels_A = np.random.uniform(A_zoom, 1.0, size=(len(self) // opt.batch_size + 1, 1, 2))
        self.zoom_levels_A = np.reshape(np.tile(zoom_levels_A, (1, opt.batch_size, 1)), [-1, 2])

        B_zoom = 1 / self.opt.random_scale_max
        zoom_levels_B = np.random.uniform(B_zoom, 1.0, size=(len(self) // opt.batch_size + 1, 1, 2))
        self.zoom_levels_B = np.reshape(np.tile(zoom_levels_B, (1, opt.batch_size, 1)), [-1, 2])

        # While the crop locations are randomized, the negative samples should
        # not come from the same location. To do this, we precompute the
        # crop locations with no repetition.
        self.patch_indices_A = list(range(len(self)))
        random.shuffle(self.patch_indices_A)
        self.patch_indices_B = list(range(len(self)))
        random.shuffle(self.patch_indices_B)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[0]
        B_path = self.B_paths[0]
        A_img = self.A_img
        B_img = self.B_img

        # apply image transformation
        if self.opt.phase == "train":
            param = {'scale_factor': self.zoom_levels_A[index],
                     'patch_index': self.patch_indices_A[index],
                     'flip': random.random() > 0.5}

            transform_A = get_transform(self.opt, params=param, method=Image.BILINEAR)
            A = transform_A(A_img)

            param = {'scale_factor': self.zoom_levels_B[index],
                     'patch_index': self.patch_indices_B[index],
                     'flip': random.random() > 0.5}
            transform_B = get_transform(self.opt, params=param, method=Image.BILINEAR)
            B = transform_B(B_img)
        else:
            transform = get_transform(self.opt, method=Image.BILINEAR)
            A = transform(A_img)
            B = transform(B_img)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """ Let's pretend the single image contains 100,000 crops for convenience.
        """
        return 100000
