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
        else:
            imagepaths = make_dataset(opt.dataroot, opt.max_dataset_size)
            self.A_paths = [p for p in imagepaths if "_ref" not in p]
            self.B_paths = [p for p in imagepaths if "_ref" in p]
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B

        A_img = Image.open(self.A_paths[0]).convert('RGB')
        B_imgs = [Image.open(self.B_paths[i]).convert('RGB') for i in range(self.B_size)]
        print("Image sizes %s and %s" % (str(A_img.size), str(B_imgs[0].size)))
        # if A_img.size[0] >= opt.load_size:
        A_img_size = opt.load_size * 2 if self.opt.use_balanced_zooming else opt.load_size
        A_img = A_img.resize((A_img_size, int(round(A_img.size[1] * A_img_size / A_img.size[0]))), Image.BICUBIC)
        resized_B_imgs = []
        for B_img in B_imgs:
            if B_img.size[0] >= opt.load_size:
                resized_B_imgs.append(B_img.resize((opt.load_size, int(round(B_img.size[1] * opt.load_size / B_img.size[0]))), Image.BICUBIC))
            else:
                resized_B_imgs.append(B_img)
        B_imgs = resized_B_imgs
        self.B_imgs = B_imgs

        if "rotate" in opt.preprocess:
            rotated_images = []
            for B_img in self.B_imgs:
                for angle in [0, 90, 180, 270]:
                    rotated_images.append(B_img.rotate(angle))
            self.B_imgs = rotated_images

        self.A_img = A_img
        self.B_img = B_imgs[0]
        larger_size = max(A_img.size[0], self.B_img.size[0])

        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        self.seed_prime_number_A = 104729
        self.seed_prime_number_B = 104723
        if opt.phase == "train":
            self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1), method=Image.BILINEAR)
            self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1), method=Image.BILINEAR)
        else:
            self.transform_A = get_transform(util.copyconf(self.opt, preprocess="scale_width", no_flip=True),
                                             grayscale=(input_nc == 1), method=Image.BILINEAR)
            # self.transform_B = get_transform(util.copyconf(self.opt, preprocess="scale_width"),
            #                                 grayscale=(output_nc == 1), method=Image.BILINEAR)

        max_zoom = 384 / 1024  # just arbitrary number
        A_zoom = max_zoom
        # A_zoom = min(0.9, (max_zoom * larger_size) / A_img.size[0]) if opt.use_balanced_zooming else max_zoom
        zoom_levels_A = np.random.uniform(A_zoom, 1.0, size=(1000 // opt.batch_size + 1, 1, 2))
        self.zoom_levels_A = np.reshape(np.tile(zoom_levels_A, (1, opt.batch_size, 1)), [-1, 2])
        B_zoom = max_zoom
        # B_zoom = min(0.9, (max_zoom * larger_size) / B_img.size[0]) if opt.use_balanced_zooming else max_zoom
        zoom_levels_B = np.random.uniform(B_zoom, 1.0, size=(1000 // opt.batch_size + 1, 1, 2))
        self.zoom_levels_B = np.reshape(np.tile(zoom_levels_B, (1, opt.batch_size, 1)), [-1, 2])

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
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]


<< << << < HEAD
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
== == == =
        A_img = self.A_img
        B_img = self.B_imgs[index_B]
>>>>>> > 6b836b5c63eeb2bb9719662acce3c4c62960a37a

        # apply image transformation
        if self.opt.phase == "train":
            param = {'scale_factor': self.zoom_levels_A[index],
                     'patch_index': self.seed_prime_number_A * (index + 1),
                     'flip': random.random() > 0.5
                     }

            # print("%04d\t%.2f\t%04d" % (index, param['scale_factor'], param['patch_index']))
            transform_A = get_transform(self.opt, params=param, method=Image.BILINEAR)
            A = transform_A(A_img)

            param = {'scale_factor': self.zoom_levels_B[index],
                     'patch_index': self.seed_prime_number_B * (index + 1),
                     'flip': random.random() > 0.5
<< << << < HEAD
                     }
            transform_B = get_transform(self.opt, params=param)
== == == =
            }
            transform_B = get_transform(self.opt, params=param, method=Image.BILINEAR)
>> >>>> > 6b836b5c63eeb2bb9719662acce3c4c62960a37a
            B = transform_B(B_img)
        else:
            A = self.transform_A(A_img)
            param = {'size': (A.size(1), A.size(2)),
                     'flip': False}
            transform_B = get_transform(util.copyconf(self.opt, preprocess="fixsize"),
                                        params = param, method = Image.BILINEAR)
            B=transform_B(B_img)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return 1000
