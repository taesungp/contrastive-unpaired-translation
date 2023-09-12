"""
Usage: python generate.py --dataroot ./datasets/grumpifycat --checkpoints_dir ./checkpoints --results_dir ./results --name grumpycat_CUT --CUT_mode CUT --phase train --epoch latest --num_test 50 --gpu_ids 0
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import util.util as util
import ntpath
import time
from tqdm import tqdm

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    train_dataset = create_dataset(util.copyconf(opt, phase="train"))
    model = create_model(opt)      # create a model given opt.model and other options
    image_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))

    start = time.time()
    print("Number of available images: %d" % len(dataset))
    image_amt = opt.num_test if opt.num_test > 0 else dataset.dataset.A_size  # amount of images to generate
    pbar = tqdm(total=int(image_amt))  # progress bar
    for i, data in enumerate(dataset):
        pbar.update(1)
        if i == 0:
            model.data_dependent_initialize(data)
            model.setup(opt)               # regular setup: load and print networks; create schedulers
            model.parallelize()
            if opt.eval:
                model.eval()
        if i >= image_amt:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths

        image_name = ntpath.basename(img_path[0])  # get image name
        im_data = visuals['fake_B']  # get generated fake image
        im = util.tensor2im(im_data)  # transform tensor to image
        os.makedirs(image_dir, exist_ok=True)
        save_path = os.path.join(image_dir, image_name)
        util.save_image(im, save_path, aspect_ratio=1.0)
    pbar.close()
    end = time.time()
    print("Generating %d images took %.2f seconds." % (image_amt, (end - start)))
