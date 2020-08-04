import cv2
import os
import glob
import argparse


def get_file_paths(folder):
    image_file_paths = []
    for root, dirs, filenames in os.walk(folder):
        filenames = sorted(filenames)
        for filename in filenames:
            input_path = os.path.abspath(root)
            file_path = os.path.join(input_path, filename)
            if filename.endswith('.png') or filename.endswith('.jpg'):
                image_file_paths.append(file_path)

        break  # prevent descending into subfolders
    return image_file_paths


SF = 1.05
N = 3


def detect_cat(img_path, cat_cascade, output_dir, ratio=0.05, border_ratio=0.25):
    print('processing {}'.format(img_path))
    output_width = 286
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, W = img.shape[0], img.shape[1]
    minH = int(H * ratio)
    minW = int(W * ratio)
    cats = cat_cascade.detectMultiScale(gray, scaleFactor=SF, minNeighbors=N, minSize=(minH, minW))

    for cat_id, (x, y, w, h) in enumerate(cats):
        x1 = max(0, x - w * border_ratio)
        x2 = min(W, x + w * (1 + border_ratio))
        y1 = max(0, y - h * border_ratio)
        y2 = min(H, y + h * (1 + border_ratio))
        img_crop = img[int(y1):int(y2), int(x1):int(x2)]
        img_name = os.path.basename(img_path)
        out_path = os.path.join(output_dir, img_name.replace('.jpg', '_cat%d.jpg' % cat_id))
        print('write', out_path)
        img_crop = cv2.resize(img_crop, (output_width, output_width), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(out_path, img_crop, [int(cv2.IMWRITE_JPEG_QUALITY), 100])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='detecting cat faces using opencv detector')
    parser.add_argument('--input_dir', type=str,  help='input image directory')
    parser.add_argument('--output_dir', type=str, help='wihch directory to store cropped cat faces')
    parser.add_argument('--use_ext', action='store_true', help='if use haarcascade_frontalcatface_extended or not')
    args = parser.parse_args()

    if args.use_ext:
        cat_cascade = cv2.CascadeClassifier('haarcascade_frontalcatface.xml')
    else:
        cat_cascade = cv2.CascadeClassifier('haarcascade_frontalcatface_extended.xml')
    img_paths = get_file_paths(args.input_dir)
    print('total number of images {} from {}'.format(len(img_paths), args.input_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    for img_path in img_paths:
        detect_cat(img_path, cat_cascade, args.output_dir)
