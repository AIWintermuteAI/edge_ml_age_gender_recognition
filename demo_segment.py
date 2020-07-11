from pathlib import Path
import cv2
import random
import numpy as np
import argparse
from contextlib import contextmanager
import keras
from keras.utils.data_utils import get_file

import tensorflow as tf

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
config = tf.ConfigProto(gpu_options=gpu_options)
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


class_colors = [(random.randint(0, 255), random.randint(
    0, 255), random.randint(0, 255)) for _ in range(5000)]

def get_args():
    parser = argparse.ArgumentParser(description="This script detects faces from web cam input, "
                                                 "and estimates age and gender for the detected faces.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--weight_file", type=str, default='Segnet_best_val_loss.h5',
                        help="path to weight file (e.g. weights.28-3.73.hdf5)")
    parser.add_argument("--image_dir", type=str, default=None,
                        help="target image directory; if set, images in image_dir are used instead of webcam")
    parser.add_argument("--labels", type=list, default=['background','person'],
                        help="target image directory; if set, images in image_dir are used instead of webcam")
    args = parser.parse_args()
    return args

@contextmanager
def video_capture(*args, **kwargs):
    cap = cv2.VideoCapture(*args, **kwargs)
    try:
        yield cap
    finally:
        cap.release()


def yield_images():
    # capture video
    with video_capture(0) as cap:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while True:
            # get video frame
            ret, img = cap.read()

            if not ret:
                raise RuntimeError("Failed to capture image")

            yield img


def yield_images_from_dir(image_dir):
    image_dir = Path(image_dir)

    for image_path in image_dir.glob("*.*"):
        img = cv2.imread(str(image_path), 1)

        if img is not None:
            h, w, _ = img.shape
            r = 640 / max(w, h)
            yield cv2.resize(img, (int(w * r), int(h * r)))

def overlay_seg_image( inp_img , seg_img  ):
    orininal_h = inp_img.shape[0]
    orininal_w = inp_img.shape[1]
    seg_img = cv2.resize(seg_img, (orininal_w, orininal_h))

    fused_img = (inp_img/2 + seg_img/2 ).astype('uint8')
    return fused_img 

def visualize_segmentation(seg_arr, inp_img=None, n_classes=None , 
    colors=class_colors, class_names=None,overlay_img=False, show_legends=False , 
    prediction_width=None, prediction_height=None):
    
    print("Found the following classes in the segmentation image:", np.unique(seg_arr))

    if n_classes is None:
        n_classes = np.max(seg_arr)

    seg_img = get_colored_segmentation_image(seg_arr, n_classes, colors=colors)

    if not inp_img is None:
        orininal_h = inp_img.shape[0]
        orininal_w = inp_img.shape[1]
        seg_img = cv2.resize(seg_img, (orininal_w, orininal_h))


    if (not prediction_height is None) and  (not prediction_width is None):
        seg_img = cv2.resize(seg_img, (prediction_width, prediction_height ))
        if not inp_img is None:
            inp_img = cv2.resize(inp_img, (prediction_width, prediction_height ))


    if overlay_img:
        assert not inp_img is None
        seg_img = overlay_seg_image(inp_img , seg_img)


    if show_legends:
        assert not class_names is None
        legend_img = get_legends(class_names , colors=colors )

        seg_img = concat_lenends( seg_img , legend_img )

    return seg_img

def get_image_array(image_input, width, height, imgNorm="sub_mean", ordering='channels_last'):
    """ Load image array from input """
    if type(image_input) is np.ndarray:
        # It is already an array, use it as it is
        img = image_input
    elif  isinstance(image_input, six.string_types)  :
        if not os.path.isfile(image_input):
            raise DataLoaderError("get_image_array: path {0} doesn't exist".format(image_input))
        img = cv2.imread(image_input, 1)
    else:
        raise DataLoaderError("get_image_array: Can't process input type {0}".format(str(type(image_input))))

    img = cv2.resize(img, (width, height))
    img = img.astype(np.float32)
    img = img / 255.
    img = img - 0.5
    img = img * 2.
    img = img[:, :, ::-1]

    if ordering == 'channels_first':
        img = np.rollaxis(img, 2, 0)
    return img

def get_colored_segmentation_image(seg_arr, n_classes, colors=class_colors):
    output_height = seg_arr.shape[0]
    output_width = seg_arr.shape[1]

    seg_img = np.zeros((output_height, output_width, 3))

    for c in range(n_classes):
        seg_img[:, :, 0] += ((seg_arr[:, :] == c)*(colors[c][0])).astype('uint8')
        seg_img[:, :, 1] += ((seg_arr[:, :] == c)*(colors[c][1])).astype('uint8')
        seg_img[:, :, 2] += ((seg_arr[:, :] == c)*(colors[c][2])).astype('uint8')

    return seg_img 

def main():
    args = get_args()
    weight_file = args.weight_file
    image_dir = args.image_dir
    labels = args.labels

    # load model and weights
    model = keras.models.load_model(weight_file, compile=False)
    output_width, output_height = model.outputs[0].shape[1:3]
    input_width, input_height = model.inputs[0].shape[1:3]
    n_classes = len(labels)

    image_generator = yield_images_from_dir(image_dir) if image_dir else yield_images()

    for img in image_generator:
        orininal_h, orininal_w, _ = np.shape(img)

        x = get_image_array(img, input_width, input_height)
        pr = model.predict(np.array([x]))[0]
        pr = pr.argmax(axis=2)
        seg_img = visualize_segmentation(pr, img, n_classes=None, colors=class_colors, overlay_img=True, show_legends=False, class_names=None, prediction_width=orininal_w, prediction_height=orininal_h)

        cv2.imshow("result", seg_img)
        key = cv2.waitKey(-1) if image_dir else cv2.waitKey(30)

        if key == 27:  # ESC
            break


if __name__ == '__main__':
    main()
