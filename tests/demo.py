from pathlib import Path
import cv2
import dlib
import numpy as np
import argparse
from contextlib import contextmanager

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import get_file

#left from original code predicting a number 0 -100 and attempts to use age groups
#ages = np.arange(0, 101).reshape(101, 1)
ages=['0-10', '11-20', '21-45', '46-60', '60-100']

pretrained_model = "https://files.seeedstudio.com/ml/age_gender/weights.46-0.73.hdf5"
modhash = '90875017ac84552e4f1a828629053064f3d3ebf5'

def get_args():
    parser = argparse.ArgumentParser(description="This script detects faces from web cam input, "
                                                 "and estimates age and gender for the detected faces.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model_file", type=str, default=None,
                        help="path to weight file (e.g. weights.28-3.73.hdf5)")
    parser.add_argument("--margin", type=float, default=0.4,
                        help="margin around detected face for age-gender estimation")
    parser.add_argument("--image_dir", type=str, default=None,
                        help="target image directory; if set, images in image_dir are used instead of webcam")
    args = parser.parse_args()
    return args


def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
               font_scale=0.8, thickness=1):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
    cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness, lineType=cv2.LINE_AA)

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


def main():
    args = get_args()

    model_file = args.model_file
    margin = args.margin
    image_dir = args.image_dir

    if not model_file:
        model_file = get_file("MobileFaceNet_128_2_0-73.hdf5", pretrained_model, cache_subdir="pretrained_models",
                               file_hash=modhash, cache_dir=str(Path(__file__).resolve().parent))

    # for face detection
    detector = dlib.get_frontal_face_detector()

    # load model and weights
    img_size = 128
    model = load_model(model_file)
    image_generator = yield_images_from_dir(image_dir) if image_dir else yield_images()

    for img in image_generator:
        input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_h, img_w, _ = np.shape(input_img)

        # detect faces using dlib detector
        detected = detector(input_img, 1)
        faces = np.empty((len(detected), img_size, img_size, 3), dtype=np.float32)

        if len(detected) > 0:
            for i, d in enumerate(detected):
                x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
                xw1 = max(int(x1 - margin * w), 0)
                yw1 = max(int(y1 - margin * h), 0)
                xw2 = min(int(x2 + margin * w), img_w - 1)
                yw2 = min(int(y2 + margin * h), img_h - 1)

                cv2.rectangle(img, (xw1, yw1), (xw2, yw2), (122, 122, 0), 2)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

                faces[i, :, :, :] = cv2.resize(img[yw1:yw2 + 1, xw1:xw2 + 1, :], (img_size, img_size))
                faces[i, :, :, :] = faces[i, :, :, :] / 255.
                faces[i, :, :, :] = faces[i, :, :, :] - 0.5
                faces[i, :, :, :] = faces[i, :, :, :] * 2.

            # predict ages and genders of the detected faces
            results = model.predict(faces)
            predicted_genders = results[0]
            predicted_ages = np.argmax(results[1], axis = 1)

            #left from original code predicting a number 0 -100
            #predicted_ages = results[1].dot(ages).flatten()

            # draw results
            for i, d in enumerate(detected):
                #left from original code predicting a number 0 -100
                label = "{}, {}".format(ages[predicted_ages[i]], "M" if predicted_genders[i][0] < 0.5 else "F")
                #label = "{}, {}".format(int(predicted_ages[i]*10), "M" if predicted_genders[i] < 0.5 else "F")
                draw_label(img, (d.left(), d.top()), label)

        cv2.imshow("result", img)
        key = cv2.waitKey(-1) if image_dir else cv2.waitKey(30)

        if key == 27:  # ESC
            break


if __name__ == '__main__':
    main()
