# -*- coding: utf-8 -*-
# @Time : 20-6-9 下午3:06
# @Author : zhuying
# @Company : Minivision
# @File : test.py
# @Software : PyCharm

import argparse
import os
import time
import warnings

import cv2
import numpy as np

from face_recog.models.face_boxes_location import FaceBoxesLocation
from silent_face.src.anti_spoof_predictor import AntiSpoofPredictor
from silent_face.src.generate_patches import CropImage
from silent_face.src.utility import parse_model_name

warnings.filterwarnings('ignore')

SAMPLE_IMAGE_PATH = "./images/sample/"

wanted_model_index = [0]


def test_webcam(model_dir, device_id):
    fbl = FaceBoxesLocation(weights='face_recog/weights/FaceBoxes.pth')
    models = []
    params = []
    for i, model_name in enumerate(os.listdir(model_dir)):
        if i not in wanted_model_index:
            continue
        models.append(AntiSpoofPredictor(device_id, os.path.join(model_dir, model_name)))
        params.append(parse_model_name(model_name))
        print(f"Load model {model_name}")
    image_cropper = CropImage()
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        ret, frame = cap.read()
    while ret:
        image_bboxes = fbl.face_location(frame)
        image_bboxes[:, 2:] = image_bboxes[:, 2:] - image_bboxes[:, :2]
        for image_bbox in image_bboxes.astype(int):
            prediction = np.zeros((1, 3))
            test_speed = 0
            # sum the prediction from single model's result
            start = time.time()
            for model, (h_input, w_input, model_type, scale) in zip(models, params):
                param = {
                    "org_img": frame,
                    "bbox": image_bbox,
                    "scale": scale,
                    "out_w": w_input,
                    "out_h": h_input,
                    "crop": True,
                }
                if scale is None:
                    param["crop"] = False
                img = image_cropper.crop(**param)
                prediction += model.predict(img)
            test_speed += time.time() - start

            # draw result of prediction
            label = np.argmax(prediction)
            value = prediction[0][label] / len(models)
            if label == 1:
                result_text = "RealFace Score: {:.2f}".format(value)
                color = (255, 0, 0)
            else:
                result_text = "FakeFace Score: {:.2f}".format(value)
                color = (0, 0, 255)
            print("Prediction cost {:.2f} s".format(test_speed))
            cv2.rectangle(
                frame,
                (image_bbox[0], image_bbox[1]),
                (image_bbox[0] + image_bbox[2], image_bbox[1] + image_bbox[3]),
                color, 2)
            cv2.putText(
                frame,
                result_text,
                (image_bbox[0], image_bbox[1] - 5),
                cv2.FONT_HERSHEY_COMPLEX, 0.5 * frame.shape[0] / 500, color)
        cv2.imshow("silent-face", frame)
        # 下一帧
        ret, frame = cap.read()
        if cv2.waitKey(30) and 0xFF == 'q':
            break
    # Releasing all the resources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    desc = "test"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "--device_id",
        type=int,
        default=-1,
        help="which gpu id, [0/1/2/3]")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./silent_face/resources/anti_spoof_models",
        help="model_lib used to test")
    args = parser.parse_args()
    test_webcam(args.model_dir, args.device_id)
