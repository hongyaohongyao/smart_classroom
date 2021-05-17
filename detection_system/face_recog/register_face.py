import json
import os

import cv2
import numpy as np

from models import face_recog
from models.face_boxes_location import FaceBoxesLocation
from utils.face_bank import face_bank_path, get_known_face_encodes

face_source = 'people'
default_new_name = 'face'


def get_newest_default_name_id(known_face_names, default_name=default_new_name):
    if len(known_face_names) == 0:
        return -1
    default_names = filter(lambda x: x.startswith(default_name), known_face_names)
    default_names_id = (x.replace(default_name, '') for x in default_names)
    default_names_id = filter(lambda x: x.isdigit(), default_names_id)
    default_names_id = (int(s) for s in default_names_id)
    return max(*default_names_id)


def add_new_face(name, encoding, face_img, facebank=face_bank_path):
    path = os.path.join(facebank, name)
    os.mkdir(path)
    # 保存面部编码
    with open(os.path.join(path, 'encoding.json'), 'w') as f:
        json.dump(encoding, f)
    # 保存面部图片
    cv2.imwrite(os.path.join(path, 'face.jpg'), face_img)


if __name__ == '__main__':
    fbl = FaceBoxesLocation()
    imgs = os.listdir(face_source)
    imgs = [os.path.join(face_source, img) for img in imgs]
    # 获取已知的人脸
    known_face_names, known_face_encodings = get_known_face_encodes(face_bank_path)
    # 获取新人脸的默认命名
    idx = get_newest_default_name_id(known_face_names, default_new_name) + 1
    for frame in (cv2.imread(fn) for fn in imgs):
        img_shape = frame.shape
        face_locations = fbl.face_location(frame)  # 人脸定位原图
        # 获取脸部编码
        face_encodings = face_recog.face_encodings(frame, face_locations)
        for i, face_encoding in enumerate(face_encodings):
            #  use the known face with the smallest distance to the new face
            face_distances = face_recog.face_distance(known_face_encodings, face_encoding)
            if len(face_distances) != 0:
                best_match_index = np.argmin(face_distances)
                face_distance = face_distances[best_match_index]
            else:
                face_distance = 999
            # 判断 是否是新的人
            if False:
                # 跳过了一个人脸
                name = known_face_names[best_match_index]
            else:
                # 注册新的人脸
                name = default_new_name + str(idx)
                idx += 1
                # 将人脸添加到 facebank
                x1, y1, x2, y2 = face_locations[i].astype(int)
                img_cropped = frame[y1:y2, x1:x2]
                add_new_face(name, face_encoding.tolist(), img_cropped, face_bank_path)
                # 更新 known_face_names和 known_face_encodings
                known_face_names.append(name)
                known_face_encodings.append(face_encoding)
                print(f"添加新的人脸: {name}")
