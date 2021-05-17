import os
import numpy as  np
import json

face_bank_path = 'facebank'


def read_encoding_json2npy(path):
    with open(path) as f:
        return np.array(json.load(f))


def get_known_face_encodes(facebank=face_bank_path):
    known_face_names = os.listdir(facebank)  # 读取已经录入的人名
    known_face_encodings = [read_encoding_json2npy(
        os.path.join(facebank, name, 'encoding.json')
    ) for name in known_face_names]  # 读取已知encodings

    return known_face_names, known_face_encodings
