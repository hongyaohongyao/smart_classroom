import os
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from models import face_recog
from models.face_boxes_location import FaceBoxesLocation
from utils.face_bank import get_known_face_encodes

# 微软雅黑
font = ImageFont.truetype("simsun.ttc", 10, encoding="utf-8")

# 图源
face_source = 'people'

if __name__ == '__main__':
    fbl = FaceBoxesLocation()
    imgs = os.listdir(face_source)
    imgs = [os.path.join(face_source, img) for img in imgs]
    # 获取已知的人脸
    known_face_names, known_face_encodings = get_known_face_encodes()
    # 获取新人脸的默认命名
    for frame in (cv2.imread(fn) for fn in imgs):
        start_time = time.time()
        img_shape = frame.shape
        resize_w = 680
        resize_rate = resize_w / img_shape[1]
        resize_h = int(img_shape[0] * resize_rate)
        resize_frame = cv2.resize(frame, (resize_w, resize_h))
        # opencv不支持中文，这里使用PIL作为画板
        resize_frame_pil = Image.fromarray(cv2.cvtColor(resize_frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(resize_frame_pil)  # 创建画板
        # 在原图上定位人脸
        face_locations = fbl.face_location(frame)
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
            if face_distance < 0.45:
                name = known_face_names[best_match_index]
            else:
                name = 'unknown'
            # 将文字转换成utf8
            x1, y1, x2, y2 = (face_locations[i] * resize_rate).astype(int)
            # 把人脸框出来标号
            draw.rectangle([(x1, y1), (x2, y2)], outline=(0, 0, 255), width=2)

            # Draw a label with a name below the face
            draw.rectangle([(x1, y1 - 10), (x2, y1)], (0, 0, 255))
            draw.text((x1, y1 - 10), name, (255, 255, 255), font)
        end_time = time.time()
        resize_frame = cv2.cvtColor(np.array(resize_frame_pil), cv2.COLOR_RGB2BGR)
        cv2.imshow("show", resize_frame)
        print(f'用时: {round(end_time - start_time, 2)} s')
        while cv2.waitKey(-1) and 0xFF == 'p':
            pass
