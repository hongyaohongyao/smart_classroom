import json
import os
import random
import tkinter
from tkinter import messagebox

import cv2
import numpy as np
from PIL import ImageDraw, Image, ImageFont

from models import face_recog
from models.face_boxes_location import FaceBoxesLocation
from utils.face_bank import face_bank_path, get_known_face_encodes

face_source = 'people'
default_new_name = 'face'
font = ImageFont.truetype("simsun.ttc", 20, encoding="utf-8")
# 系统参数
encoding_per_second = 5
accept_step = 1
# 识别指标
close_eye_gate = 0.2  # 闭眼的门限值
front_face_gate = 0.15  # 非正脸的门限值
open_mouth_gate = 0.35  # 张嘴的门限值
turn_face_gate = 0.20  # 转头的门限值
cum_rate = 0.46  # cum_value = cum_value*(1-cum_rate)+cum_rate
# face_encoding = (1-encoding_update_rate)*face_encoding+encoding_update_rate*new_face_encoding
encoding_update_rate = 0.15
accept_gate = 0.95  # 累计值高过该门限值表示通过
save_face_gate = 0.1  # 录入的人脸的平均距离小于该值时保存人脸


def get_newest_default_name_id(known_face_names, default_name=default_new_name):
    if len(known_face_names) == 0:
        return -1
    default_names = [*filter(lambda x: x.startswith(default_name), known_face_names)]
    if len(default_names) == 0:
        return -1
    default_names_id = (x.replace(default_name, '') for x in default_names)
    default_names_id = filter(lambda x: x.isdigit(), default_names_id)
    default_names_id = (int(s) for s in default_names_id)
    return max(*default_names_id)


def add_new_face(encoding, face_img, know_face_name, facebank=face_bank_path):
    root = tkinter.Tk()

    tkinter.Label(root, text='会员名称:').grid(row=0, column=0)

    v1 = tkinter.StringVar()
    e1 = tkinter.Entry(root, textvariable=v1)  # Entry 是 Tkinter 用来接收字符串等输入的控件.
    e1.grid(row=0, column=1, padx=10, pady=5)  # 设置输入框显示的位置，以及长和宽属性

    def save():
        name = e1.get()
        # 保存=======
        if name in know_face_name:
            messagebox.showinfo('该名称已经存在')
        else:
            path = os.path.join(facebank, name)
            os.mkdir(path)
            # 保存面部编码
            with open(os.path.join(path, 'encoding.json'), 'w') as f:
                json.dump(encoding, f)
            # 保存面部图片
            cv2.imencode('.jpg', face_img)[1].tofile(os.path.join(path, 'face.jpg'))
            know_face_name.append(name)
            root.destroy()

    tkinter.Button(root, text='确认', width=10, command=save) \
        .grid(row=2, column=0, sticky='W', padx=10, pady=5)

    tkinter.Button(root, text='取消', width=10, command=root.destroy) \
        .grid(row=2, column=1, sticky='E', padx=10, pady=5)

    tkinter.mainloop()


step_type = ['眨眼', '张嘴', '左转头', '右转头']


def generate_new_step_state():
    return random.sample(range(len(step_type)), 1)[0]


if __name__ == '__main__':
    fbl = FaceBoxesLocation()
    imgs = os.listdir(face_source)
    imgs = [os.path.join(face_source, img) for img in imgs]
    # 获取已知的人脸
    known_face_names, known_face_encodings = get_known_face_encodes(face_bank_path)
    # 获取新人脸的默认命名
    idx = get_newest_default_name_id(known_face_names, default_new_name) + 1

    cap = cv2.VideoCapture(0)
    encoding_interval = cap.get(cv2.CAP_PROP_FPS) // encoding_per_second  # 计算encoding的间隔
    encoding_counter = 0  # encoding时间间隔的计数器
    if cap.isOpened():
        ret, frame = cap.read()
    face_encoding = None
    step_state = generate_new_step_state()
    current_step = 0
    cum_value = 0
    while ret:
        # 摄像头反转
        frame = cv2.flip(frame, 1)
        orig_frame = frame
        # frame = face_recog.face_enhance(frame)
        # 人脸检测
        face_locations = fbl.face_location(frame).astype(int)
        face_landmarks = None
        if len(face_locations) == 0:
            tips_text = '没有人脸'
            face_encoding = None
            step_state = generate_new_step_state()
            current_step = 0
            cum_value = 0
        elif len(face_locations) > 1:
            tips_text = '请一个人站在机器前注册'
            face_encoding = None
            step_state = generate_new_step_state()
            current_step = 0
            cum_value = 0
        else:
            # 处理人脸录入逻辑
            face_landmarks = face_recog.face_landmarks(frame, face_locations)
            face_keypoints = np.array(face_landmarks)
            # 转头检测
            scaled_face_keypoints, _ = face_recog.get_scaled_face_keypoints(face_keypoints[0])
            turn_face_degree = face_recog.turn_face_degree(scaled_face_keypoints)
            # 是否是正脸 防止嘴部检测失效 或者录入人脸
            if abs(turn_face_degree) > front_face_gate:
                not_front_face = 1
            else:
                not_front_face = 0
            # 提示文字
            tips_text = f'Step.{current_step + 1}: 请 {step_type[step_state]}'
            cum_value = cum_value * (1 - cum_rate)
            # 注册条件达成判断========================
            if current_step >= accept_step:
                tips_text = f'请正视摄像头完成录入 '
                if encoding_counter % encoding_interval != 0:
                    pass
                elif not not_front_face:
                    new_face_encoding = face_recog.face_encodings(frame, face_locations)[0]
                    if face_encoding is None:
                        face_encoding = new_face_encoding
                    else:
                        face_distance = face_recog.face_distance(face_encoding[np.newaxis], new_face_encoding)[0]
                        if face_distance < save_face_gate:
                            x1, y1, x2, y2 in face_locations[0]
                            add_new_face(face_encoding.tolist(),
                                         frame[y1:y2, x1:x2],
                                         known_face_names,
                                         face_bank_path)
                            face_encoding = None
                            step_state = generate_new_step_state()
                            current_step = 0
                            cum_value = 0
                        else:
                            face_encoding *= 1 - encoding_update_rate
                            face_encoding += encoding_update_rate * new_face_encoding
                            cum_value = min(max(1 - abs(save_face_gate - face_distance), 0), 1)
                else:
                    tips_text += '(请正视摄像头)'
                encoding_counter += 1
            # 注册流程
            elif step_state == 0:
                # 闭眼识别
                close_eye_degree = face_recog.close_eye_degree(frame, face_keypoints[0])
                if close_eye_degree > close_eye_gate:
                    cum_value += cum_rate
            elif step_state == 1:  # 嘴型检测
                scaled_mouth_keypoints, _ = face_recog.get_scaled_mouth_keypoints(face_keypoints[0])
                open_mouth_degree = face_recog.mouth_open_degree(scaled_mouth_keypoints)

                if all(open_mouth_degree > open_mouth_gate) and not not_front_face:
                    cum_value += cum_rate
                elif not_front_face:
                    tips_text += f'(请使用正脸)'
            elif step_state == 2:
                # 左转头
                if turn_face_degree < -turn_face_gate:
                    cum_value += cum_rate
            elif step_state == 3:
                # 右转头
                if turn_face_degree > turn_face_gate:
                    cum_value += cum_rate
            tips_text += f'完成率 {round(cum_value * 100, 2)} %'

            if cum_value >= accept_gate and current_step < accept_step:
                step_state = generate_new_step_state()
                current_step += 1
                cum_value = 0
                pass_tips_imgs = np.full_like(frame, 255)
                pass_tips_imgs = cv2.putText(pass_tips_imgs,
                                             'OK', (100, 100),
                                             cv2.FONT_HERSHEY_TRIPLEX,
                                             10, (0, 0, 255), 50,
                                             bottomLeftOrigin=True)
                cv2.imshow("register_face", pass_tips_imgs)
                cv2.waitKey(1000)

        # 视频显示
        # opencv不支持中文，这里使用PIL作为画板
        frame_pil = Image.fromarray(cv2.cvtColor(orig_frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(frame_pil)  # 创建画板
        for x1, y1, x2, y2 in face_locations:
            # 把人脸框出来标号
            draw.rectangle([(x1, y1), (x2, y2)], outline=(255, 0, 0), width=2)
            # 绘制面部关键点
            if face_landmarks is not None and len(face_landmarks) == 1:
                for x, y in face_landmarks[0]:
                    draw.point((x, y), (255, 255, 255))
        # 绘制文字提示
        draw.rectangle([(0, 0), (500, 20)], (255, 0, 0))
        draw.text((0, 0), tips_text, (255, 255, 255), font)
        # 显示图片
        frame_show = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
        cv2.imshow("register_face", frame_show)
        # 下一帧
        ret, frame = cap.read()
        if cv2.waitKey(30) and 0xFF == 'q':
            break
    cap.release()
