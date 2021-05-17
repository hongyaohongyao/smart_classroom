import os
import time

import cv2
import face_recognition
import numpy as np


class FaceLocationScanner:
    """
    扫描图片进行hog人脸检测，对于尺寸较大的图片非常慢，但能够进行远距离识别
    """

    @staticmethod
    def scan(img, scan_win=(200, 200), win_step=(150, 150), number_of_times_to_upsample=3):
        img_shape = img.shape
        # resize_w = 640
        # resize_h = int(img_shape[0] * resize_w / img_shape[1])
        # resize_img = cv2.resize(img, (resize_w, resize_h))
        face_locations = []
        for x, y in ((x, y)
                     for x in range(0, img_shape[1], win_step[0])
                     for y in range(0, img_shape[0], win_step[1])):
            # 剪切窗口内的图像
            y1 = min(y + scan_win[1], img_shape[0])
            x1 = min(x + scan_win[0], img_shape[1])
            win_img = img[y:y1, x:x1]
            local_face_locations = FaceLocationScanner._scan_one_frame(win_img, 1)
            # 修正窗口位置
            if len(local_face_locations) != 0:
                local_face_locations = np.array(local_face_locations)
                local_face_locations[:, 0::2] += y
                local_face_locations[:, 1::2] += x
                face_locations.extend(local_face_locations)
        return face_locations

    @staticmethod
    def skin_detect(img):
        skinCrCbHist = np.zeros((256, 256), dtype=np.uint8)
        # 利用opencv自带的椭圆生成函数先生成一个肤色椭圆模型
        cv2.ellipse(skinCrCbHist, (113, 156), (23, 15), 43, 0, 360, (255, 255, 255), -1)
        output_mask = np.zeros(img.shape, dtype=np.uint8)
        ycrcb_image = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)  # 首先转换成到YCrCb空间
        output_mask[skinCrCbHist[ycrcb_image[:, :, 1], ycrcb_image[:, :, 2]] > 0] = 255
        return output_mask

    @staticmethod
    def _scan_one_frame(frame, scale_times=1, number_of_times_to_upsample=3):
        # Resize frame of video to 1/scale_times size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=1 / scale_times, fy=1 / scale_times)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        face_locations = face_recognition.face_locations(rgb_small_frame, number_of_times_to_upsample)
        # return [[top * scale_times,
        #          right * scale_times,
        #          bottom, scale_times,
        #          left * scale_times]
        #         for top, right, bottom, left in face_locations]
        return face_locations * scale_times


face_source = '../people'
if __name__ == '__main__':
    imgs = os.listdir(face_source)
    imgs = [os.path.join(face_source, img) for img in imgs]
    for frame in (cv2.imread(fn) for fn in imgs):
        start_time = time.time()
        img_shape = frame.shape
        resize_w = 640
        resize_rate = resize_w / img_shape[1]
        resize_h = int(img_shape[0] * resize_rate)
        resize_frame = cv2.resize(frame, (resize_w, resize_h))
        # 扫描原图
        face_locations = FaceLocationScanner.scan(frame)
        if len(face_locations) != 0:
            face_locations = np.stack(face_locations) * resize_rate
            face_locations = face_locations.astype(int)
        for top, right, bottom, left in face_locations:
            # Draw a box around the face
            cv2.rectangle(resize_frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # # Draw a label with a name below the face
            # cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            # font = cv2.FONT_HERSHEY_DUPLEX
        end_time = time.time()
        cv2.imshow("show", resize_frame)
        print(f'用时: {round(end_time - start_time, 2)} s')
        while cv2.waitKey(-1) and 0xFF == 'p':
            pass
