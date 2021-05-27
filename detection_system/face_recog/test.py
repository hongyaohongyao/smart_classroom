import os
import time
import face_recognition
import cv2

# from face_recog.models.face_boxes_location import FaceBoxesLocation, face_source
#
scale_rate = 1
face_source = './people'

# if __name__ == '__main__':
#     fbl = FaceBoxesLocation(weights='./weights/FaceBoxes.pth')
#     imgs = os.listdir(face_source)
#     imgs = [os.path.join(face_source, img) for img in imgs]
#     for frame in (cv2.imread(fn) for fn in imgs):
#         start_time = time.time()
#         img_shape = frame.shape
#         resize_w = 640
#         resize_rate = resize_w / img_shape[1]
#         resize_h = int(img_shape[0] * resize_rate)
#         resize_frame = cv2.resize(frame, (resize_w, resize_h))
#         # 人脸定位
#         face_locations = fbl.face_location(frame)
#         face_locations[:, :4] *= resize_rate
#
#         for x1, y1, x2, y2 in face_locations:
#             # Draw a box around the face
#             cv2.rectangle(resize_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
#
#             # # Draw a label with a name below the face
#             # cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
#             # font = cv2.FONT_HERSHEY_DUPLEX
#         end_time = time.time()
#         cv2.imshow("show", resize_frame)
#         print(f'用时: {round(end_time - start_time, 2)} s')
#         while cv2.waitKey(-1) and 0xFF == 'p':
#             pass
if __name__ == '__main__':
    imgs = os.listdir('people')
    imgs = [os.path.join('people', img) for img in imgs]
    for frame in (cv2.imread(fn) for fn in imgs):
        start_time = time.time()
        shape = frame.shape
        w = 640
        h = int(shape[0] * w / shape[1])
        frame = cv2.resize(frame, (w, h))
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=1 / scale_rate, fy=1 / scale_rate)
        small_frame = frame

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        face_locations = face_recognition.face_locations(rgb_small_frame, 4)
        for top, right, bottom, left in face_locations:
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= scale_rate
            right *= scale_rate
            bottom *= scale_rate
            left *= scale_rate

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            # cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            # font = cv2.FONT_HERSHEY_DUPLEX
        cv2.imshow("show", frame)
        end_time = time.time()
        print(f'用时: {round(end_time - start_time, 2)} s')
        while cv2.waitKey(-1) and 0xFF == 'p':
            pass
