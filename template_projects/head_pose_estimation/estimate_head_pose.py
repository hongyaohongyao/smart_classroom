"""Demo code shows how to estimate human head pose.
Currently, human face is detected by a detector from an OpenCV DNN module.
Then the face box is modified a little to suits the need of landmark
detection. The facial landmark detection is done by a custom Convolutional
Neural Network trained with TensorFlow. After that, head pose is estimated
by solving a PnP problem.
"""
import threading
from argparse import ArgumentParser
from multiprocessing import Queue

import cv2
import numpy as np

from mark_detector import MarkDetector
from pose_estimator import PoseEstimator
from stabilizer import Stabilizer

print("OpenCV version: {}".format(cv2.__version__))

# multiprocessing may not work on Windows and macOS, check OS for safety.
# detect_os()

CNN_INPUT_SIZE = 128

# Take arguments from user input.
parser = ArgumentParser()
parser.add_argument("--video", type=str, default=None,
                    help="Video file to be processed.")
parser.add_argument("--cam", type=int, default=None,
                    help="The webcam index.")
args = parser.parse_args()


def get_face(detector, img_queue, box_queue):
    """Get face from image queue. This function is used for multiprocessing"""
    while True:
        image = img_queue.get()
        box = detector.extract_cnn_facebox(image)
        box_queue.put(box)


if __name__ == '__main__':
    """MAIN"""
    # Video source from webcam or video file.
    video_src = args.cam if args.cam is not None else args.video
    if video_src is None:
        print("Warning: video source not assigned, default webcam will be used.")
        video_src = 0

    cap = cv2.VideoCapture(video_src)
    if video_src == 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    _, sample_frame = cap.read()

    # Introduce mark_detector to detect landmarks.
    mark_detector = MarkDetector()

    # Setup process and queues for multiprocessing.
    img_queue = Queue()
    box_queue = Queue()
    img_queue.put(sample_frame)
    thread = threading.Thread(target=get_face, args=(mark_detector, img_queue, box_queue))
    thread.daemon = True
    thread.start()

    # Introduce pose estimator to solve pose. Get one frame to setup the
    # estimator according to the image size.
    height, width = sample_frame.shape[:2]
    pose_estimator = PoseEstimator(img_size=(height, width))

    # Introduce scalar stabilizers for pose.
    pose_stabilizers = [Stabilizer(
        state_num=2,
        measure_num=1,
        cov_process=0.1,
        cov_measure=0.1) for _ in range(6)]

    tm = cv2.TickMeter()

    while True:
        # Read frame, crop it, flip it, suits your needs.
        frame_got, frame = cap.read()
        if frame_got is False:
            break

        # Crop it if frame is larger than expected.
        # frame = frame[0:480, 300:940]

        # If frame comes from webcam, flip it so it looks like a mirror.
        # if video_src == 0:
        #     frame = cv2.flip(frame, 2)

        # Pose estimation by 3 steps:
        # 1. detect face;
        # 2. detect landmarks;
        # 3. estimate pose

        # Feed frame to image queue.
        img_queue.put(frame)

        # Get face from box queue.
        facebox = box_queue.get()

        if facebox is not None:
            # Detect landmarks from image of 128x128.
            face_img = frame[facebox[1]: facebox[3],
                       facebox[0]: facebox[2]]
            face_img = cv2.resize(face_img, (CNN_INPUT_SIZE, CNN_INPUT_SIZE))
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

            tm.start()
            marks = mark_detector.detect_marks(face_img)
            tm.stop()

            # Convert the marks locations from local CNN to global image.
            marks *= (facebox[2] - facebox[0])
            marks[:, 0] += facebox[0]
            marks[:, 1] += facebox[1]

            # Uncomment following line to show raw marks.
            # mark_detector.draw_marks(frame, marks, color=(0, 255, 0))

            # Uncomment following line to show facebox.
            # mark_detector.draw_box(frame, [facebox])

            # Try pose estimation with 68 points.
            marks = np.array([[218.4677, 89.87073],
                              [220.22453, 121.96762],
                              [223.96681, 152.65334],
                              [230.02744, 182.37177],
                              [239.01709, 211.59616],
                              [257.98718, 239.22322],
                              [279.6545, 258.51376],
                              [311.12903, 277.57117],
                              [350.71585, 286.94177],
                              [379.4071, 279.94937],
                              [405.33643, 262.30646],
                              [424.69977, 240.23265],
                              [439.89352, 214.46347],
                              [447.55374, 183.39143],
                              [454.3741, 152.91576],
                              [461.261, 119.00888],
                              [467.35553, 87.80364],
                              [259.43222, 57.652813],
                              [275.41937, 49.985016],
                              [294.87283, 48.28077],
                              [312.70032, 51.04401],
                              [330.32123, 56.870705],
                              [391.8867, 60.823048],
                              [408.12036, 56.251102],
                              [424.36072, 53.19054],
                              [440.7141, 55.756718],
                              [453.48267, 64.0959, ],
                              [360.50037, 94.4313, ],
                              [361.2175, 119.300705],
                              [362.04, 140.86725],
                              [362.67114, 160.99261],
                              [334.13586, 174.79489],
                              [346.0203, 177.28938],
                              [358.7469, 180.27216],
                              [369.87564, 177.55322],
                              [379.49637, 174.3986, ],
                              [277.042, 87.26237],
                              [292.437, 78.486275],
                              [309.88132, 80.89449],
                              [324.76895, 91.85259],
                              [310.34476, 96.826385],
                              [292.4507, 94.96313],
                              [390.42746, 92.99907],
                              [406.47168, 83.25359],
                              [423.17642, 82.80899],
                              [435.07288, 92.21508],
                              [421.45468, 98.3501, ],
                              [405.01886, 98.83813],
                              [310.4157, 218.5501, ],
                              [327.8565, 211.95906],
                              [347.26505, 207.37141],
                              [356.66602, 209.59866],
                              [366.31415, 207.59067],
                              [380.94736, 211.48885],
                              [392.5682, 217.67201],
                              [380.5252, 229.08566],
                              [368.8378, 235.0082, ],
                              [355.32568, 236.70053],
                              [340.88968, 235.58282],
                              [326.3202, 229.22781],
                              [314.68103, 218.55443],
                              [343.84302, 217.35928],
                              [355.64767, 218.21783],
                              [367.20605, 217.28725],
                              [388.59412, 218.09315],
                              [367.44897, 221.60754],
                              [355.48022, 222.92123],
                              [342.7749, 221.94604]])
            pose = pose_estimator.solve_pose_by_68_points(marks)

            # Stabilize the pose.
            steady_pose = []
            pose_np = np.array(pose).flatten()
            for value, ps_stb in zip(pose_np, pose_stabilizers):
                ps_stb.update([value])
                steady_pose.append(ps_stb.state[0])
            steady_pose = np.reshape(steady_pose, (-1, 3))

            # Uncomment following line to draw pose annotation on frame.
            # pose_estimator.draw_annotation_box(
            #     frame, pose[0], pose[1], color=(255, 128, 128))

            # Uncomment following line to draw stabile pose annotation on frame.
            pose_estimator.draw_annotation_box(
                frame, steady_pose[0], steady_pose[1], color=(128, 255, 128))

            # Uncomment following line to draw head axes on frame.
            # pose_estimator.draw_axes(frame, steady_pose[0], steady_pose[1])

        # Show preview.
        cv2.imshow("Preview", frame)
        if cv2.waitKey(10) == 27:
            break

    # Clean up the multiprocessing process.
