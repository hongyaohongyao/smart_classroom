import cv2
import dlib
import numpy as np
from PIL import ImageFilter, Image

predictor_68_point_model = "weights/shape_predictor_68_face_landmarks.dat"
face_recognition_model = "weights/dlib_face_recognition_resnet_model_v1.dat"

pose_predictor_68_point = dlib.shape_predictor(predictor_68_point_model)

face_encoder = dlib.face_recognition_model_v1(face_recognition_model)


def _css_to_rect(css):
    """
    Convert a tuple in (top, right, bottom, left) order to a dlib `rect` object

    :param css:  plain tuple representation of the rect in (top, right, bottom, left) order
    :return: a dlib `rect` object
    """
    return dlib.rectangle(css[0], css[1], css[2], css[3])


def _raw_face_landmarks(face_image, face_locations):
    pose_predictor = pose_predictor_68_point
    face_locations = [_css_to_rect(face_location) for face_location in face_locations]

    return [pose_predictor(face_image, face_location) for face_location in face_locations]


def face_landmarks(face_image, face_locations):
    raw_face_landmarks = _raw_face_landmarks(face_image, face_locations)

    return [[[p.x, p.y] for p in landmark.parts()] for landmark in raw_face_landmarks]


def face_encodings(face_image, known_face_locations=None, num_jitters=1):
    """
    Given an image, return the 128-dimension face encoding for each face in the image.

    :param face_image: The image that contains one or more faces
    :param known_face_locations: Optional - the bounding boxes of each face if you already know them.
    :param num_jitters: How many times to re-sample the face when calculating encoding. Higher is more accurate, but slower (i.e. 100 is 100x slower)
    :param model: Optional - which model to use. "large" (default) or "small" which only returns 5 points but is faster.
    :return: A list of 128-dimensional face encodings (one for each face in the image)
    """
    raw_landmarks = _raw_face_landmarks(face_image, known_face_locations)
    return [np.array(face_encoder.compute_face_descriptor(face_image, raw_landmark_set, num_jitters)) for
            raw_landmark_set in raw_landmarks]


def face_distance(x1, x2, metric="norm"):
    """
        Given a list of face encodings, compare them to a known face encoding and get a euclidean distance
        for each comparison face. The distance tells you how similar the faces are.

        :param x1: face_encodings
        :param x2: face_to_compare
        :param metric: metric to compute distance
        :return: A numpy ndarray with the distance for each face in the same order as the 'faces' array
        """
    if len(x1) == 0:
        return np.empty(0)

    if metric == "cos":
        return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))
    else:
        return np.linalg.norm(x1 - x2, axis=1)


def get_scaled_mouth_keypoints(face_keypoints):
    mouth_keypoints = face_keypoints[48:68]
    mouth_x1 = np.min(mouth_keypoints[:, 0])
    mouth_y1 = np.min(mouth_keypoints[:, 1])
    mouth_x2 = np.max(mouth_keypoints[:, 0])
    mouth_y2 = np.max(mouth_keypoints[:, 1])

    mouth_x1_y1 = np.array([mouth_x1, mouth_y1])
    mouth_width = np.array([mouth_x2 - mouth_x1, mouth_y2 - mouth_y1])
    scaled_mouth_keypoints = (mouth_keypoints - mouth_x1_y1) / mouth_width
    return scaled_mouth_keypoints, (mouth_x1, mouth_y1, mouth_x2, mouth_y2)


def mouth_open_degree(scaled_mouth_keypoints):
    """
    计算张嘴程度
    :param scaled_mouth_keypoints: 按嘴部框范围标准化后的关键点坐标
    :return:
    """
    up_mouth_keypoints = scaled_mouth_keypoints[13:16]
    down_mouth_keypoints = scaled_mouth_keypoints[17:20]
    # 计算嘴对应点之间的距离
    mouth_distance = np.linalg.norm(up_mouth_keypoints - down_mouth_keypoints, axis=1)
    return mouth_distance


def get_scaled_face_keypoints(face_keypoints):
    """
    获取标准化后的人脸关键点坐标
    :param face_keypoints: 脸部关键点
    :return: 标准化后的人脸关键点坐标，人脸框的位置
    """
    face_outline_keypoints = face_keypoints[:27]
    face_x1 = np.min(face_outline_keypoints[:, 0])
    face_y1 = np.min(face_outline_keypoints[:, 1])
    face_x2 = np.max(face_outline_keypoints[:, 0])
    face_y2 = np.max(face_outline_keypoints[:, 1])
    # 获取标准化的脸部坐标
    face_x1_y1 = np.array([face_x1, face_y1])
    face_width = np.array([face_x2 - face_x1, face_y2 - face_y1])
    scaled_face_keypoints = (face_keypoints - face_x1_y1) / face_width
    return scaled_face_keypoints, (face_x1, face_y1, face_x2, face_y2)


def get_scaled_eye_keypoints(face_keypoints, witch='left'):
    """
    获取标准化后的人脸关键点坐标
    :param witch: 识别哪只眼睛
    :param face_keypoints: 脸部关键点
    :return: 标准化后的人脸关键点坐标，人脸框的位置
    """
    if witch == 'left':
        eye_keypoints = face_keypoints[36:42]
    else:
        eye_keypoints = face_keypoints[42:48]
    x1 = np.min(eye_keypoints[:, 0])
    y1 = np.min(eye_keypoints[:, 1])
    x2 = np.max(eye_keypoints[:, 0])
    y2 = np.max(eye_keypoints[:, 1])
    # 获取标准化的脸部坐标
    x1_y1 = np.array([x1, y1])
    eye_width = np.array([x2 - x1, y2 - y1])
    scaled_eye_keypoints = (face_keypoints - x1_y1) / eye_width
    return scaled_eye_keypoints, (x1, y1, x2, y2)


def turn_face_degree(scaled_face_keypoints):
    return np.average(scaled_face_keypoints[:, 0] * 2 - 1)


def skin_detect(img):
    skinCrCbHist = np.zeros((256, 256), dtype=np.uint8)
    # 利用opencv自带的椭圆生成函数先生成一个肤色椭圆模型
    cv2.ellipse(skinCrCbHist, (113, 156), (23, 15), 43, 0, 360, (255, 255, 255), -1)
    output_mask = np.zeros(img.shape, dtype=np.uint8)
    ycrcb_image = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)  # 首先转换成到YCrCb空间
    output_mask[skinCrCbHist[ycrcb_image[:, :, 1], ycrcb_image[:, :, 2]] > 0] = 255
    return output_mask


def skin_detect_one_zero_matrix(img):
    skinCrCbHist = np.zeros((256, 256), dtype=np.uint8)
    # 利用opencv的椭圆生成函数先生成一个肤色椭圆模型
    cv2.ellipse(skinCrCbHist, (113, 156), (23, 15), 43, 0, 360, (255, 255, 255), -1)
    output_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    ycrcb_image = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)  # 首先转换成到YCrCb空间
    output_mask[skinCrCbHist[ycrcb_image[:, :, 1], ycrcb_image[:, :, 2]] > 0] = 1
    return output_mask


def close_eye_degree(frame, face_keypoints):
    # 左眼
    _, (x1, y1, x2, y2) = get_scaled_eye_keypoints(face_keypoints, 'left')
    eye_img = frame[y1:y2, x1:x2]
    mask = skin_detect_one_zero_matrix(eye_img)
    mask = mask.reshape(1, -1)
    degree = np.sum(mask) / mask.size
    # 右眼
    _, (x1, y1, x2, y2) = get_scaled_eye_keypoints(face_keypoints, 'right')
    eye_img = frame[y1:y2, x1:x2]
    mask = skin_detect_one_zero_matrix(eye_img)
    mask = mask.reshape(1, -1)
    degree += np.sum(mask) / mask.size
    return degree


def close_eye_degree_v2(frame, face_keypoints):
    # 左眼
    _, (x1, y1, x2, y2) = get_scaled_eye_keypoints(face_keypoints, 'left')
    eye_img = frame[y1:y2, x1:x2]
    mask = skin_detect_one_zero_matrix(eye_img)
    mask = mask.reshape(1, -1)
    degree = np.sum(mask) / mask.size
    # 右眼
    _, (x1, y1, x2, y2) = get_scaled_eye_keypoints(face_keypoints, 'right')
    eye_img = frame[y1:y2, x1:x2]
    mask = skin_detect_one_zero_matrix(eye_img)
    mask = mask.reshape(1, -1)
    degree += np.sum(mask) / mask.size
    return degree


def face_enhance(img):
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    enhance = img.filter(ImageFilter.EDGE_ENHANCE_MORE)  # 大阈值边缘增强
    result = cv2.cvtColor(np.asarray(enhance), cv2.COLOR_RGB2BGR)
    gamma = 0.2
    scale = float(np.iinfo(result.dtype).max - np.iinfo(result.dtype).min)
    result = ((result.astype(np.float32) / scale) ** gamma) * scale  # 自适应gamma增强
    result = np.clip(result, 0, 255).astype(np.uint8)
    return result
