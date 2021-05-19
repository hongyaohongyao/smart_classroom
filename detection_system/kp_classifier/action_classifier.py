import joblib
import torch

cheating_classifier_path = 'weights/cheating_detector_rfc_kp.pkl'

cheating_classifier = joblib.load(cheating_classifier_path)

action_type = ['学习', '托腮', '撑头', '探头', '转头', '偷看', '传递', '捡东西']
cheating_type = [*range(3, 7)]

kp_without_face = [x for x in range(11)] + [17, 18]
print(len(kp_without_face))


def from_136_kp(keypoints):
    keypoints = keypoints[kp_without_face]
    x_min = torch.min(keypoints[:, 0])
    y_min = torch.min(keypoints[:, 1])
    x_max = torch.max(keypoints[:, 0])
    y_max = torch.max(keypoints[:, 1])
    x1_y1 = torch.tensor([x_min, y_min])
    width = torch.tensor([x_max - x_min, y_max - y_min])
    scaled_keypoints = (keypoints - x1_y1) / width
    return scaled_keypoints.flatten()


def action_classify(keypoints):
    return cheating_classifier.predict(from_136_kp(keypoints).reshape(1, -1))[0]


def cheating_detect(keypoints):
    action = action_classify(keypoints)
    return action in cheating_type
