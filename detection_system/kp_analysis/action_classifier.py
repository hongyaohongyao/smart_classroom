import joblib
import torch

cheating_classifier_path = 'weights/rfc_cheating_with_hip_v1.pkl'

cheating_classifier = joblib.load(cheating_classifier_path)

action_type = ['', 'learn', 'think', 'think2', 'probe', 'turn', 'peep', 'passing', 'pick']
cheating_type = [*range(4, 8)]

kp_without_face = [x for x in range(11)] + [17, 18, 19]
print(len(kp_without_face))


def from_136_kp(keypoints):
    keypoints = keypoints[:, kp_without_face]
    x_min = torch.min(keypoints[:, :, 0], dim=1).values
    y_min = torch.min(keypoints[:, :, 1], dim=1).values
    x_max = torch.max(keypoints[:, :, 0], dim=1).values
    y_max = torch.max(keypoints[:, :, 1], dim=1).values
    x1_y1 = torch.stack([x_min, y_min], dim=1).unsqueeze(1)
    width = torch.stack([x_max - x_min, y_max - y_min], dim=1).unsqueeze(1)
    scaled_keypoints = (keypoints - x1_y1) / width
    return scaled_keypoints.flatten(start_dim=1)


def action_classify(keypoints):
    return cheating_classifier.predict(from_136_kp(keypoints))


def cheating_detect(keypoints):
    action = action_classify(keypoints)
    return action in cheating_type
