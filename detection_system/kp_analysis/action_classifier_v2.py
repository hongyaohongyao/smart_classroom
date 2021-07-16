import torch

from kp_analysis.logistic_regression import LogisticRegression

# classroom_action_classifier_path = './weights/classroom_action_lr_front_v1.pth'
# classroom_action_side_classifier_path = './weights/classroom_action_lr_side_v1.pth'

# classroom_action_classifier = LogisticRegression(28, 16)
# classroom_action_classifier.load_state_dict(torch.load(classroom_action_classifier_path))
# classroom_action_classifier.cpu().eval()

# classroom_action_side_classifier = LogisticRegression(28, 16)
# classroom_action_side_classifier.load_state_dict(torch.load(classroom_action_side_classifier_path))
# classroom_action_side_classifier.cpu().eval()

classroom_action_classifier_path = './weights/classroom_action_lr_front_v2.pth'
classroom_action_classifier = LogisticRegression(28, 19)
classroom_action_classifier.load_state_dict(torch.load(classroom_action_classifier_path))
classroom_action_classifier.cpu().eval()

cheating_type = [*range(4, 8)]

kp_without_face = [x for x in range(11)] + [17, 18, 19]

# v1
# action_type = ['seat', 'write', 'stretch', 'hand_up_R', 'hand_up_L',
#                'hand_up_highly_R', 'hand_up_highly_L',
#                'relax', 'pass_R', 'pass_L',
#                'turn_round_R', 'turn_round_L', 'turn_head_R', 'turn_head_L',
#                'lower_head', 'sleep']

# 0 正常坐姿不动
# 1 正常低头写字
# 2 正常伸懒腰
# 3 举右手低
# 4 举左手低
# 5 举右手高
# 6 举左手高
# 7 起立
# 8 右伸手
# 9 左伸手
# 10 右回头
# 11 左回头
# 12 右扭头
# 13 左扭头
# 14 严重低头
# 15 上课睡觉

# v2
action_type = ['seat', 'write', 'stretch', 'hand_up_R', 'hand_up_L',
               'hand_up_highly_R', 'hand_up_highly_L',
               'relax', 'hand_up', 'pass_R', 'pass_L', 'pass2_R', 'pass2_L',
               'turn_round_R', 'turn_round_L', 'turn_head_R', 'turn_head_L',
               'sleep', 'lower_head']


# 0 正常坐姿不动
# 1 正常低头写字
# 2 正常伸懒腰
# 3 举右手低
# 4 举左手低
# 5 举右手高
# 6 举左手高
# 7 起立
# 8 抬手
# 9 右伸手
# 10 左伸手
# 11 右伸手2
# 12 左伸手2
# 13 右转身
# 14 左转身
# 15 右转头
# 16 左转头
# 17 上课睡觉
# 18 严重低头


def from_136_kp(keypoints):
    keypoints = keypoints[:, kp_without_face]
    x_min = torch.min(keypoints[:, :, 0], dim=1).values
    y_min = torch.min(keypoints[:, :, 1], dim=1).values
    x_max = torch.max(keypoints[:, :, 0], dim=1).values
    y_max = torch.max(keypoints[:, :, 1], dim=1).values
    x1_y1 = torch.stack([x_min, y_min], dim=1).unsqueeze(1)
    width = torch.stack([x_max - x_min, y_max - y_min], dim=1).unsqueeze(1)
    scaled_keypoints = (keypoints - x1_y1) / width
    scaled_keypoints = (scaled_keypoints - 0.5) / 0.5
    return scaled_keypoints.flatten(start_dim=1)


@torch.no_grad()
def action_classify(keypoints):
    result = torch.argmax(classroom_action_classifier(from_136_kp(keypoints)), dim=1)
    return result


def action_name(label):
    return action_type[label]


def cheating_detect(keypoints):
    action = action_classify(keypoints)
    return action in cheating_type
