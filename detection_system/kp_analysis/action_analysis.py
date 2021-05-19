import torch


def stretch_out_degree(keypoints, left=True, right=True):
    """
    :param keypoints: Halpe 26 keypoints 或 136关键点 [N,keypoints]
    :param left: 是否计算作弊的伸手情况
    :param right: 是否计算右臂的伸手情况
    :return: ([N,left_hand_degree],[N,left_hand_degree]), hand_degree = (arm out?,forearm out?,straight arm?)
    """
    result = []
    if left:
        shoulder_vec = keypoints[:, 18] - keypoints[:, 5]
        arm_vec = keypoints[:, 5] - keypoints[:, 7]
        forearm_vec = keypoints[:, 7] - keypoints[:, 9]
        _results = torch.hstack([torch.cosine_similarity(shoulder_vec, arm_vec).unsqueeze(1),
                                 torch.cosine_similarity(shoulder_vec, forearm_vec).unsqueeze(1),
                                 torch.cosine_similarity(arm_vec, forearm_vec).unsqueeze(1)])
        result.append(_results)
    if right:
        shoulder_vec = keypoints[:, 18] - keypoints[:, 6]
        arm_vec = keypoints[:, 6] - keypoints[:, 8]
        forearm_vec = keypoints[:, 8] - keypoints[:, 10]
        _results = torch.hstack([torch.cosine_similarity(shoulder_vec, arm_vec).unsqueeze(1),
                                 torch.cosine_similarity(shoulder_vec, forearm_vec).unsqueeze(1),
                                 torch.cosine_similarity(arm_vec, forearm_vec).unsqueeze(1)])
        result.append(_results)
    return result


def is_stretch_out(degree, threshold=None, dim=1):
    """

    :param degree: 通过stretch_out_degree计算的手臂余弦相似度数组
    :param threshold: 阈值
    :param dim: 处理的维度
    :return:
    """
    if threshold is None:
        threshold = torch.tensor([0.8, 0.8, 0.5])
    return torch.all(degree > threshold, dim=dim)
