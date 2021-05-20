import cv2
import numpy as np
import torch
from PIL import Image

color2index = {
    (0, 0, 0): 0,  # 过道区域
    (255, 255, 255): 1,  # 座位区域
    (255, 0, 0): 2  # 天花板区域
}

color_map = np.array([
    [0, 0, 0],
    [255, 255, 255],
    [255, 0, 0]
], dtype=np.uint8)


class SceneMasker:
    def __init__(self, scene_mask):
        img = cv2.imread(scene_mask)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.img = img
        scene_mask = [[color2index[tuple(img[y, x])]
                       for x in range(img.shape[1])]
                      for y in range(img.shape[0])]
        self.scene_mask = torch.tensor(scene_mask)
        self.scene_mask_img = Image.fromarray(np.uint8(img), mode='RGB')

        self.scene_mask

    def is_in_seat(self, boxes):
        scene_mask = self.scene_mask
        boxes = boxes.astype(int)
        head_points_center_x = (boxes[:, 0] + boxes[:, 2]) // 2
        head_points_center_y = (boxes[:, 1] + boxes[:, 3]) // 2
        head_points_top_y = boxes[:, 1]
        head_on_ceiling = np.array([scene_mask[
                                        head_points_top_y[i],
                                        head_points_center_x[i]
                                    ] == 2 for i in range(boxes.shape[0])])
        body_in_seat = np.array([scene_mask[
                                     head_points_center_y[i],
                                     head_points_center_x[i]
                                 ] == 1 for i in range(boxes.shape[0])])
        return body_in_seat & ~head_on_ceiling

    def mask_on_img(self, img):
        if isinstance(img, Image.Image):
            return Image.blend(img, self.scene_mask_img, 0.5)
        elif isinstance(img, torch.Tensor):
            img = Image.fromarray(img)
            img = Image.blend(img, self.scene_mask_img, 0.5)
            return torch.tensor(img)
        elif isinstance(img, np.ndarray):
            img = Image.fromarray(img)
            img = Image.blend(img, self.scene_mask_img, 0.5)
            return np.array(img)
        else:
            raise Exception("不是合适的类型")
