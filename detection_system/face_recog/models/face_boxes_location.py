import os
import time

import cv2
import numpy as np
import torch

try:
    from layers.functions.prior_box import PriorBox
    from models.faceboxes import FaceBoxes
    from utils.box_utils import decode
    from utils.nms.py_cpu_nms import py_cpu_nms as nms
except ImportError:
    from face_recog.layers.functions.prior_box import PriorBox
    from face_recog.models.faceboxes import FaceBoxes
    from face_recog.utils.box_utils import decode
    from face_recog.utils.nms.py_cpu_nms import py_cpu_nms as nms

cfg = {
    'name': 'FaceBoxes',
    # 'min_dim': 1024,
    # 'feature_maps': [[32, 32], [16, 16], [8, 8]],
    # 'aspect_ratios': [[1], [1], [1]],
    'min_sizes': [[32, 64, 128], [256], [512]],
    'steps': [32, 64, 128],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True
}


class FaceBoxesLocation:
    def __init__(self, weights='weights/FaceBoxes.pth', **opt):
        net = FaceBoxes(phase='test', size=None, num_classes=2)  # initialize detector
        net = load_model(net, opt.get("weights", weights), True)
        net.eval()
        self.net = net
        self.top_k = opt.get('top_k', 5000)
        self.confidence_threshold = opt.get('confidence_threshold', 0.05)
        self.nms_threshold = opt.get('nms_threshold', 0.3)
        self.keep_top_k = opt.get('keep_top_k', 750)

    def face_location(self, img, resize=1, cof=0.5):
        # 处理图片
        img = np.float32(img)
        if resize != 1:
            img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
        im_height, im_width, _ = img.shape
        priorbox = PriorBox(cfg, image_size=(im_height, im_width))
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        # 前向传播
        loc, conf = self.net(img)  # forward pass
        #
        priors = priorbox.forward()
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        # ignore low scores
        inds = np.where(scores > self.confidence_threshold)[0]
        boxes = boxes[inds]
        scores = scores[inds]
        # keep top-K before NMS
        order = scores.argsort()[::-1][:self.top_k]
        boxes = boxes[order]
        scores = scores[order]
        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        # keep = py_cpu_nms(dets, args.nms_threshold)
        keep = nms(dets, self.nms_threshold)
        dets = dets[keep, :]

        # keep top-K faster NMS
        dets = dets[:self.keep_top_k, :]
        # 筛选出置信度较高的人脸
        dets = dets[dets[:, 4] > cof, :4]
        return dets


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


pretrained_path = '../weights/FaceBoxes.pth'
face_source = '../people'

if __name__ == '__main__':
    fbl = FaceBoxesLocation(weights=pretrained_path)
    imgs = os.listdir(face_source)
    imgs = [os.path.join(face_source, img) for img in imgs]
    for frame in (cv2.imread(fn) for fn in imgs):
        start_time = time.time()
        img_shape = frame.shape
        resize_w = 640
        resize_rate = resize_w / img_shape[1]
        resize_h = int(img_shape[0] * resize_rate)
        resize_frame = cv2.resize(frame, (resize_w, resize_h))
        # 人脸定位
        face_locations = fbl.face_location(frame)
        face_locations[:, :4] *= resize_rate

        for x1, y1, x2, y2 in face_locations:
            # Draw a box around the face
            cv2.rectangle(resize_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # # Draw a label with a name below the face
            # cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            # font = cv2.FONT_HERSHEY_DUPLEX
        end_time = time.time()
        cv2.imshow("show", resize_frame)
        print(f'用时: {round(end_time - start_time, 2)} s')
        while cv2.waitKey(-1) and 0xFF == 'p':
            pass
