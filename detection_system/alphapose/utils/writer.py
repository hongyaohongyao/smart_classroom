import os
import time
from queue import Queue
from threading import Thread

import cv2
import numpy as np
import torch
import torch.multiprocessing as mp

from alphapose.utils.pPose_nms import pose_nms, write_json
from alphapose.utils.transforms import get_func_heatmap_to_coord
from headpose.pose_estimator import PoseEstimator
from headpose.stabilizer import Stabilizer

DEFAULT_VIDEO_SAVE_OPT = {
    'savepath': 'examples/res/1.mp4',
    'fourcc': cv2.VideoWriter_fourcc(*'mp4v'),
    'fps': 25,
    'frameSize': (640, 480)
}

EVAL_JOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

reid_loss_interval = 5  # 单位秒
face_hide_refresh_interval = 1.5  # 单位秒
face_hide_lambda = 0.75


class DataDealer:
    def __init__(self, cfg, opt, save_video=False,
                 video_save_opt=DEFAULT_VIDEO_SAVE_OPT,
                 queueSize=1024):
        self.cfg = cfg
        self.opt = opt
        self.video_save_opt = video_save_opt
        self.head_pose = opt.head_pose  # 是否开启头部姿态相关内容

        self.eval_joints = EVAL_JOINTS
        self.save_video = save_video
        self.heatmap_to_coord = get_func_heatmap_to_coord(cfg)
        # initialize the queue used to store frames read from
        # the video file
        if opt.sp:
            self.result_queue = Queue(maxsize=queueSize)
        else:
            self.result_queue = mp.Queue(maxsize=queueSize)

        if opt.save_img:
            if not os.path.exists(opt.outputpath + '/vis'):
                os.mkdir(opt.outputpath + '/vis')

        if opt.pose_flow:
            from trackers.PoseFlow.poseflow_infer import PoseFlowWrapper
            self.pose_flow_wrapper = PoseFlowWrapper(save_path=os.path.join(opt.outputpath, 'poseflow'))
        if opt.tracking:  # 实例状态
            self.reid_states = {}
            self.reid_global_states = {'frame': 0, "interval": 1, "time": time.time()}

    def start_worker(self, target):
        if self.opt.sp:
            p = Thread(target=target, args=())
        else:
            p = mp.Process(target=target, args=())
        # p.daemon = True
        p.start()
        return p

    def start(self):
        # start a thread to read pose estimation results per frame
        self.result_worker = self.start_worker(self.update)
        return self

    def update(self):
        final_result = []
        norm_type = self.cfg.LOSS.get('NORM_TYPE', None)
        hm_size = self.cfg.DATA_PRESET.HEATMAP_SIZE
        if self.save_video:
            # initialize the file video stream, adapt ouput video resolution to original video
            stream = cv2.VideoWriter(*[self.video_save_opt[k] for k in ['savepath', 'fourcc', 'fps', 'frameSize']])
            if not stream.isOpened():
                print("Try to use other video encoders...")
                ext = self.video_save_opt['savepath'].split('.')[-1]
                fourcc, _ext = self.recognize_video_ext(ext)
                self.video_save_opt['fourcc'] = fourcc
                self.video_save_opt['savepath'] = self.video_save_opt['savepath'][:-4] + _ext
                stream = cv2.VideoWriter(*[self.video_save_opt[k] for k in ['savepath', 'fourcc', 'fps', 'frameSize']])
            assert stream.isOpened(), 'Cannot open video for writing'
        # ======头部姿态估计准备=========
        if self.head_pose:
            # 进行头部姿态估计
            pose_estimator = PoseEstimator(img_size=self.opt.img_size)
            # Introduce scalar stabilizers for pose.
            pose_stabilizers = [Stabilizer(
                state_num=2,
                measure_num=1,
                cov_process=0.1,
                cov_measure=0.1) for _ in range(6)]
            masks_list = []  # 头部关键点列表
            emoji_available_list = []  # 需要进行表情识别的目标的索引
            face_naked_rate = []  # 所有人的脸部露出率
        # keep looping infinitelyd
        while True:
            if self.opt.tracking:  # 处理重识别状态
                reid_states = self.reid_states
                reid_global_states = self.reid_global_states
                reid_global_states["frame"] = (reid_global_states["frame"] + 1) % 9999
                current_time = time.time()
                reid_global_states["interval"] = current_time - reid_global_states['time']
                reid_global_states['time'] = current_time
            # ensure the queue is not empty and get item
            (boxes, scores, ids, hm_data, cropped_boxes, orig_img, im_name) = self.wait_and_get(self.result_queue)
            if orig_img is None:
                # if the thread indicator variable is set (img is None), stop the thread
                if self.save_video:
                    stream.release()
                write_json(final_result, self.opt.outputpath, form=self.opt.format, for_eval=self.opt.eval)
                print("Results have been written to json.")
                return
            # ==========================进一步处理=================================
            # image channel RGB->BGR
            orig_img = np.array(orig_img, dtype=np.uint8)[:, :, ::-1]
            if boxes is None or len(boxes) == 0:
                if self.opt.save_img or self.save_video or self.opt.vis:
                    self.write_image(orig_img, im_name, stream=stream if self.save_video else None)
            else:
                # location prediction (n, kp, 2) | score prediction (n, kp, 1)
                assert hm_data.dim() == 4
                # pred = hm_data.cpu().data.numpy()

                if hm_data.size()[1] == 136:
                    self.eval_joints = [*range(0, 136)]
                elif hm_data.size()[1] == 26:
                    self.eval_joints = [*range(0, 26)]
                pose_coords = []
                pose_scores = []
                for i in range(hm_data.shape[0]):
                    bbox = cropped_boxes[i].tolist()
                    pose_coord, pose_score = self.heatmap_to_coord(hm_data[i][self.eval_joints], bbox, hm_shape=hm_size,
                                                                   norm_type=norm_type)
                    pose_coords.append(torch.from_numpy(pose_coord).unsqueeze(0))
                    pose_scores.append(torch.from_numpy(pose_score).unsqueeze(0))
                preds_img = torch.cat(pose_coords)
                preds_scores = torch.cat(pose_scores)
                if not self.opt.pose_track:
                    boxes, scores, ids, preds_img, preds_scores, pick_ids = \
                        pose_nms(boxes, scores, ids, preds_img, preds_scores, self.opt.min_box_area)
                    if len(preds_img) != 0:
                        preds_img = torch.stack(preds_img)
                # print(boxes[0], cropped_boxes[0],hm_data[0].shape)
                # =========================目标检测对象处理===========================
                if len(preds_img) != 0:
                    if self.head_pose:
                        masks_list.clear()
                        emoji_available_list.clear()
                    for i in range(preds_img.shape[0]):
                        if self.opt.tracking:
                            self_state = self.get_reid_state(ids[i], reid_states, reid_global_states)
                            self_state['index'] = i

                        # ===头部姿态估计相关======
                        if self.head_pose:
                            # 取出脸部关键点
                            face_keypoints = preds_img[i, 26:94]
                            face_keypoints_scores = preds_scores[i, 26:94]
                            # 获取标准化后的人脸关键点坐标
                            # scale_face_keypoints, _ = self.get_scaled_face_keypoints(face_keypoints)
                            # =====脸部露出判定======
                            face_naked = torch.sum(face_keypoints_scores[27:48] > 0.01) / 21  # 这部分暂时不包括嘴部数据
                            mouth_naked = torch.sum(face_keypoints_scores[48:68] > 0.1) / 20  # 这部分是嘴部的裸露程度
                            if face_naked > 0.5 or mouth_naked > 0.5:
                                # 判断是否能够识别表情
                                emoji_available_list.append(i)

                            # ====指标====脸部遮挡检测=======
                            if self_state is not None:
                                self.face_hide(self_state, reid_global_states, face_naked)
                            # ====进行头部姿态估计=====
                            self.estimate_head_pose(pose_estimator, face_keypoints, masks_list)
                            # ==口型识别== 打哈欠和说话
                            if mouth_naked > 0.5 and False:
                                scaled_mouth_keypoints, _ = self.get_scaled_mouth_keypoints(face_keypoints)
                                mouth_distance = self.mouth_open_degree(scaled_mouth_keypoints)
                                if mouth_distance[1] > 0.3:
                                    open_mouth = "open mouth!!!!"
                                elif mouth_distance[1] > 0.2:
                                    open_mouth = "open"
                                else:
                                    open_mouth = ""
                                print(mouth_distance, open_mouth)

                    # =====开始表情识别=====
                # =========================目标检测对象处理完成=========================
                # =========================整理数据========================
                _result = []
                for k in range(len(scores)):
                    _result.append(
                        {
                            'keypoints': preds_img[k],
                            'kp_score': preds_scores[k],
                            'proposal_score': torch.mean(preds_scores[k]) + scores[k] + 1.25 * max(preds_scores[k]),
                            'idx': ids[k],
                            'box': [boxes[k][0], boxes[k][1], boxes[k][2] - boxes[k][0], boxes[k][3] - boxes[k][1]]
                        }
                    )

                result = {
                    'imgname': im_name,
                    'result': _result
                }

                if self.opt.pose_flow:
                    poseflow_result = self.pose_flow_wrapper.step(orig_img, result)
                    for i in range(len(poseflow_result)):
                        result['result'][i]['idx'] = poseflow_result[i]['idx']

                final_result.append(result)
                # ==========================绘图=================================
                if self.opt.save_img or self.save_video or self.opt.vis:
                    if hm_data.size()[1] == 49:
                        from alphapose.utils.vis import vis_frame_dense as vis_frame, DEFAULT_FONT
                    elif self.opt.vis_fast:
                        from alphapose.utils.vis import vis_frame_fast as vis_frame, DEFAULT_FONT
                    else:
                        from alphapose.utils.vis import vis_frame, DEFAULT_FONT
                    # 开始绘图==============
                    img = vis_frame(orig_img, result, self.opt)
                    if self.head_pose and len(masks_list) != 0:
                        for p in masks_list:
                            pose_estimator.draw_annotation_box(
                                img, p[0], p[1], color=(128, 255, 128))
                        if self.opt.tracking:
                            # 行人重识别状态
                            for _id in ids:
                                _state = reid_states[_id]
                                index = _state['index']
                                bbox = _result[index]['box']
                                bbox = [bbox[0], bbox[0] + bbox[2], bbox[1], bbox[1] + bbox[3]]
                                cv2.putText(img, f'no focus: {round(_state["face_hide_rate"], 2)}',
                                            (int(bbox[0]), int((bbox[2] + 52))), DEFAULT_FONT,
                                            1, (255, 0, 0), 2)
                    # 结束绘图==============》显示图片
                    self.write_image(img, im_name, stream=stream if self.save_video else None)

    @staticmethod
    def get_reid_state(idx, reid_states, reid_global_states):
        # 获取重识别状态
        if idx in reid_states:
            self_state = reid_states[idx]
            if (reid_global_states['time'] - self_state['time']) > reid_loss_interval:
                self_state = {"time": time.time()}
                reid_states[idx] = self_state
            else:
                self_state['time'] = time.time()
        else:
            self_state = {"time": time.time()}
            reid_states[idx] = self_state
        return self_state

    @staticmethod
    def get_scaled_face_keypoints(face_keypoints):
        """
        获取标准化后的人脸关键点坐标
        :param face_keypoints: 脸部关键点
        :return: 标准化后的人脸关键点坐标，人脸框的位置
        """
        face_outline_keypoints = face_keypoints[:27]
        face_x1 = torch.min(face_outline_keypoints[:, 0])
        face_y1 = torch.min(face_outline_keypoints[:, 1])
        face_x2 = torch.max(face_outline_keypoints[:, 0])
        face_y2 = torch.max(face_outline_keypoints[:, 1])
        # 获取标准化的脸部坐标
        face_x1_y1 = torch.tensor([face_x1, face_y1])
        face_width = torch.tensor([face_x2 - face_x1, face_y2 - face_y1])
        scaled_face_keypoints = (face_keypoints - face_x1_y1) / face_width
        return scaled_face_keypoints, (face_x1, face_y1, face_x2, face_y2)

    @staticmethod
    def get_scaled_mouth_keypoints(face_keypoints):
        mouth_keypoints = face_keypoints[48:68]
        mouth_x1 = torch.min(mouth_keypoints[:, 0])
        mouth_y1 = torch.min(mouth_keypoints[:, 1])
        mouth_x2 = torch.max(mouth_keypoints[:, 0])
        mouth_y2 = torch.max(mouth_keypoints[:, 1])

        mouth_x1_y1 = torch.tensor([mouth_x1, mouth_y1])
        mouth_width = torch.tensor([mouth_x2 - mouth_x1, mouth_y2 - mouth_y1])
        scaled_mouth_keypoints = (mouth_keypoints - mouth_x1_y1) / mouth_width
        return scaled_mouth_keypoints, (mouth_x1, mouth_y1, mouth_x2, mouth_y2)

    @staticmethod
    def mouth_open_degree(scaled_mouth_keypoints):
        """
        计算张嘴程度
        :param scaled_mouth_keypoints: 按嘴部框范围标准化后的关键点坐标
        :return:
        """
        up_mouth_keypoints = scaled_mouth_keypoints[13:16]
        down_mouth_keypoints = scaled_mouth_keypoints[17:20]
        # 计算嘴对应点之间的距离
        mouth_distance = torch.linalg.norm(up_mouth_keypoints - down_mouth_keypoints, axis=1)
        return mouth_distance

    @staticmethod
    def estimate_head_pose(pose_estimator, face_68_keypoints, masks_list=None):
        """
        :param pose_estimator: 姿态评估器
        :param face_68_keypoints: 人脸68关键点
        :param masks_list: 保存姿态评估结果的列表
        :return:
        """
        if masks_list is None:
            masks_list = []
        marks = face_68_keypoints
        marks = np.float32(marks)
        pose = pose_estimator.solve_pose(marks)
        # Stabilize the pose.
        # 这段使用卡尔曼滤波平滑结果，目前没有必要
        # steady_pose = []
        # pose_np = np.array(pose).flatten()
        # for value, ps_stb in zip(pose_np, pose_stabilizers):
        #     ps_stb.update([value])
        #     steady_pose.append(ps_stb.state[0])
        # steady_pose = np.reshape(steady_pose, (-1, 3))
        # masks_list.append(steady_pose)
        masks_list.append(pose)
        return masks_list

    @staticmethod
    def face_hide(self_state, reid_global_states, face_naked, face_hide_lambda=face_hide_lambda):
        """

        :param self_state: 目标的状态对象
        :param reid_global_states: 全局的状态对象
        :param face_naked: 人脸裸露率
        :param face_hide_lambda: 平滑人脸遮挡率
        :return: 修改后的状态字典
        """
        face_hide_time = self_state.get("face_hide_time", 0)
        if face_naked < 0.5:
            face_hide_time += reid_global_states['interval']
        else:
            face_hide_time = 0
        # 动态遮挡率
        face_hide_rate = self_state.get("face_hide_rate", 0)
        if face_hide_time > face_hide_refresh_interval:
            face_hide_rate = (1 - face_hide_lambda) + face_hide_lambda * face_hide_rate
        elif face_hide_time == 0:
            face_hide_rate = face_hide_lambda * face_hide_rate
        self_state["face_hide_time"] = face_hide_time
        self_state["face_hide_rate"] = face_hide_rate
        return self_state

    def write_image(self, img, im_name, stream=None):
        if self.opt.vis:
            cv2.imshow("AlphaPose Demo", img)
            cv2.waitKey(30)
        if self.opt.save_img:
            cv2.imwrite(os.path.join(self.opt.outputpath, 'vis', im_name), img)
        if self.save_video:
            stream.write(img)

    @staticmethod
    def wait_and_put(queue, item):
        queue.put(item)

    @staticmethod
    def wait_and_get(queue):
        return queue.get()

    def save(self, boxes, scores, ids, hm_data, cropped_boxes, orig_img, im_name):
        # save next frame in the queue
        self.wait_and_put(self.result_queue, (boxes, scores, ids, hm_data, cropped_boxes, orig_img, im_name))

    def running(self):
        # indicate that the thread is still running
        return not self.result_queue.empty()

    def count(self):
        # indicate the remaining images
        return self.result_queue.qsize()

    def stop(self):
        # indicate that the thread should be stopped
        self.save(None, None, None, None, None, None, None)
        self.result_worker.join()

    def terminate(self):
        # directly terminate
        self.result_worker.terminate()

    def clear_queues(self):
        self.clear(self.result_queue)

    @staticmethod
    def clear(queue):
        while not queue.empty():
            queue.get()

    def results(self):
        # return final result
        print(self.final_result)
        return self.final_result

    @staticmethod
    def recognize_video_ext(ext=''):
        if ext == 'mp4':
            return cv2.VideoWriter_fourcc(*'mp4v'), '.' + ext
        elif ext == 'avi':
            return cv2.VideoWriter_fourcc(*'XVID'), '.' + ext
        elif ext == 'mov':
            return cv2.VideoWriter_fourcc(*'XVID'), '.' + ext
        else:
            print("Unknow video format {}, will use .mp4 instead of it".format(ext))
            return cv2.VideoWriter_fourcc(*'mp4v'), '.mp4'


class DataWriter:
    """修改前的DataWriter"""

    def __init__(self, cfg, opt, save_video=False,
                 video_save_opt=DEFAULT_VIDEO_SAVE_OPT,
                 queueSize=1024):
        self.cfg = cfg
        self.opt = opt
        self.video_save_opt = video_save_opt

        self.eval_joints = EVAL_JOINTS
        self.save_video = save_video
        self.heatmap_to_coord = get_func_heatmap_to_coord(cfg)
        # initialize the queue used to store frames read from
        # the video file
        if opt.sp:
            self.result_queue = Queue(maxsize=queueSize)
        else:
            self.result_queue = mp.Queue(maxsize=queueSize)

        if opt.save_img:
            if not os.path.exists(opt.outputpath + '/vis'):
                os.mkdir(opt.outputpath + '/vis')

        if opt.pose_flow:
            from trackers.PoseFlow.poseflow_infer import PoseFlowWrapper
            self.pose_flow_wrapper = PoseFlowWrapper(save_path=os.path.join(opt.outputpath, 'poseflow'))

    def start_worker(self, target):
        if self.opt.sp:
            p = Thread(target=target, args=())
        else:
            p = mp.Process(target=target, args=())
        # p.daemon = True
        p.start()
        return p

    def start(self):
        # start a thread to read pose estimation results per frame
        self.result_worker = self.start_worker(self.update)
        return self

    def update(self):
        final_result = []
        norm_type = self.cfg.LOSS.get('NORM_TYPE', None)
        hm_size = self.cfg.DATA_PRESET.HEATMAP_SIZE
        if self.save_video:
            # initialize the file video stream, adapt ouput video resolution to original video
            stream = cv2.VideoWriter(*[self.video_save_opt[k] for k in ['savepath', 'fourcc', 'fps', 'frameSize']])
            if not stream.isOpened():
                print("Try to use other video encoders...")
                ext = self.video_save_opt['savepath'].split('.')[-1]
                fourcc, _ext = self.recognize_video_ext(ext)
                self.video_save_opt['fourcc'] = fourcc
                self.video_save_opt['savepath'] = self.video_save_opt['savepath'][:-4] + _ext
                stream = cv2.VideoWriter(*[self.video_save_opt[k] for k in ['savepath', 'fourcc', 'fps', 'frameSize']])
            assert stream.isOpened(), 'Cannot open video for writing'
        # keep looping infinitelyd
        while True:
            # ensure the queue is not empty and get item
            (boxes, scores, ids, hm_data, cropped_boxes, orig_img, im_name) = self.wait_and_get(self.result_queue)
            if orig_img is None:
                # if the thread indicator variable is set (img is None), stop the thread
                if self.save_video:
                    stream.release()
                write_json(final_result, self.opt.outputpath, form=self.opt.format, for_eval=self.opt.eval)
                print("Results have been written to json.")
                return
            # image channel RGB->BGR
            orig_img = np.array(orig_img, dtype=np.uint8)[:, :, ::-1]
            if boxes is None or len(boxes) == 0:
                if self.opt.save_img or self.save_video or self.opt.vis:
                    self.write_image(orig_img, im_name, stream=stream if self.save_video else None)
            else:
                # location prediction (n, kp, 2) | score prediction (n, kp, 1)
                assert hm_data.dim() == 4
                # pred = hm_data.cpu().data.numpy()

                if hm_data.size()[1] == 136:
                    self.eval_joints = [*range(0, 136)]
                elif hm_data.size()[1] == 26:
                    self.eval_joints = [*range(0, 26)]
                pose_coords = []
                pose_scores = []
                for i in range(hm_data.shape[0]):
                    bbox = cropped_boxes[i].tolist()
                    pose_coord, pose_score = self.heatmap_to_coord(hm_data[i][self.eval_joints], bbox, hm_shape=hm_size,
                                                                   norm_type=norm_type)
                    pose_coords.append(torch.from_numpy(pose_coord).unsqueeze(0))
                    pose_scores.append(torch.from_numpy(pose_score).unsqueeze(0))
                preds_img = torch.cat(pose_coords)
                preds_scores = torch.cat(pose_scores)
                if not self.opt.pose_track:
                    boxes, scores, ids, preds_img, preds_scores, pick_ids = \
                        pose_nms(boxes, scores, ids, preds_img, preds_scores, self.opt.min_box_area)

                _result = []
                for k in range(len(scores)):
                    _result.append(
                        {
                            'keypoints': preds_img[k],
                            'kp_score': preds_scores[k],
                            'proposal_score': torch.mean(preds_scores[k]) + scores[k] + 1.25 * max(preds_scores[k]),
                            'idx': ids[k],
                            'box': [boxes[k][0], boxes[k][1], boxes[k][2] - boxes[k][0], boxes[k][3] - boxes[k][1]]
                        }
                    )

                result = {
                    'imgname': im_name,
                    'result': _result
                }

                if self.opt.pose_flow:
                    poseflow_result = self.pose_flow_wrapper.step(orig_img, result)
                    for i in range(len(poseflow_result)):
                        result['result'][i]['idx'] = poseflow_result[i]['idx']

                final_result.append(result)
                if self.opt.save_img or self.save_video or self.opt.vis:
                    if hm_data.size()[1] == 49:
                        from alphapose.utils.vis import vis_frame_dense as vis_frame
                    elif self.opt.vis_fast:
                        from alphapose.utils.vis import vis_frame_fast as vis_frame
                    else:
                        from alphapose.utils.vis import vis_frame
                    img = vis_frame(orig_img, result, self.opt)
                    self.write_image(img, im_name, stream=stream if self.save_video else None)

    def write_image(self, img, im_name, stream=None):
        if self.opt.vis:
            cv2.imshow("AlphaPose Demo", img)
            cv2.waitKey(500)
        if self.opt.save_img:
            cv2.imwrite(os.path.join(self.opt.outputpath, 'vis', im_name), img)
        if self.save_video:
            stream.write(img)

    def wait_and_put(self, queue, item):
        queue.put(item)

    def wait_and_get(self, queue):
        return queue.get()

    def save(self, boxes, scores, ids, hm_data, cropped_boxes, orig_img, im_name):
        # save next frame in the queue
        self.wait_and_put(self.result_queue, (boxes, scores, ids, hm_data, cropped_boxes, orig_img, im_name))

    def running(self):
        # indicate that the thread is still running
        return not self.result_queue.empty()

    def count(self):
        # indicate the remaining images
        return self.result_queue.qsize()

    def stop(self):
        # indicate that the thread should be stopped
        self.save(None, None, None, None, None, None, None)
        self.result_worker.join()

    def terminate(self):
        # directly terminate
        self.result_worker.terminate()

    def clear_queues(self):
        self.clear(self.result_queue)

    def clear(self, queue):
        while not queue.empty():
            queue.get()

    def results(self):
        # return final result
        print(self.final_result)
        return self.final_result

    def recognize_video_ext(self, ext=''):
        if ext == 'mp4':
            return cv2.VideoWriter_fourcc(*'mp4v'), '.' + ext
        elif ext == 'avi':
            return cv2.VideoWriter_fourcc(*'XVID'), '.' + ext
        elif ext == 'mov':
            return cv2.VideoWriter_fourcc(*'XVID'), '.' + ext
        else:
            print("Unknow video format {}, will use .mp4 instead of it".format(ext))
            return cv2.VideoWriter_fourcc(*'mp4v'), '.mp4'
