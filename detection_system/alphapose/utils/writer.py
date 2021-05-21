import os
from queue import Queue
from threading import Thread

import cv2
import numpy as np
import torch
import torch.multiprocessing as mp

from alphapose.utils.pPose_nms import pose_nms, write_json
from alphapose.utils.transforms import get_func_heatmap_to_coord
from kp_analysis import action_analysis, action_classifier
from pose_estimation.pose_estimator import PoseEstimator
from utils.attention_by_expression_angle import attention_degrees
from utils.reid_states import ReIDStates
from utils.scene_masker import SceneMasker

DEFAULT_VIDEO_SAVE_OPT = {
    'savepath': 'examples/res/1.mp4',
    'fourcc': cv2.VideoWriter_fourcc(*'mp4v'),
    'fps': 25,
    'frameSize': (800, 480)
}

EVAL_JOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

face_hide_refresh_interval = 1.5  # 单位秒
default_focus_lambda = 0.25
head_pose_roll_correction = 0


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
            self.reid_states = ReIDStates()
        # 是否使用场景蒙版
        self.scene_mask = SceneMasker(opt.scene_mask) if opt.scene_mask else None

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
            self.pose_estimator = PoseEstimator(img_size=self.opt.img_size)
            # Introduce scalar stabilizers for pose.
            # pose_stabilizers = [Stabilizer(
            #     state_num=2,
            #     measure_num=1,
            #     cov_process=0.1,
            #     cov_measure=0.1) for _ in range(6)]

            face_naked_rate = []  # 所有人的脸部露出率
        # keep looping infinitelyd
        while True:
            if self.opt.tracking:  # 处理重识别状态
                self.reid_states.next_frame()
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
                    self.deal_objects(np.stack(boxes), scores, ids, hm_data,
                                      cropped_boxes, orig_img, im_name,
                                      preds_img, preds_scores)
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
                    pose_list = self.pose_list
                    img = vis_frame(orig_img, result, self.opt)
                    if self.head_pose and len(pose_list) != 0:
                        self.draw_pose(self.pose_estimator, img, pose_list)
                        if self.opt.tracking:
                            # 行人重识别状态
                            for reid in ids:
                                self.draw_objects(img, reid, _result, DEFAULT_FONT)

                    if self.scene_mask and self.opt.show_scene_mask:
                        img = self.scene_mask.mask_on_img(img)
                    # 结束绘图==============》显示图片
                    self.write_image(img, im_name, stream=stream if self.save_video else None)

    def deal_objects(self, boxes, scores, ids, hm_data,
                     cropped_boxes, orig_img, im_name,
                     preds_img, preds_scores):
        """
        处理识别出来的目标
        """
        object_num = preds_img.shape[0]  # 目标数量

        # 场景位置识别 场景遮罩
        # print(self.scene_mask.is_in_seat(boxes))
        indexes = torch.arange(0, len(preds_img))  # 辅助索引
        if self.head_pose:  # 头部状态估计
            self.emoji_available_list = []  # 判断是否要识别表情
            # 取出脸部关键点
            face_keypoints = preds_img[:, 26:94]
            face_keypoints_scores = preds_scores[:, 26:94]
            # =====脸部露出判定======
            naked_faces = torch.sum(face_keypoints_scores[:, 27:48, 0] > 0.01,
                                    dim=1) / 21  # 这部分暂时不包括嘴部数据
            naked_mouths = torch.sum(face_keypoints_scores[:, 48:68, 0] > 0.1,
                                     dim=1) / 20  # 这部分是嘴部的裸露程度
            # 判断是否能够识别表情
            self.emoji_available_list.extend(indexes[(naked_faces > 0.5) & (naked_mouths > 0.5)])
            # 头部姿态估计 也包括其他姿态估计
            pose_estimator = self.pose_estimator
            self.pose_list = [{'head': self.estimate_head_pose(pose_estimator, face_keypoints[i]),
                               'body': None,
                               'neck': self.estimate_neck_pose(pose_estimator, preds_img[i, [17, 18, 5, 6]]),
                               'index': i}
                              for i in range(object_num)]
            if self.opt.analyse_focus:
                self.angles = [pose['head'][0][0][0] * 90 for pose in self.pose_list]
                self.attention_scores = attention_degrees(face_keypoints, self.angles)
            if self.opt.analyse_cheating:
                # 伸手识别处理
                self.is_passing = action_analysis.is_passing(preds_img)
                self.passing_tips = [
                    "left" if p == 1 else "right" if p == -1 else 'no'
                    for p in self.is_passing
                ]
                self.actions = action_classifier.action_classify(preds_img)
                self.actions_texts = [action_classifier.action_type[i] for i in self.actions]
                self.actions_colors = [(255, 0, 0) if i <= 3 else (0, 0, 255) for i in self.actions]
        # 逐个处理
        for i in range(object_num):
            if self.opt.tracking:
                self_state = self.reid_states[ids[i]]
                self_state['index'] = i
            if self.head_pose:
                # ====指标====脸部遮挡检测=======
                if self_state is not None and self.opt.analyse_focus:
                    self.focus_rates(ids[i], self.attention_scores[i], naked_faces[i])
                # ==口型识别== 打哈欠和说话
                if naked_mouths[i] > 0.5 and False:
                    scaled_mouth_keypoints, _ = self.get_scaled_mouth_keypoints(face_keypoints)
                    mouth_distance = self.mouth_open_degree(scaled_mouth_keypoints)
                    if mouth_distance[1] > 0.3:
                        open_mouth = "open mouth!!!!"
                    elif mouth_distance[1] > 0.2:
                        open_mouth = "open"
                    else:
                        open_mouth = ""
                    print(mouth_distance, open_mouth)
            # =====伸手识别=====
        # =====开始表情识别=====

    def draw_objects(self, img, reid, result, font):
        self_state = self.reid_states[reid]
        i = self_state['index']
        bbox = result[i]['box']
        bbox = [bbox[0], bbox[0] + bbox[2], bbox[1], bbox[1] + bbox[3]]
        if self.opt.analyse_focus:
            angle = round(self.angles[i], 2)
            focus_rate = round(self_state["focus_rate"], 2)
            text = f'focus:{focus_rate}'
            cv2.putText(img, text,
                        (int(bbox[0]), int((bbox[2] + 52))), font,
                        0.6, (0, 0, 255), 2)
            text = f'angle:{angle}'
            cv2.putText(img, text,
                        (int(bbox[0]), int((bbox[2] + 72))), font,
                        0.6, (0, 0, 255), 2)

        if self.opt.analyse_cheating:
            passing_tips = f'passing: {self.passing_tips[i]}'
            cv2.putText(img, passing_tips,
                        (int(bbox[0]), int((bbox[2] + 52))), font,
                        0.6, (0, 0, 255), 2)
            action_text = self.actions_texts[i]
            action_color = self.actions_colors[i]
            cv2.putText(img, action_text,
                        (int(bbox[0]), int((bbox[2] + 72))), font,
                        0.6, action_color, 2)

    @staticmethod
    def draw_pose(pose_estimator, img, pose_list):
        for pose in pose_list:
            # pose_estimator.draw_annotation_box(
            #     img, p[0], p[1], color=(128, 255, 128))
            head_pose = pose['head']  # 头部姿态
            pose_estimator.draw_axis(
                img, head_pose[0], head_pose[1])
            neck_pose = pose['neck']  # 颈部姿态
            # pose_estimator.draw_axis(
            #     img, neck_pose[0], neck_pose[1])
            # r2 = head_pose[0][1]
            # print(head_pose[0][0], head_pose[0][1], head_pose[0][2])

            # print(r1, r2, abs(r1 - r2))
            # body_pose = pose['body']
            # if body_pose is not None:
            #     pose_estimator.draw_axis(
            #         img, body_pose[0], body_pose[1])
            #     r1 = body_pose[0][1]
            #     r2 = head_pose[0][1]
            #     print(abs(r1 - r2), r1, r2)

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
    def estimate_head_pose(pose_estimator, face_68_keypoints):
        """
        :param pose_estimator: 姿态评估器
        :param face_68_keypoints: 人脸68关键点
        :param masks_list: 保存姿态评估结果的列表
        :return:
        """
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
        return pose

    @staticmethod
    def estimate_body_pose(pose_estimator, body_keypoints):
        """
        :param body_keypoints:
        :param pose_estimator: 姿态评估器
        :return:
        """
        marks = body_keypoints
        marks = np.float32(marks)
        pose = pose_estimator.solve_body_pose(marks)
        # Stabilize the pose.
        # 这段使用卡尔曼滤波平滑结果，目前没有必要
        # steady_pose = []
        # pose_np = np.array(pose).flatten()
        # for value, ps_stb in zip(pose_np, pose_stabilizers):
        #     ps_stb.update([value])
        #     steady_pose.append(ps_stb.state[0])
        # steady_pose = np.reshape(steady_pose, (-1, 3))
        # masks_list.append(steady_pose)
        return pose

    @staticmethod
    def estimate_neck_pose(pose_estimator, neck_keypoints):
        """
        :param neck_keypoints:
        :param pose_estimator: 姿态评估器
        :return:
        """
        marks = neck_keypoints
        marks = np.float32(marks)
        pose = pose_estimator.solve_neck_pose(marks)
        # Stabilize the pose.
        # 这段使用卡尔曼滤波平滑结果，目前没有必要
        # steady_pose = []
        # pose_np = np.array(pose).flatten()
        # for value, ps_stb in zip(pose_np, pose_stabilizers):
        #     ps_stb.update([value])
        #     steady_pose.append(ps_stb.state[0])
        # steady_pose = np.reshape(steady_pose, (-1, 3))
        # masks_list.append(steady_pose)
        return pose

    def focus_rates(self, reid, attention_score, face_naked, focus_lambda=default_focus_lambda):
        """

        :param attention_score:  注意力分数
        :param reid: 目标的重识别id
        :param face_naked: 人脸露出率
        :param focus_lambda: 平滑专注度
        :return: 修改后的状态字典
        """

        if face_naked < 0.5:
            face_hide_time = self.reid_states.timer_set(reid, "face_hide_time")
        else:
            self.reid_states.timer_reset(reid, "face_hide_time")
            face_hide_time = 0
        # 动态遮挡率
        if face_hide_time > face_hide_refresh_interval:
            focus_rate = self.reid_states.smooth_set(reid, "focus_rate", 0, focus_lambda)
        elif face_hide_time == 0:
            focus_rate = self.reid_states.smooth_set(reid, "focus_rate", attention_score, focus_lambda)
        else:
            focus_rate = self.reid_states.get(reid, "focus_rate")
        return focus_rate

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
