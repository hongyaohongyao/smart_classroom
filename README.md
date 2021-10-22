# 智慧教室

课堂专注度及考试作弊系统、课堂动态点名，情绪识别、表情识别和人脸识别结合

相关项目

- [PyQt Demo](https://github.com/hongyaohongyao/smart_classroom_demo) 
- [Java 版本](https://github.com/hongyaohongyao/SmartClassroomJava) 

## 课堂专注度分析

课堂专注度+表情识别

![正面专注度](.img/README/正面专注度.png)

## 作弊检测

#### 关键点计算方法

转头(probe)+低头(peep)+传递物品(passing)

![正面作弊动作](.img/README/正面作弊动作.png)

侧面的传递物品识别

![侧面作弊动作](.img/README/侧面作弊动作.png)

#### 逻辑回归关键点

![image-20210620223428871](.img/README/image-20210620223428871.png)

## 下载权重

### 1. [Halpe dataset](https://github.com/Fang-Haoshu/Halpe-FullBody) (136 keypoints)

| Model                                                        | Backbone | Detector | Input Size | AP   | Speed       | Download                                                     | Config                                                       | Training Log                                                 |
| ------------------------------------------------------------ | -------- | -------- | ---------- | ---- | ----------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| [Fast Pose](https://github.com/MVIG-SJTU/AlphaPose/blob/master/configs/halpe_136/resnet/256x192_res50_lr1e-3_1x.yaml) | ResNet50 | YOLOv3   | 256x192    | 69.0 | 3.54 iter/s | [Google](https://drive.google.com/file/d/17vnGsMDbG4rf50kyj586BVJsiAspQv5v/view?usp=sharing) [Baidu](https://pan.baidu.com/s/1--9DsFjTyQrTMwsMjY7FGg) | [cfg](https://github.com/MVIG-SJTU/AlphaPose/blob/master/configs/halpe_136/resnet/256x192_res50_lr1e-3_2x-regression.yaml) | [log](https://github.com/MVIG-SJTU/AlphaPose/blob/master/docs) |

- 放到detection_system/checkpoints

### 2. Human-ReID based tracking (Recommended)

Currently the best performance tracking model. Paper coming soon.

#### Getting started

Download [human reid model](https://mega.nz/#!YTZFnSJY!wlbo_5oa2TpDAGyWCTKTX1hh4d6DvJhh_RUA2z6i_so) and place it into `AlphaPose/trackers/weights/`.

Then simply run alphapose with additional flag `--pose_track`

You can try different person reid model by modifing `cfg.arch` and `cfg.loadmodel` in `./trackers/tracker_cfg.py`.

If you want to train your own reid model, please refer to this [project](https://github.com/KaiyangZhou/deep-person-reid)

### 3. Yolo Detector

Download the object detection model manually: **yolov3-spp.weights**([Google Drive](https://drive.google.com/open?id=1D47msNOOiJKvPOXlnpyzdKA3k6E97NTC) | [Baidu pan](https://pan.baidu.com/s/1Zb2REEIk8tcahDa8KacPNA)). Place it into `detector/yolo/data`.

### 4. face boxes 预训练权重

[google drive](https://drive.google.com/file/d/1tRVwOlu0QtjvADQ2H7vqrRwsWEmaqioI) 

- 放到face_recog/weights文件夹下

### 5. 其他

[百度云](https://pan.baidu.com/s/1X6TR2jiqdqg3Zi8wl7mkxw)  提取码：rwtl 

**人脸识别**：dlib_face_recognition_resnet_model_v1.dat

- detection_system/face_recog/weights

**人脸对齐**：shape_predictor_68_face_landmarks.dat

- detection_system/face_recog/weights

**作弊动作分类器**：cheating_detector_rfc_kp.pkl

- detection_system/weights

## 使用

### 运行setup.py安装必要内容

```shell
python setup.py build develop
```
[windows上安装scipy1.1.0可能会遇到的问题](https://github.com/MVIG-SJTU/AlphaPose/issues/722)

### 运行demo_inference.py

将detection_system设置为source root

![image-20210514153925536](.img/README/image-20210514153925536.png)

使用摄像头运行程序

```
python demo_inference.py --vis --webcam 0
```

# 参考项目

- [人体姿态估计 AlphaPose](https://github.com/MVIG-SJTU/AlphaPose) 
- [头部姿态估计 head-pose-estimation](https://github.com/yinguobing/head-pose-estimation) 
- [人脸检测 faceboxes](https://github.com/zisianw/FaceBoxes.PyTorch) 
- [静默人脸识别 Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing) 

# 相关信息

