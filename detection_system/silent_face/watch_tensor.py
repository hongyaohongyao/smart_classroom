import torch
from tensorboardX import SummaryWriter

from silent_face.src.model_lib.MiniFASNet import MiniFASNetV2
from silent_face.src.utility import get_kernel

if __name__ == '__main__':
    # 其实就两句话
    model = MiniFASNetV2(conv6_kernel=get_kernel(80, 80, ))
    dummy_input = torch.rand(1, 3, 80, 80)  # 假设输入20张1*28*28的图片
    with SummaryWriter(comment='MiniFASNetV2') as w:
        w.add_graph(model, (dummy_input,))
