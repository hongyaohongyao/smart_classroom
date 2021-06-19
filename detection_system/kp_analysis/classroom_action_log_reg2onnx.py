from pathlib import Path

import onnx
import torch

from kp_analysis.logistic_regression import LogisticRegression


def file_size(file):
    # Return file size in MB
    return Path(file).stat().st_size / 1e6


def colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
    colors = {'black': '\033[30m',  # basic colors
              'red': '\033[31m',
              'green': '\033[32m',
              'yellow': '\033[33m',
              'blue': '\033[34m',
              'magenta': '\033[35m',
              'cyan': '\033[36m',
              'white': '\033[37m',
              'bright_black': '\033[90m',  # bright colors
              'bright_red': '\033[91m',
              'bright_green': '\033[92m',
              'bright_yellow': '\033[93m',
              'bright_blue': '\033[94m',
              'bright_magenta': '\033[95m',
              'bright_cyan': '\033[96m',
              'bright_white': '\033[97m',
              'end': '\033[0m',  # misc
              'bold': '\033[1m',
              'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']


model_path = '../weights/classroom_action_lr_front_v1.pth'
if __name__ == '__main__':
    prefix = colorstr('ONNX:')
    model = LogisticRegression(28, 16)
    model.load_state_dict(torch.load(model_path))
    model.cpu().eval()
    inp = torch.zeros(1, 28)
    f = model_path.replace('.pth', '.onnx')  # filename
    opset_version = 13
    dynamic = False
    torch.onnx.export(model, inp, f, verbose=False, opset_version=opset_version, input_names=['keypoints'],
                      training=torch.onnx.TrainingMode.EVAL,
                      do_constant_folding=True,
                      dynamic_axes={'keypoints': {0: 'batch', 1: 'kp28'},  # size(1,3,640,640)
                                    'output': {0: 'batch', 1: 'classes'}} if dynamic else None)
    # Checks
    model_onnx = onnx.load(f)  # load onnx model
    onnx.checker.check_model(model_onnx)  # check onnx model
    # print(onnx.helper.printable_graph(model_onnx.graph))  # print

    # Simplify
    simplify = True
    if simplify:
        try:
            import onnxsim

            print(f'simplifying with onnx-simplifier {onnxsim.__version__}...')
            model_onnx, check = onnxsim.simplify(
                model_onnx,
                dynamic_input_shape=dynamic,
                input_shapes={'images': list(inp.shape)} if dynamic else None)
            assert check, 'assert check failed'
            onnx.save(model_onnx, f)
        except Exception as e:
            print(f'{prefix} simplifier failure: {e}')
    print(f'{prefix} export success, saved as {f} ({file_size(f):.1f} MB)')
