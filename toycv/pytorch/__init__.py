import os
import random

import numpy
import torch


def get_devices():
    devices = []
    backends = ["cuda", "mps", "ipu", "xpu", "mkldnn", "opengl", "opencl", "ideep", "hip", "ve", "fpga", "ort", "xla",
                "lazy", "vulkan", "meta", "hpu", "mtia", "privateuseone"]
    for backend in backends:
        if hasattr(torch.backends, backend):
            print(backend, getattr(torch, "has_" + backend))
            if getattr(torch, "has_" + backend):
                print("has_" + backend, getattr(torch, "has_" + backend))
                devices.append(backend)

    devices.append("cpu")
    return devices


def fix_seed(seed=18):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
