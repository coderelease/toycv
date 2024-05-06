import numpy
import torch
from torch import Tensor


def numpy_to_tensor(image_or_video, with_channel=True):
    """
    Convert numpy.ndarray to torch.Tensor.
    If the input is 2D with shape (h, w), the output will be (h, w).
    If the input is 3D with shape (h, w, c), the output will be (c, h, w).
    If the input is 4D with shape (frames, h, w, c), the output will be (frames, c, h, w).
    :param with_channel: whether there is a channel dimension in the input.
    :param image_or_video: numpy.ndarray. (h, w) or (h, w, c) or (frames, h, w, c).
    :return: torch.Tensor with shape (c, h, w) or (frames, c, h, w).
    """
    if isinstance(image_or_video, numpy.ndarray):
        if image_or_video.ndim == 2:
            return torch.from_numpy(image_or_video).float()
        elif image_or_video.ndim == 3 and with_channel:
            return torch.from_numpy(image_or_video.transpose((2, 0, 1))).float()
        elif image_or_video.ndim == 3 and not with_channel:
            return torch.from_numpy(image_or_video).float()
        elif image_or_video.ndim == 4:
            return torch.from_numpy(image_or_video.transpose((3, 0, 1, 2))).float()
        else:
            raise ValueError(f"image_or_video should be 3D or 4D. Got {image_or_video.ndim}D.")
    else:
        raise TypeError(f"pic should be ndarray. Got {type(image_or_video)}")


def tensor_to_numpy(image_or_video_tensor: Tensor, with_channel=True):
    """
    Convert torch.Tensor to numpy.ndarray.
    If the input is 2D with shape (h, w), the output will be (h, w).
    If the input is 3D with shape (c, h, w), the output will be (h, w, c).
    If the input is 4D with shape (frames, c, h, w), the output will be (frames, h, w, c).
    :param with_channel: whether there is a channel dimension in the input.
    :param image_or_video_tensor: torch.Tensor with shape (c, h, w) or (frames, c, h, w).
    :return: numpy.ndarray with shape (h, w, c) or (frames, h, w, c).
    """
    if isinstance(image_or_video_tensor, Tensor):
        if image_or_video_tensor.dim() == 2:
            return image_or_video_tensor.numpy()
        elif image_or_video_tensor.dim() == 3 and with_channel:
            return image_or_video_tensor.permute(1, 2, 0).numpy()
        elif image_or_video_tensor.dim() == 3 and not with_channel:
            return image_or_video_tensor.numpy()
        elif image_or_video_tensor.dim() == 4:
            return image_or_video_tensor.permute(1, 2, 3, 0).numpy()
        else:
            raise ValueError(f"image_or_video_tensor should be 3D or 4D. Got {image_or_video_tensor.dim()}D.")
    else:
        raise TypeError(f"pic should be Tensor. Got {type(image_or_video_tensor)}")
