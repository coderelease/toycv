import os
from typing import Optional, List, Literal, Union

import cv2
import imagesize
import numpy


def get_image_size(image_path):
    """
    :param image_path: 图像路径
    :return: height, width
    """

    width, height = imagesize.get(image_path)

    if (height, width) == (-1, -1):
        image: numpy.ndarray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # 获取图像尺寸
        height, width = image.shape

    return height, width


def read_image(image_path, grey: bool = False, resize=None, rgb_mode: bool = True):
    """
    读取图像函数

    :param image_path: 图像路径
    :param grey: 是否将图像转换为灰度图像，默认为False
    :param resize: (h, w)，默认为None
    :param rgb_mode: 使用opencv读出的图像是BGR格式，是否将图像转换为RGB格式，默认为True
    :return: 读取到的图像数组
    """

    if grey:
        # 读取灰度图像
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    else:
        # 读取彩色图像
        image = cv2.imread(image_path)
        if rgb_mode:
            # 将BGR格式转换为RGB格式
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if resize:
        image = cv2.resize(image, (resize[1], resize[0]))

    return image


def read_images(image_paths: List[str], grey: bool = False, resize=None, rgb_mode: bool = True):
    if not len(image_paths):
        raise ValueError("The length of image_paths should not be 0.")

    if not resize:
        resize = get_image_size(image_paths[0])
    image_list = []
    for image_path in image_paths:
        image = read_image(image_path, grey, resize, rgb_mode)
        image_list.append(image)

    return numpy.stack(image_list, axis=0)


def get_video_info(video_path):
    """
    获取帧速率、帧数、宽度和高度
    :param video_path: 视频文件地址
    :return: 帧数、宽度、高度和帧速率。如果视频文件无法打开，则返回None。
    """

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    # 检查视频文件是否成功打开
    if not cap.isOpened():
        raise ValueError(f"Can't open video file {video_path}.")

    # 获取帧速率、帧数、宽度和高度
    frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    # 释放VideoCapture对象
    cap.release()

    # 返回帧数、宽度、高度和帧速率
    return frames_count, fps, height, width


def read_video(video_path,
               *specific_frames: Optional[int],
               # start: int = 0, stop: int = -1, step: int = 1,
               resize=None,
               rgb_mode=True):
    """
    读取视频文件，并返回指定范围的帧。 \n
    运行此函数需要cv2模块进行视频文件的读取，numpy模块进行高效的数值计算。 \n
    示例：\n
    video = read_video("example.mp4", 0, 100, 200, 300, resize=(224, 224), rgb_mode=True) \n
    video = read_video("example.mp4", *range(0, 1000, 100), resize=(224, 224), rgb_mode=True) \n

    :param video_path: 视频文件的路径，字符串格式。
    :param specific_frames: 需要读取的特定帧的序列，每个元素为帧的索引，整数类型。如果该参数为空，则读取视频中的所有帧。
    :param resize: 重新设定帧的宽和高的尺寸，形式为(h, w)，默认为None，即不改变帧的尺寸。
    :param rgb_mode: 布尔值，确定是否将帧的颜色空间从BGR转换为RGB，为True时进行转换，默认为True。

    :return: 如果读取成功，返回一个numpy数组，形式为 (帧数，高，宽，颜色通道数)，即每一帧图像上包含的像素信息。如果读取失败，返回None。

    :raise ValueError: 当需要读取的特定帧的索引超出指定范围时，引发ValueError。
    """
    cap = cv2.VideoCapture(video_path)

    frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # check specific_frames
    if specific_frames:
        for frame in specific_frames:
            if frame < 0 or frame >= frames_count:
                raise ValueError(f"frame index {frame} out of range [0, {frames_count}).")
    else:
        specific_frames = range(frames_count)

    video = []
    for frame in specific_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ret, frame = cap.read()
        if rgb_mode:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if resize:
            frame = cv2.resize(frame, (resize[1], resize[0]))
        video.append(frame)

    cap.release()

    # post-processing frames
    video = numpy.stack(video, axis=0)

    return video


def write_image(image: numpy.ndarray, filename: str, *, resize=None, rgb_mode=True) -> None:
    """
    将图像写入文件。
    :param image: numpy.array, 输入图像。
    :param filename: str, 输出文件名。
    :param resize: (h, w)，默认为None
    :param rgb_mode: cv2是以BGR的形式写入，如果当前是RGB需要转化为BGR，其他软件读取才会正常。默认为True。
    """
    if rgb_mode:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if resize:
        image = cv2.resize(image, (resize[1], resize[0]))
    cv2.imwrite(filename, image)

    print(f"{filename} written successfully!")


# 将文件格式及对应的fourcc编码方式放入常量中
FOURCC_MAP = {'.mp4': 'mp4v', '.avi': 'XVID', '.mkv': 'avc1',
              '.mov': 'mp4v', '.3gp': 'mp4v', '.webm': 'VP80', '.wmv': 'WMV1'}


def write_video(video: numpy.ndarray, filename: str, *, fps: int = 30, resize=None, rgb_mode=True) -> None:
    """
    将cv2读取的维度格式FHWC（frame_count, height, width, channel_count）视频帧写入文件。
    如果缺少channel_count维度，则会repeat补上。
    函数接受一个视频帧、一个输出文件名和一个视频帧率（默认为30帧/秒），

    使用OpenCV的VideoWriter函数创建视频写入对象，然后逐帧将视频写入文件。
    最后，释放视频写入对象。注意，为了使写入的视频具有更好的兼容性，我们选择了MP4编码格式。

    :param video: numpy.array, 输入视频帧。
    :param filename: str, 输出文件名。
    :param fps: int, 输出视频帧率。
    :param resize: (h, w)，默认为None
    :param rgb_mode: cv2是以BGR的形式写入，如果当前是RGB需要转化为RGB，默认为True。

    :raise Exception: 当输入视频维度不是FHW[C]时，引发异常。
    """

    video = numpy.array(video, dtype=numpy.uint8)

    if video.ndim not in [3, 4]:
        raise Exception("The input video should be a 3D or 4D numpy array, but got {}D array.".format(video.ndim))

    if video.ndim == 3:
        video = numpy.repeat(video[..., numpy.newaxis], 3, axis=-1)

    frame_count, height, width, channel_count = video.shape

    _, ext = os.path.splitext(filename)
    fourcc = cv2.VideoWriter_fourcc(*FOURCC_MAP[ext.lower()])  # 获取fourcc编码

    if resize:
        # 在resize后更新宽和高
        height, width = resize
        video = numpy.array([cv2.resize(frame, (resize[1], resize[0])) for frame in video])

    if rgb_mode:
        # 如果在RGB模式下，将所有帧一次性转换以提高效率
        video = [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in video]

    writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))

    for frame in video:
        writer.write(frame)

    writer.release()

    print(filename, "written")


def keep_ratio_resize_image(image, target_size, position: Union[Literal["center", "random"],] = "center"):
    """
    根据target_size，如果图像的比例和target_size不一致，则先按照target_size的比例切割图像，然后再进行缩放
    :param position:
    :param image: numpy.array, (h, w[, c])，输入图像。
    :param target_size: (h, w)，目标尺寸。
    :return: numpy.array, 缩放后的图像。
    """
    ndim = image.ndim
    h, w = image.shape[:2]
    th, tw = target_size

    if h / w > th / tw:
        # 高度过长，需要先按照宽度切割
        new_h = int(w / tw * th)
        if position == "center":
            start = (h - new_h) // 2
            image = image[start: start + new_h, :, ...]
        elif position == "random":
            start = numpy.random.randint(0, h - new_h)
            image = image[start: start + new_h, :, ...]
        else:
            raise ValueError(f"Unknown position {position}.")
    else:
        # 宽度过长，需要先按照高度切割
        new_w = int(h / th * tw)
        if position == "center":
            start = (w - new_w) // 2
            image = image[:, start: start + new_w, ...]
        elif position == "random":
            start = numpy.random.randint(0, w - new_w)
            image = image[:, start: start + new_w, ...]
        else:
            raise ValueError(f"Unknown position {position}.")

    image = cv2.resize(image, (tw, th))

    if ndim == 3 and image.ndim == 2:
        return image[..., numpy.newaxis]
    else:
        return image


def keep_ratio_resize_video(video, target_size, position: Union[Literal["center", "random"],] = "center"):
    """
    根据target_size，如果视频的比例和target_size不一致，则先按照target_size的比例切割视频，然后再进行缩放
    :param position:
    :param video: numpy.array, (frames, h, w[, c])，输入视频。
    :param target_size: (h, w)，目标尺寸。
    :return: numpy.array, 缩放后的视频。
    """
    ndim = video.ndim
    frames, h, w = video.shape[:3]
    th, tw = target_size

    if h / w > th / tw:
        # 高度过长，需要先按照宽度切割
        new_h = int(w / tw * th)
        if position == "center":
            start = (h - new_h) // 2
            video = video[:, start: start + new_h, :]
        elif position == "random":
            start = numpy.random.randint(0, h - new_h)
            video = video[:, start: start + new_h, :]
        else:
            raise ValueError(f"Unknown position {position}.")
    else:
        # 宽度过长，需要先按照高度切割
        new_w = int(h / th * tw)
        if position == "center":
            start = (w - new_w) // 2
            video = video[:, :, start: start + new_w]
        elif position == "random":
            start = numpy.random.randint(0, w - new_w)
            video = video[:, :, start: start + new_w]
        else:
            raise ValueError(f"Unknown position {position}.")

    video = numpy.array([cv2.resize(frame, (tw, th)) for frame in video])

    if ndim == 3 and video.ndim == 2:
        return video[..., numpy.newaxis]
    elif ndim == 4 and video.ndim == 3:
        return video[..., numpy.newaxis]
    else:
        return video


def keep_ratio_resize(image_or_video, target_size, with_channel=True,
                      position: Union[Literal["center", "random"],] = "center"):
    """
    根据target_size，如果图像或视频的比例和target_size不一致，则先按照target_size的比例切割图像或视频，然后再进行缩放
    :param position: {"center", "random"}，默认为"center"，切割位置。
    :param image_or_video: numpy.array, (h, w[, c]) or (frames, h, w[, c])，输入图像或视频。
    :param target_size: (h, w)，目标尺寸。
    :param with_channel: 是否有通道维度，默认为True。
    :return: numpy.array, 缩放后的图像或视频。
    """
    if image_or_video.ndim == 2:
        return keep_ratio_resize_image(image_or_video, target_size, position)
    elif image_or_video.ndim == 3 and with_channel:
        return keep_ratio_resize_image(image_or_video, target_size, position)
    elif image_or_video.ndim == 3 and not with_channel:
        return keep_ratio_resize_video(image_or_video, target_size, position)
    elif image_or_video.ndim == 4:
        return keep_ratio_resize_video(image_or_video, target_size, position)
    else:
        raise ValueError(f"image_or_video should be 3D or 4D. Got {image_or_video.ndim}D.")


def resize_image(image, target_size):
    """
    将图像缩放到指定尺寸。
    :param image: numpy.array, 输入图像。
    :param target_size: (h, w)，目标尺寸。
    :return: numpy.array, 缩放后的图像。
    """
    return cv2.resize(image, (target_size[1], target_size[0]))


def resize_video(video, target_size):
    """
    将视频缩放到指定尺寸。
    :param video: numpy.array, 输入视频。
    :param target_size: (h, w)，目标尺寸。
    :return: numpy.array, 缩放后的视频。
    """
    return numpy.array([cv2.resize(frame, (target_size[1], target_size[0])) for frame in video])


def resize(image_or_video, target_size, with_channel=True):
    """
    将图像或视频缩放到指定尺寸。
    :param image_or_video: numpy.array, (h, w[, c]) or (frames, h, w[, c])，输入图像或视频。
    :param target_size: (h, w)，目标尺寸。
    :param with_channel: 是否有通道维度，默认为True。
    :return: numpy.array, 缩放后的图像或视频。
    """
    if image_or_video.ndim == 2:
        return resize_image(image_or_video, target_size)
    elif image_or_video.ndim == 3 and with_channel:
        return resize_image(image_or_video, target_size)
    elif image_or_video.ndim == 3 and not with_channel:
        return resize_video(image_or_video, target_size)
    elif image_or_video.ndim == 4:
        return resize_video(image_or_video, target_size)
    else:
        raise ValueError(f"image_or_video should be 3D or 4D. Got {image_or_video.ndim}D.")


def pil_to_numpy(x):
    return numpy.array(x)


def to_grey(image_or_video: numpy.ndarray, with_channel=True, keep_dim=True):
    """
    Convert ndarray image or video to grey.
    :param keep_dim: whether to keep the channel dimension.
    :param image_or_video:  numpy.ndarray, (h, w) or (h, w, c) or (frames, h, w, c).
    :param with_channel:  whether there is a channel dimension in the input.
    :return: numpy.ndarray, (h, w) or (h, w, 1) or (frames, h, w, 1).
    """

    if image_or_video.ndim == 2:
        # 已经是灰度图像
        result_image = image_or_video
    elif image_or_video.ndim == 3 and with_channel:
        result_image = numpy.mean(image_or_video, axis=-1)
    elif image_or_video.ndim == 3 and not with_channel:
        # 已经是灰度图像
        result_image = image_or_video
    elif image_or_video.ndim == 4:
        result_image = numpy.mean(image_or_video, axis=-1)
    else:
        raise ValueError(
            f"image_or_video should be 2D (gray image), 3D (grey video or rgb image) or 4D (rgb video). "
            f"Got {image_or_video.ndim}D.")
    if keep_dim:
        return result_image[..., numpy.newaxis]
    else:
        return result_image


def cv2_binary(image):
    """
    cv2二值化，对MNIST效果很好
    :param image:
    :return:
    """
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # image = cv2.GaussianBlur(image, (5, 5), 0)
    is_255 = numpy.max(image) > 1
    is_float = image.dtype == numpy.float32 or image.dtype == numpy.float64
    if not is_255:
        image = (image * 255).astype(numpy.uint8)
    if is_float:
        image = image.astype(numpy.uint8)
    # print(image.shape, image.dtype)
    # 高斯模糊
    image = cv2.GaussianBlur(image, (3, 3), 0)

    # 直方图均衡化
    equalized = cv2.equalizeHist(image)

    # 全局 Otsu 二值化
    _, global_otsu = cv2.threshold(equalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 自适应阈值
    adaptive_thresh = cv2.adaptiveThreshold(equalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # 结合 Otsu 和自适应阈值的结果
    combined = cv2.bitwise_and(global_otsu, adaptive_thresh)

    if not is_255:
        combined = combined / 255.0

    return combined
