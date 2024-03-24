import os
from typing import Optional, List

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
        image = cv2.resize(image, resize)

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
    :param resize: 重新设定帧的宽和高的尺寸，形式为(w, h)，默认为None，即不改变帧的尺寸。
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
            frame = cv2.resize(frame, resize)
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
        image = cv2.resize(image, resize)
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
        width, height = resize
        video = numpy.array([cv2.resize(frame, resize) for frame in video])

    if rgb_mode:
        # 如果在RGB模式下，将所有帧一次性转换以提高效率
        video = [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in video]

    writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))

    for frame in video:
        writer.write(frame)

    writer.release()

    print(filename, "written")
