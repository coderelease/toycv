from typing import List, Literal

import matplotlib.pyplot as plt
import numpy
import numpy.random
import seaborn
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from torchvision.models import resnet18


def easy_confusion_matrix(y_true, y_pred, show=True, save_path=None):
    """
        绘制混淆矩阵。

        :param y_true : list
            真实标签。
        :param y_pred : list
            预测标签。
        :param show : bool, optional
            是否在函数执行后显示图表。
        :param save_path : str or None, optional
            如果提供，将图表保存到指定路径。
    """

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    seaborn.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')

    if save_path:
        plt.savefig(save_path)

    if show:
        plt.show()


def easy_gradcam(model, images, target_layers=None, input_tensor=None):
    """
    This function is used to generate Class Activation Maps (CAM). CAM is a visualization technique used to understand
    the decision-making process of CNN.

    :param model: Model object, if not provided, the default will be the ResNet18 model.
    :param target_layers: Target layer, if not provided, the default will be the fourth layer of the ResNet18 model.
    :param images: Images that need to be visualized with CAM, the shape should be [b, h, w, c],
                    where b is the batch size, h is the height, w is the width, and c is the number of channels.
                    The value of the image should be in the range [0, 1].
    :param input_tensor: If not provided, images will be converted to a floating point tensor.
    :return: A list containing all original images, visual images, and grayscale CAM.
    """

    if model is None:
        model = resnet18()

    if target_layers is None:
        target_layers = [model.layer4]

    if isinstance(images, list):
        images = torch.stack(images, dim=0)

    if input_tensor is None:
        input_tensor = torch.tensor(images, dtype=torch.float)

    if images.shape[1] == 1 or images.shape[1] == 3:
        # images = einops.rearrange(images, "b c h w -> b h w c")
        images = images.permute(0, 2, 3, 1)
        images = images.to("cpu").numpy()

    # pip install grad-cam
    # Construct the CAM object once, and then re-use it on many images:
    cam = GradCAM(model=model, target_layers=target_layers)

    # You can also use it within a with statement, to make sure it is freed,
    # In case you need to re-create it inside an outer loop:
    # with GradCAM(model=model, target_layers=target_layers, use_cuda=args.use_cuda) as cam:
    #   ...

    # We have to specify the target we want to generate
    # the Class Activation Maps for.
    # If targets is None, the highest scoring category
    # will be used for every image in the batch.
    # Here we use ClassifierOutputTarget, but you can define your own custom targets
    # That are, for example, combinations of categories, or specific outputs in a non standard model.

    # targets = [ClassifierOutputTarget(11),ClassifierOutputTarget(0)]
    targets = None

    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    # print(input_tensor)
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

    # In this example grayscale_cam has only one image in the batch:
    out_images = []
    for i in range(len(images)):
        # print(images.shape)
        # print(grayscale_cam.shape)
        visualization = show_cam_on_image(images[i], grayscale_cam[i], use_rgb=True)
        out_images.extend([images[i], visualization, grayscale_cam[i]])

    return out_images


def easy_tsne(data, labels, oids=None, point_mode: Literal["point", "digital", "id", "both"] = "both", show=True,
              save_path=None, font_size=5):
    """
        使用 t-SNE 算法对数据进行降维并可视化。

        :param data :
            待降维的数据，形状最终会转化为 (n_samples, n_features)。
        :param labels :
            与数据相对应的标签，形状为 (n_samples,)。
        :param oids : list or None, optional
            对象的标识符列表，仅当 point_mode 为 "id" 时需要提供。
        :param point_mode : Literal["point", "digital", "id"], optional
            点的显示模式。"id" 模式在点旁显示 oids，"digital" 显示数字标签，"point" 仅显示点。
        :param show : bool, optional
            是否在函数执行后显示图表。
        :param save_path : str or None, optional
            如果提供，将图表保存到指定路径。
        :param font_size : int, optional

        :return: None

        :raise ValueError
            如果 point_mode 为 "id" 且没有提供 oids。
    """
    if point_mode in ["id", "both"] and not oids:
        raise ValueError("If point_mode is 'id' or 'both', oids must be provided.")

    if oids:
        oids = numpy.array(oids)

    labels_type_count = len(set(labels))
    data_size = len(data)
    # 将数据展平 (10000, 28*28)
    data = data.view(data_size, -1)

    # 初始化 t-SNE 并将数据降维到 2D
    tsne = TSNE(n_components=2, random_state=42, perplexity=labels_type_count)
    images_tsne = tsne.fit_transform(data)

    plt.figure(figsize=(8, 8))
    # 绘制 t-SNE 结果
    if point_mode == "id":
        scatter = plt.scatter(images_tsne[:, 0], images_tsne[:, 1], c=labels, cmap='tab10', alpha=0.7, s=0)

        for i in range(data_size):
            plt.text(images_tsne[i, 0], images_tsne[i, 1], str(oids[i].item()),
                     color=plt.cm.tab10(labels[i].item() / labels_type_count),
                     fontdict={'weight': 'bold', 'size': font_size})
    elif point_mode == "digital":
        scatter = plt.scatter(images_tsne[:, 0], images_tsne[:, 1], c=labels, cmap='tab10', alpha=0.7, s=0)

        for i in range(data_size):
            plt.text(images_tsne[i, 0], images_tsne[i, 1], str(labels[i].item()),
                     color=plt.cm.tab10(labels[i].item() / labels_type_count),
                     fontdict={'weight': 'bold', 'size': font_size})
    elif point_mode == "both":
        scatter = plt.scatter(images_tsne[:, 0], images_tsne[:, 1], c=labels, cmap='tab10', alpha=0.7, s=0)

        for i in range(data_size):
            plt.text(images_tsne[i, 0], images_tsne[i, 1], f"{oids[i].item()}({labels[i].item()})",
                     color=plt.cm.tab10(labels[i].item() / labels_type_count),
                     fontdict={'weight': 'bold', 'size': font_size})
    else:
        scatter = plt.scatter(images_tsne[:, 0], images_tsne[:, 1], c=labels, cmap='tab10', alpha=0.7)

    # 添加图例
    legend1 = plt.legend(*scatter.legend_elements(), title="Classes")
    plt.gca().add_artist(legend1)

    # 添加标题和标签
    plt.title("t-SNE Visualization")

    if save_path:
        plt.savefig(save_path)
        plt.close()

    if show:
        # 显示图像
        plt.show()


def show_images(*images: List[numpy.array], subtitles=None, col_num: int = 3, title: str = None,
                each_size=(8, 7), show: bool = True, save_path: str = None):
    """
    Display or save a collection of images with optional subtitles.

    :param images: List of images to be displayed or saved.
    :param subtitles: List of subtitles for the images. If the number of subtitles is less than the number of images, None will be appended to the subtitles list.
    :param col_num: Number of columns for the image grid. Default is 3.
    :param title: Title for the plot. Default is None.
    :param each_size: Size for each subplot in the grid. Default is (8, 7).
    :param show: Boolean indicating whether to display the plot. Default is True.
    :param save_path: Path to save the plot. If both show and save_path are None, a ValueError is raised. Default is None.
    :return: None
    """
    if not show and save_path is None:
        raise ValueError("Must specify an output method, either show or save to a file.")

    images_count = len(images)
    col_num = min(col_num, images_count)

    each_size = (each_size, each_size * 0.875) if isinstance(each_size, int) else each_size

    subtitles = [None] * images_count if subtitles is None else subtitles + [None] * (images_count - len(subtitles))

    row_num = (images_count + col_num - 1) // col_num
    plt.subplots_adjust(top=0.85)  # 调整顶部间距
    fig, axs = plt.subplots(nrows=row_num, ncols=col_num, squeeze=False)
    if title is not None:
        fig.suptitle(title, fontsize=each_size[0] * 4)

    fig.set_size_inches(each_size[0] * col_num, each_size[1] * row_num)

    for ax, image, subtitle in zip(axs.ravel(), images, subtitles):
        ax.set_title(subtitle, fontsize=each_size[0] * 3)
        ax.imshow(image, cmap='gray' if len(image.shape) == 2 else None)
        ax.axis('off')

    for ax in axs.ravel()[images_count:]:
        ax.axis('off')

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path)

    if show:
        plt.show()


def easy_roc_curve(target_list, predict_list, curve_name_list=None, show=True, save_path=None):
    """
    绘制 ROC 曲线。

    :param target_list : list
        真实标签的列表，每个元素代表一个模型的真实标签序列。
    :param predict_list : list
        预测标签的列表，每个元素代表一个模型的预测序列。
    :param curve_name_list: list or None, optional
        不同曲线的名称, 默认为 None。
    :param show: bool, optional
        是否在函数执行后显示图表。
    :param save_path : str or None, optional
        如果提供，将图表保存到指定路径。
    """
    plt.figure()

    if curve_name_list is None:
        curve_name_list = range(len(target_list))

    for i in range(len(target_list)):
        target = target_list[i]
        predict = predict_list[i]
        fpr, tpr, thresholds = roc_curve(target, predict)
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, lw=2, label=f'{curve_name_list[i]} (area = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # 对角线代表随机猜测
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")

    if save_path:
        plt.savefig(save_path)

    if show:
        plt.show()
