from typing import List

import matplotlib.pyplot as plt
import numpy


def easy_gradcam(model, target_layers, images, input_tensor=None):
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
    import torch
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from torchvision.models import resnet18

    if model is None:
        model = resnet18()

    if target_layers is None:
        target_layers = [model.layer4]

    if isinstance(images, list):
        images = torch.stack(images, dim=0)

    if input_tensor is None:
        input_tensor = torch.tensor(images, dtype=torch.float)

    if images.shape[1] == 1:
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


def easy_tsne(data, labels, perplexity=30, n_iter=1000, random_state=0, dot_type: str = "scatter", figsize=(8, 8),
              fontsize=15, cmap="rainbow", show=True):
    """
    This function is used to visualize high-dimensional data in a low-dimensional space. The t-SNE algorithm is used to
    reduce the dimensionality of the data and visualize it in a two-dimensional space.


    :param data: High dimensional data that requires t-SNE visualization. if the data is a multi-dimensional array,
                it will be reshaped to a two-dimensional array.
    :param labels: Labels of the data, the shape should be [n].
    :param perplexity: The perplexity is related to the number of nearest neighbors that is used in other manifold
                        learning algorithms. Larger datasets usually require a larger perplexity. Consider selecting a
                        value between 5 and 50.
    :param n_iter: Maximum number of iterations for the optimization. Should be at least 250.
    :param random_state: Determines the random number generation for the optimization. Use an int to make the randomness
                        deterministic.
    :param dot_type: The type of visualization, "scatter" or "figure".
    :param figsize: The size of the matplotlib figure, the default is (10, 10).
    :param show: Whether to display the visualization, the default is True.
    :param fontsize: The font size of the label, the default is 15.
    :param cmap: The color map of the visualization, the default is "rainbow".
    :return: A list containing the original data and the visualization data.
    """
    import io
    import numpy
    from PIL import Image
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    if isinstance(figsize, int):
        figsize = (figsize, figsize)

    data = data.reshape((data.shape[0], -1))
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=random_state)
    tsne_results = tsne.fit_transform(data)

    plt.figure(figsize=figsize)

    if dot_type == "scatter":
        plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap=cmap)
    else:
        plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c="white")

        color_map = plt.get_cmap(cmap, len(set(labels)))  # 创建颜色映射，设置为10类

        for i in range(tsne_results.shape[0]):
            plt.text(tsne_results[i, 0], tsne_results[i, 1], str(labels[i]), c=color_map(labels[i]), fontsize=fontsize)

    plt.tight_layout()

    if show:
        plt.show()
        plt.close()
    else:
        image = io.BytesIO()

        plt.savefig(image, format='png')
        plt.close()

        image.seek(0)
        image = Image.open(image)

        return numpy.array(image)


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
