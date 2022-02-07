import matplotlib.pyplot as plt


def trim_axs(axs, N):
    """
    Reduce *axs* to *N* Axes. All further Axes are removed from the figure.
    """
    axs = axs.flat
    for ax in axs[N:]:
        ax.remove()
    return axs[:N]


def show_imgs(imgs, cols=10):
    n = len(imgs)
    rows = (n // cols) + 1
    figure, axs = plt.subplots(rows, cols, figsize=(20, 20))
    axs = trim_axs(axs, n)
    for i, (ax, img) in enumerate(zip(axs, imgs)):
        ax.imshow(img, cmap='gray')
        ax.set_title(f'{i}')
        ax.set_axis_off()

    plt.show()


def show_pred_mask(image, label, pred):
    fig, axis = plt.subplots(1, 2, figsize=(10, 10))
    axis[0].set_title('prediction')
    axis[0].imshow(image, cmap='gray')
    axis[0].imshow(pred, cmap='viridis', alpha=0.3)
    axis[1].set_title('label')
    axis[1].imshow(image, cmap='gray')
    axis[1].imshow(label, cmap='viridis', alpha=0.3)
    plt.show()
