import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import display


def plot_topk(image_paths, scores):

    cols = 5
    k = len(image_paths)
    rows = (k + cols - 1) // cols

    fig = plt.figure(figsize=(4*cols,4*rows))

    for i,(path,score) in enumerate(zip(image_paths,scores)):

        img = Image.open(path).convert("RGB")

        ax = fig.add_subplot(rows, cols, i+1)
        ax.imshow(img)
        ax.set_title(f"{score:.3f}")
        ax.axis("off")

    display(fig)

    plt.show()