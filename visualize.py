import matplotlib.pyplot as plt
from PIL import Image


def plot_topk(image_paths, scores, cols=5):

    k = len(image_paths)
    rows = (k + cols - 1) // cols

    plt.figure(figsize=(4*cols, 4*rows))

    for i, (img_path, score) in enumerate(zip(image_paths, scores)):

        img = Image.open(img_path).convert("RGB")

        plt.subplot(rows, cols, i+1)
        plt.imshow(img)
        plt.title(f"{score:.3f}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()