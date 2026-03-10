import matplotlib.pyplot as plt
from PIL import Image


def plot_topk(image_paths, scores):

    cols = 5
    k = len(image_paths)
    rows = (k + cols - 1) // cols

    plt.figure(figsize=(4*cols,4*rows))

    for i,(path,score) in enumerate(zip(image_paths,scores)):

        img = Image.open(path).convert("RGB")

        plt.subplot(rows,cols,i+1)
        plt.imshow(img)
        plt.title(f"{score:.3f}")
        plt.axis("off")

    plt.tight_layout()

    save_path = "topk_result.png"
    plt.savefig(save_path)

    print("Saved visualization:", save_path)

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def plot_topk_with_bbox(image_paths, boxes, scores):

    cols = 5
    k = len(image_paths)
    rows = (k + cols - 1) // cols

    fig, ax = plt.subplots(rows, cols, figsize=(4*cols,4*rows))

    if rows == 1:
        ax = [ax]

    for i,(path,box,score) in enumerate(zip(image_paths,boxes,scores)):

        img = Image.open(path).convert("RGB")

        r = i // cols
        c = i % cols

        ax[r][c].imshow(img)

        x1,y1,x2,y2 = box

        rect = patches.Rectangle(
            (x1,y1),
            x2-x1,
            y2-y1,
            linewidth=3,
            edgecolor='red',
            facecolor='none'
        )

        ax[r][c].add_patch(rect)

        ax[r][c].set_title(f"{score:.3f}")
        ax[r][c].axis("off")

    plt.tight_layout()
    plt.savefig("retrieval_bbox.png")

    print("Saved → retrieval_bbox.png")