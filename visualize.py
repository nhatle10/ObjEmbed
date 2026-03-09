from PIL import Image
from IPython.display import display


def plot_topk(image_paths, scores):

    for path,score in zip(image_paths,scores):

        print(f"Score: {score:.3f}")
        display(Image.open(path))