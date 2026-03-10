# import matplotlib.pyplot as plt
# from PIL import Image


# def plot_topk(image_paths, scores):

#     cols = 5
#     k = len(image_paths)
#     rows = (k + cols - 1) // cols

#     plt.figure(figsize=(4*cols,4*rows))

#     for i,(path,score) in enumerate(zip(image_paths,scores)):

#         img = Image.open(path).convert("RGB")

#         plt.subplot(rows,cols,i+1)
#         plt.imshow(img)
#         plt.title(f"{score:.3f}")
#         plt.axis("off")

#     plt.tight_layout()

#     save_path = "topk_result.png"
#     plt.savefig(save_path)

#     print("Saved visualization:", save_path)

import matplotlib.pyplot as plt
from PIL import Image

def plot_topk(image_paths, scores):
    cols = 5
    k = len(image_paths)
    rows = (k + cols - 1) // cols

    plt.figure(figsize=(4*cols, 4*rows))

    for i, (path, score) in enumerate(zip(image_paths, scores)):
        img = Image.open(path).convert("RGB")
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img)
        plt.title(f"{score:.3f}")
        plt.axis("off")

    plt.tight_layout()

    # --- ĐOẠN THAY ĐỔI Ở ĐÂY ---
    plt.show()  # Lệnh này để in ảnh ra màn hình Kaggle
    
    # Nếu bạn vẫn muốn lưu file để tải về sau thì giữ dòng này, 
    # nhưng nhớ gọi nó TRƯỚC plt.show() nếu bị lỗi hình trắng.
    # plt.savefig("topk_result.png")