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
from IPython.display import display

# Đảm bảo hiển thị inline
%matplotlib inline

def plot_topk(image_paths, scores):
    cols = 5
    k = len(image_paths)
    rows = (k + cols - 1) // cols

    # Khởi tạo figure
    fig = plt.figure(figsize=(4*cols, 4*rows))

    for i, (path, score) in enumerate(zip(image_paths, scores)):
        try:
            img = Image.open(path).convert("RGB")
            plt.subplot(rows, cols, i + 1)
            plt.imshow(img)
            plt.title(f"{score:.4f}", fontsize=12) # Tăng font cho dễ nhìn
            plt.axis("off")
        except Exception as e:
            print(f"Lỗi không mở được ảnh: {path} - {e}")

    plt.tight_layout()
    
    # Lưu và hiển thị
    plt.savefig("topk_result.png")
    plt.show() # Hoặc display(fig)

# Gọi hàm với dữ liệu của bạn
# plot_topk(image_paths, scores)