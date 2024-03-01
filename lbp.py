import cv2
import numpy as np
from skimage import feature
import matplotlib.pyplot as plt

# 读取原始图像
def read_image(image_path):
    return cv2.imread(image_path)

# 灰度处理
def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 计算梯度
def gradient(image):
    gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
    gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
    return np.uint8(255 * gradient_magnitude / np.max(gradient_magnitude))

# 计算梯度直方图
def gradient_histogram(gradient_image):
    hist, _ = np.histogram(gradient_image.flatten(), bins=256, range=(0,255))
    return hist

# 绘制直方图
def plot_histogram(hist):
    plt.bar(np.arange(len(hist)), hist, color='blue')
    plt.xlim([0, 256])
    plt.ylim([0, 4000])
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.title('Gradient Histogram')

# 保存图像
def save_image(image, filename):
    cv2.imwrite(filename, image)

# 读取原始图像
original_images = []
original_images.append(read_image("./lbp11/pic1.BMP"))
original_images.append(read_image("./lbp11/pic2.bmp"))
original_images.append(read_image("./lbp11/pic3.bmp"))
original_images.append(read_image("./lbp11/pic4.bmp"))

# 循环处理每张图像
for i, image in enumerate(original_images):
    # 灰度处理
    gray_image = grayscale(image)
    # 计算梯度
    gradient_image = gradient(gray_image)
    # 计算梯度直方图
    hist = gradient_histogram(gradient_image)
    # 保存灰度图
    save_image(gray_image, f"gray_image_{i + 1}.jpg")
    # 保存梯度图
    save_image(gradient_image, f"gradient_image_{i + 1}.jpg")
    # 保存直方图
    plt.figure()
    plot_histogram(hist)
    plt.tight_layout()
    plt.savefig(f"histogram_{i + 1}.jpg")
