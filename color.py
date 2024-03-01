
import cv2
import numpy as np
import matplotlib.pyplot as plt


# 读取图像并转换为CIE XYZ色彩空间
def read_and_convert_to_xyz(image_path):
    # 读取图像
    image = cv2.imread(image_path)
    # 将图像转换为CIE XYZ色彩空间
    xyz_image = cv2.cvtColor(image, cv2.COLOR_BGR2XYZ)
    return xyz_image


# 计算色彩特征的直方图
def compute_color_histogram(xyz_image):
    # 将图像展平为一维数组
    flattened_image = xyz_image.reshape(-1, 3)
    # 计算直方图
    hist, bins = np.histogram(flattened_image, bins=256, range=(0, 255))
    return hist, bins


# 绘制直方图
def plot_histogram(hist, bins):
    plt.bar(bins[:-1], hist, width=np.diff(bins), color='blue')
    plt.xlim([0, 256])
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.title('Color Feature Histogram')


# 保存图像
def save_image(image, filename):
    cv2.imwrite(filename, image)


# 读取四张原始图像并进行处理
image_paths = ["./lbp11/pic1.BMP", "./lbp11/pic2.bmp", "./lbp11/pic3.bmp", "./lbp11/pic4.bmp"]
xyz_images = []
color_histograms = []

for image_path in image_paths:
    # 读取并转换为CIE XYZ色彩空间
    xyz_image = read_and_convert_to_xyz(image_path)
    xyz_images.append(xyz_image)

    # 计算色彩特征的直方图
    hist, bins = compute_color_histogram(xyz_image)
    color_histograms.append(hist)

# 绘制原始图像、CIE XYZ图像和色彩特征直方图，并保存
plt.figure(figsize=(12, 9))

for i in range(4):
    # 绘制原始图像
    plt.subplot(3, 4, i + 1)
    image = cv2.imread(image_paths[i])
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    # plt.title(f'Original Image {i + 1}')

    # 绘制CIE XYZ图像
    plt.subplot(3, 4, 5 + i)
    plt.imshow(cv2.cvtColor(xyz_images[i], cv2.COLOR_BGR2RGB))
    plt.axis('off')
    # plt.title(f'XYZ Image {i + 1}')

    # 绘制色彩特征直方图
    plt.subplot(3, 4, 9 + i)
    plot_histogram(color_histograms[i], np.arange(257))
    # plt.title(f'Color Histogram {i + 1}')

plt.tight_layout()
plt.savefig('combined_images_histograms.png')
plt.show()
