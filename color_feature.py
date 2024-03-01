import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit


# 定义颜色匹配函数
def color_matching_function(input_lambda_val):
    try:
        x = 0.4002 * expit((731.616 - input_lambda_val) / (4.029e-5 * input_lambda_val))
        z = 0.2545 * expit((632.816 - input_lambda_val) / (2.907e-3 * input_lambda_val))
        y = 1 - x - z
        return x, y, z
    except OverflowError:
        return None, 0.33, None


# 定义RGB到CIEXYZ的转换函数
def rgb_to_xyz(rgb):
    M = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    # 将rgb重塑为形状为(3, 384*512)的数组
    rgb_reshaped = rgb.transpose(2, 0, 1).reshape(3, -1)
    xyz = np.dot(M, rgb_reshaped)
    output_xyz_image = xyz.reshape(rgb.shape)
    return output_xyz_image


# 定义计算彩色信息特征的函数
def compute_color_feature(input_xyz_image, input_x, input_y, input_z):
    output_color_feature = np.sum(input_xyz_image[:, :, 0] * input_x +
                                  input_xyz_image[:, :, 1] * input_y +
                                  input_xyz_image[:, :, 2] * input_z)
    return output_color_feature


# 读取图像并转换为CIE XYZ色彩空间
def read_and_convert_to_xyz(image_path):
    image = cv2.imread(image_path)
    xyz_image = cv2.cvtColor(image, cv2.COLOR_BGR2XYZ)
    return xyz_image


# 计算颜色特征的直方图
def compute_color_histogram(xyz_image, x_vals, y_vals, z_vals):
    color_histogram = []
    for i in range(len(x_vals)):
        color_feature = compute_color_feature(xyz_image, x_vals[i], y_vals[i], z_vals[i])
        color_histogram.append(color_feature)
    return color_histogram


# 绘制直方图
def plot_histogram(color_histogram, image_name):
    plt.bar(range(len(color_histogram)), color_histogram, color='blue')
    plt.xlabel('Bins')
    plt.ylabel('Color Feature')
    plt.title(f'Color Feature Histogram for {image_name}')
    plt.show()


# 读取四张原始图像
image_paths = ["./lbp1/pic1.BMP", "./lbp1/pic2.bmp", "./lbp1/pic3.bmp", "./lbp1/pic4.bmp"]

# 逐张图像处理
for image_path in image_paths:
    # 读取并转换为CIE XYZ色彩空间
    xyz_image = read_and_convert_to_xyz(image_path)

    # 计算颜色匹配函数
    lambda_range = np.arange(380, 781, 4)
    x_vals, y_vals, z_vals = [], [], []
    for lambda_val in lambda_range:
        x_val, y_val, z_val = color_matching_function(lambda_val)
        x_vals.append(x_val)
        y_vals.append(y_val)
        z_vals.append(z_val)

    # 计算色彩特征的直方图
    color_histogram = compute_color_histogram(xyz_image, x_vals, y_vals, z_vals)

    # 绘制直方图
    plot_histogram(color_histogram, image_path)
