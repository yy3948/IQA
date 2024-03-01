
import cv2
import os
import numpy as np
from skimage import feature
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import StandardScaler
import scipy.stats
from scipy.special import expit
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt

# 函数：计算颜色矩
def calculate_color_moments(image):
    # 将彩色图像转为HSV模型
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # 提取S和V通道
    s_channel = hsv_image[:, :, 1]
    v_channel = hsv_image[:, :, 2]

    # 计算颜色矩
    s_moments = cv2.moments(s_channel).values()
    v_moments = cv2.moments(v_channel).values()

    # 添加额外的颜色矩，第三个颜色矩是偏度，第四个颜色矩是峰度
    s_skewness = cv2.moments(s_channel)['mu03'] if cv2.moments(s_channel)['m00'] != 0 else 0
    v_skewness = cv2.moments(v_channel)['mu03'] if cv2.moments(v_channel)['m00'] != 0 else 0
    s_kurtosis = cv2.moments(s_channel)['mu02'] if cv2.moments(s_channel)['m00'] != 0 else 0
    v_kurtosis = cv2.moments(v_channel)['mu02'] if cv2.moments(v_channel)['m00'] != 0 else 0

    return list(s_moments) + list(v_moments) + [s_skewness, v_skewness, s_kurtosis, v_kurtosis]

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
def compute_color_feature(inpuy_xyz_image, input_x, input_y, input_z):
    output_color_feature = np.sum(inpuy_xyz_image[:, :, 0] * input_x + inpuy_xyz_image[:, :, 1] * input_y + inpuy_xyz_image[:, :, 2] * input_z)
    return output_color_feature

# 函数：梯度局部二值法
def gradient_local_binary_pattern(input_image):
    gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    # 计算梯度
    gradient_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=5)
    gradient_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=5)
    gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
    gradient_magnitude = np.uint8(255 * gradient_magnitude / np.max(gradient_magnitude))

    # 计算局部二值模式
    lbp = feature.local_binary_pattern(gradient_magnitude, P=8, R=1, method="uniform")

    return lbp


# 数据增强函数
def augment_image(image_path):
    # 使用 OpenCV 读取图像
    aug_img = cv2.imread(image_path)

    # OpenCV 图像增强：旋转和翻转
    rows, cols, _ = aug_img .shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 30, 1)
    aug_img = cv2.warpAffine(aug_img, M, (cols, rows))
    aug_img = cv2.flip(aug_img, 1)

    # 使用 Pillow 进行图像增强：调整亮度和对比度
    pil_img = Image.fromarray(cv2.cvtColor(aug_img, cv2.COLOR_BGR2RGB))
    enhancer = ImageEnhance.Brightness(pil_img)
    aug_img = np.array(enhancer.enhance(1.5))  # 增加亮度
    enhancer = ImageEnhance.Contrast(Image.fromarray(aug_img ))
    aug_img = np.array(enhancer.enhance(1.5))  # 增加对比度

    return aug_img


# 加载图像数据集
good_quality_folder = "./tid2013/reference_images_more"
bad_quality_folder = "./tid2013/distorted_images_less"

# 读取图像并提取特征
x = []
y = []

# 处理好质量图像
for filename in os.listdir(good_quality_folder):
    good_image = cv2.imread(os.path.join(good_quality_folder, filename))
    # image_path = os.path.join(good_quality_folder, filename)
    # good_image = augment_image(image_path)
    # 计算梯度局部二值模式
    lbp = gradient_local_binary_pattern(good_image)
    # 计算颜色矩
    # color_moments = calculate_color_moments(image)
    good_rgb_image = cv2.cvtColor(good_image, cv2.COLOR_BGR2RGB)
    good_xyz_image = rgb_to_xyz(good_rgb_image)
    # 提取彩色信息的特征
    color_features = []
    lambda_range = np.arange(380, 781, 4)
    for lambda_val in lambda_range:
        good_x_val, good_y_val, good_z_val = color_matching_function(lambda_val)
        if good_x_val is not None and good_y_val is not None and good_z_val is not None:
            color_feature = compute_color_feature(good_xyz_image, good_x_val, good_y_val, good_z_val)
            color_features.append(color_feature)
        else:
            print("error")
            break
    # 将特征合并
    if len(lbp.flatten()) > 0 and len(color_features) > 0:
        features = np.concatenate([lbp.flatten(), color_features])
        x.append(features)
        # print(type(x))
        y.append(1)  # 1 表示好质量图像
    else:
        print(f"Skipped adding features for image {filename} due to empty features or color features.")

# 处理坏质量图像
for filename in os.listdir(bad_quality_folder):
    bad_image = cv2.imread(os.path.join(bad_quality_folder, filename))
    # image_path = os.path.join(bad_quality_folder, filename)
    # bad_image = augment_image(image_path)
    # 计算梯度局部二值模式
    lbp = gradient_local_binary_pattern(bad_image)
    # 计算颜色矩
    # color_moments = calculate_color_moments(bad_image)
    bad_rgb_image = cv2.cvtColor(bad_image, cv2.COLOR_BGR2RGB)
    bad_xyz_image = rgb_to_xyz(bad_rgb_image)
    # hist_features = compute_histogram(bad_image)

    # 提取彩色信息的特征
    color_features = []
    lambda_range = np.arange(380, 781, 4)
    for lambda_val in lambda_range:
        bad_x_val, bad_y_val, bad_z_val = color_matching_function(lambda_val)
        if bad_x_val is not None and bad_y_val is not None and bad_z_val is not None:
            color_feature = compute_color_feature(bad_xyz_image, bad_x_val, bad_y_val, bad_z_val)
            color_features.append(color_feature)
        else:
            print("error")
            break
    # 将特征合并
    if len(lbp.flatten()) > 0 and len(color_features) > 0:
        features = np.concatenate([lbp.flatten(), color_features])
        x.append(features)
        y.append(0)  # 0 表示坏质量图像
    else:
        print(f"Skipped adding features for image {filename} due to empty features or color features.")

# 将数据集划分为训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 数据预处理：标准化数据
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# 训练SVR模型
svr_model = SVR(kernel='rbf', C=100, epsilon=0.001)  # 调整epsilon和C参数
svr_model.fit(x_train_scaled, y_train)

# 在测试集上预测
y_pred_svr = svr_model.predict(x_test_scaled)
# y_pred = svr_model.predict(x_test)
# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", accuracy)

# 评估模型
mse = mean_squared_error(y_test, y_pred_svr)
print(f"Mean Squared Error (MSE): {mse}")

plcc, _ = scipy.stats.pearsonr(y_test, y_pred_svr)
srocc, _ = scipy.stats.spearmanr(y_test, y_pred_svr)
krocc, _ = scipy.stats.kendalltau(y_test, y_pred_svr)

print(f"PLCC: {plcc}")
print(f"SROCC: {srocc}")
print(f"KROCC: {krocc}")

