
import cv2
import os
import numpy as np
from skimage import feature
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import StandardScaler
import scipy.stats
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
    # image = cv2.imread(os.path.join(good_quality_folder, filename))
    image_path = os.path.join(good_quality_folder, filename)
    image = augment_image(image_path)
    # 计算颜色矩
    color_moments = calculate_color_moments(image)
    # 计算梯度局部二值模式
    lbp = gradient_local_binary_pattern(image)
    # 将特征合并
    features = np.concatenate([lbp.flatten(), color_moments])
    x.append(features)
    # print(type(x))
    y.append(1)  # 1 表示好质量图像

# 处理坏质量图像
for filename in os.listdir(bad_quality_folder):
    # image = cv2.imread(os.path.join(bad_quality_folder, filename))
    image_path = os.path.join(bad_quality_folder, filename)
    image = augment_image(image_path)
    # 计算颜色矩
    color_moments = calculate_color_moments(image)
    # 计算梯度局部二值模式
    lbp = gradient_local_binary_pattern(image)
    # 将特征合并
    features = np.concatenate([lbp.flatten(), color_moments])
    x.append(features)
    y.append(0)  # 0 表示坏质量图像

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

# 评估模型
mse = mean_squared_error(y_test, y_pred_svr)
print(f"Mean Squared Error (MSE): {mse}")

plcc, _ = scipy.stats.pearsonr(y_test, y_pred_svr)
srocc, _ = scipy.stats.spearmanr(y_test, y_pred_svr)
krocc, _ = scipy.stats.kendalltau(y_test, y_pred_svr)

print(f"PLCC: {plcc}")
print(f"SROCC: {srocc}")
print(f"KROCC: {krocc}")

