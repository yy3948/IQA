import cv2
import os
import numpy as np
from skimage import feature
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score
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

    return list(s_moments) + list(v_moments)


# 函数：梯度局部二值法
def gradient_local_binary_pattern(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 计算梯度
    gradient_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=5)
    gradient_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=5)
    gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
    gradient_magnitude = np.uint8(255 * gradient_magnitude / np.max(gradient_magnitude))

    # 计算局部二值模式
    lbp = feature.local_binary_pattern(gradient_magnitude, P=8, R=1, method="uniform")

    return lbp


# 加载图像数据集
good_quality_folder = "./tid2013/reference_images"
bad_quality_folder = "./tid2013/distorted_images"

# 读取图像并提取特征
x = []
y = []

# 处理好质量图像
for filename in os.listdir(good_quality_folder):
    image = cv2.imread(os.path.join(good_quality_folder, filename))
    # 计算颜色矩
    color_moments = calculate_color_moments(image)
    # 计算梯度局部二值模式
    lbp = gradient_local_binary_pattern(image)
    # 将特征合并
    features = np.concatenate([color_moments, lbp.flatten()])
    # print(features)
    # print(type(x))
    x.append(features)
    # print(type(x))
    y.append(1)  # 1 表示好质量图像

# 处理坏质量图像
for filename in os.listdir(bad_quality_folder):
    image = cv2.imread(os.path.join(bad_quality_folder, filename))
    # 计算颜色矩
    color_moments = calculate_color_moments(image)
    # 计算梯度局部二值模式
    lbp = gradient_local_binary_pattern(image)
    # 将特征合并
    features = np.concatenate([color_moments, lbp.flatten()])
    x.append(features)
    y.append(0)  # 0 表示坏质量图像

# 将数据集划分为训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 训练SVM模型
svm_model = SVC()
svm_model.fit(x_train, y_train)

# 在测试集上评估模型
y_pred = svm_model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# 分类报告
print("Classification Report:")
print(classification_report(y_test, y_pred, zero_division=1))

# 混淆矩阵
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# ROC 曲线和 AUC
fpr, tpr, thresholds = roc_curve(y_test, svm_model.decision_function(x_test))
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
