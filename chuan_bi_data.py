import kagglehub
path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")

import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Đường dẫn đến bộ dữ liệu đã tải xuống
dataset_path = path

print(f"Đường dẫn bộ dữ liệu: {dataset_path}")

# Liệt kê nội dung của thư mục dataset
print("Nội dung trong thư mục bộ dữ liệu:")
for dirname, _, filenames in os.walk(dataset_path):
    for filename in filenames:
        print(os.path.join(dirname, filename))
    if not filenames and not _:
        print(dirname)
