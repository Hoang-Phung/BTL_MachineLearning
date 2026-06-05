import numpy as np
import matplotlib.pyplot as plt
from du_doan_truc_tuyen import plot_sample_predictions

loss, accuracy = model.evaluate(test_ds)
print(f"Độ chính xác trên tập kiểm tra: {accuracy*100:.2f}%")
print(f"Hàm mất mát trên tập kiểm tra: {loss:.4f}")

# Vẽ biểu đồ độ chính xác và hàm mất mát trong quá trình huấn luyện
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Độ chính xác huấn luyện')
plt.plot(epochs_range, val_acc, label='Độ chính xác Validation')
plt.legend(loc='lower right')
plt.title('Độ chính xác huấn luyện và Validation')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Hàm mất mát huấn luyện')
plt.plot(epochs_range, val_loss, label='Hàm mất mát Validation')
plt.legend(loc='upper right')
plt.title('Hàm mất mát huấn luyện và Validation')
plt.show()

# Hiển thị dự đoán mẫu trên tập kiểm tra
plot_sample_predictions(model, test_ds, num_images=9)
