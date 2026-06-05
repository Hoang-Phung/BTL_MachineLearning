# Hàm chuẩn hóa hình ảnh
def preprocess(image, label):
    image = tf.image.convert_image_dtype(image, tf.float32) # Chuyển sang float32 và chuẩn hóa về [0, 1]
    return image, label

# Áp dụng chuẩn hóa và cấu hình prefetching
train_ds = train_ds.map(preprocess).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.map(preprocess).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.map(preprocess).cache().prefetch(buffer_size=tf.data.AUTOTUNE)

# Kiểm tra một batch dữ liệu sau khi chuẩn hóa
for images, labels in train_ds.take(1):
    print(f"Phạm vi giá trị pixel sau khi chuẩn hóa: {np.min(images[0])} - {np.max(images[0])}")
    break
