# Định nghĩa các tham số
IMG_WIDTH = 128
IMG_HEIGHT = 128
BATCH_SIZE = 32

# Tạo đường dẫn cho các bộ dữ liệu con, sửa lại để chỉ đúng thư mục cấp dưới 'chest_xray'
train_dir = os.path.join(dataset_path, 'chest_xray', 'chest_xray', 'train')
val_dir = os.path.join(dataset_path, 'chest_xray', 'chest_xray', 'val')
test_dir = os.path.join(dataset_path, 'chest_xray', 'chest_xray', 'test')

# Tải dữ liệu huấn luyện
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    labels='inferred',
    label_mode='binary',
    image_size=(IMG_WIDTH, IMG_HEIGHT),
    interpolation='nearest',
    batch_size=BATCH_SIZE,
    shuffle=True
)

# Tải dữ liệu validation
val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    labels='inferred',
    label_mode='binary',
    image_size=(IMG_WIDTH, IMG_HEIGHT),
    interpolation='nearest',
    batch_size=BATCH_SIZE,
    shuffle=False
)

# Tải dữ liệu kiểm tra
test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    labels='inferred',
    label_mode='binary',
    image_size=(IMG_WIDTH, IMG_HEIGHT),
    interpolation='nearest',
    batch_size=BATCH_SIZE,
    shuffle=False
)

# Kiểm tra một batch dữ liệu và nhãn
for images, labels in train_ds.take(1):
    print(f"Dạng của một batch hình ảnh: {images.shape}") # (batch_size, IMG_HEIGHT, IMG_WIDTH, 3)
    print(f"Dạng của một batch nhãn: {labels.shape}")    # (batch_size,)
    print(f"Phạm vi giá trị pixel trước khi chuẩn hóa: {np.min(images[0])} - {np.max(images[0])}")
    break
