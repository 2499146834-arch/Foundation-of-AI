import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from data_loader import get_data_generators   # 导入成员A的函数

# 1. 获取数据生成器
train_generator, validation_generator = get_data_generators(
    train_dir='data/train',
    val_dir='data/validation',
    target_size=(150, 150),
    batch_size=32
)

print("训练样本数:", train_generator.samples)
print("验证样本数:", validation_generator.samples)
print("类别映射:", train_generator.class_indices)  # 应该是 {'cats': 0, 'dogs': 1}

# 2. 构建CNN模型
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(1, activation='sigmoid')   # 二分类输出
])

model.summary()   # 打印模型结构

# 3. 编译模型
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 4. 训练模型
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=20,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    verbose=1
)

# 5. 保存模型
model.save('baseline_cnn.h5')
print("✅ 模型已保存为 baseline_cnn.h5")

# 6. 绘制训练曲线
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(1, len(acc) + 1)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend()
plt.title('Training and Validation Loss')

plt.tight_layout()
plt.savefig('baseline_training_curves.png')
print("✅ 训练曲线已保存为 baseline_training_curves.png")
plt.show()