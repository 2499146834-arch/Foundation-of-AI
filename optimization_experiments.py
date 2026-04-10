import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import matplotlib.pyplot as plt
import os

# -------------------- Data generator factory --------------------
def get_generators(use_augmentation=True, batch_size=32, target_size=(150,150)):
    """
    Returns train/validation generators based on use_augmentation flag.
    Validation set always uses only rescaling.
    """
    if use_augmentation:
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
    else:
        train_datagen = ImageDataGenerator(rescale=1./255)

    val_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        'data/train',
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=True
    )
    val_gen = val_datagen.flow_from_directory(
        'data/validation',
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )
    return train_gen, val_gen

# -------------------- Model builder factory --------------------
def create_model(dropout_rate=0.0, use_batchnorm=False, l2_lambda=None):
    """
    Builds a CNN model with the given hyperparameters.
    """
    model = models.Sequential()
    # Convolutional base
    model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)))
    model.add(layers.MaxPooling2D(2,2))
    if use_batchnorm:
        model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(64, (3,3), activation='relu'))
    model.add(layers.MaxPooling2D(2,2))
    if use_batchnorm:
        model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(128, (3,3), activation='relu'))
    model.add(layers.MaxPooling2D(2,2))

    model.add(layers.Flatten())
    if dropout_rate > 0:
        model.add(layers.Dropout(dropout_rate))

    # Fully connected layers
    if l2_lambda:
        model.add(layers.Dense(512, activation='relu',
                               kernel_regularizer=tf.keras.regularizers.l2(l2_lambda)))
    else:
        model.add(layers.Dense(512, activation='relu'))

    model.add(layers.Dense(1, activation='sigmoid'))
    return model

# -------------------- Experiment configurations --------------------
experiments = [
    {
        'name': 'Adam_baseline',
        'optimizer': 'adam',
        'dropout': 0.0,
        'batch_norm': False,
        'l2': None,
        'augmentation': True
    },
    {
        'name': 'SGD_lr0.01',
        'optimizer': optimizers.SGD(learning_rate=0.01, momentum=0.9),
        'dropout': 0.0,
        'batch_norm': False,
        'l2': None,
        'augmentation': True
    },
    {
        'name': 'Adam_dropout0.5',
        'optimizer': 'adam',
        'dropout': 0.5,
        'batch_norm': False,
        'l2': None,
        'augmentation': True
    },
    {
        'name': 'Adam_batchnorm',
        'optimizer': 'adam',
        'dropout': 0.0,
        'batch_norm': True,
        'l2': None,
        'augmentation': True
    },
    {
        'name': 'Adam_no_aug',
        'optimizer': 'adam',
        'dropout': 0.0,
        'batch_norm': False,
        'l2': None,
        'augmentation': False
    },
    {
        'name': 'RMSprop',
        'optimizer': 'rmsprop',
        'dropout': 0.0,
        'batch_norm': False,
        'l2': None,
        'augmentation': True
    }
    # You can add more experiments here (e.g., L2 regularization)
]

# -------------------- Run experiments --------------------
results = []
batch_size = 32
epochs = 20

for exp in experiments:
    print(f"\n========== Running experiment: {exp['name']} ==========")

    # 1. Get data generators
    train_gen, val_gen = get_generators(
        use_augmentation=exp['augmentation'],
        batch_size=batch_size
    )

    # 2. Build model
    model = create_model(
        dropout_rate=exp['dropout'],
        use_batchnorm=exp['batch_norm'],
        l2_lambda=exp['l2']
    )

    # 3. Compile model
    optimizer = exp['optimizer']
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # 4. Train model
    history = model.fit(
        train_gen,
        steps_per_epoch=train_gen.samples // batch_size,
        epochs=epochs,
        validation_data=val_gen,
        validation_steps=val_gen.samples // batch_size,
        verbose=1
    )

    # 5. Record final metrics
    val_acc = history.history['val_accuracy'][-1]
    val_loss = history.history['val_loss'][-1]
    train_acc = history.history['accuracy'][-1]
    train_loss = history.history['loss'][-1]

    results.append({
        'name': exp['name'],
        'val_accuracy': val_acc,
        'val_loss': val_loss,
        'train_accuracy': train_acc,
        'train_loss': train_loss,
        'config': str(exp)   # store configuration as string for reference
    })

    # 6. Save model
    model.save(f"model_{exp['name']}.h5")
    print(f"Model saved as model_{exp['name']}.h5")

# -------------------- Save results to CSV --------------------
df = pd.DataFrame(results)
df.to_csv('experiment_results.csv', index=False)
print("\nAll experiment results saved to experiment_results.csv")

# -------------------- Plot comparison bar chart --------------------
plt.figure(figsize=(10, 6))
plt.bar(df['name'], df['val_accuracy'], color='skyblue')
plt.xlabel('Experiment')
plt.ylabel('Validation Accuracy')
plt.title('Comparison of Validation Accuracy across Experiments')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('experiment_comparison.png')
plt.show()
print("Comparison chart saved as experiment_comparison.png")