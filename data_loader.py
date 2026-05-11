from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_data_generators(
    train_dir='data/train',
    val_dir='data/validation',
    target_size=(150, 150),
    batch_size=32
):
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

    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary'
    )

    validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary'
    )

    return train_generator, validation_generator


if __name__ == "__main__":
    train_generator, validation_generator = get_data_generators()

    print("Train samples:", train_generator.samples)
    print("Validation samples:", validation_generator.samples)
    print("Class indices:", train_generator.class_indices)