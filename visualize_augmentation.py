import matplotlib.pyplot as plt
from data_loader import get_data_generators

train_generator, validation_generator = get_data_generators()

x_batch, y_batch = next(train_generator)

plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(x_batch[i])
    plt.title('Dog' if y_batch[i] == 1 else 'Cat')
    plt.axis('off')

plt.tight_layout()
plt.savefig('augmentation_examples.png')
plt.show()