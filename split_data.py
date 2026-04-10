import os
import shutil
import random

random.seed(42)

source_dir = 'data/all_images'
train_dir = 'data/train'
val_dir = 'data/validation'

for split in [train_dir, val_dir]:
    for cls in ['cats', 'dogs']:
        os.makedirs(os.path.join(split, cls), exist_ok=True)

all_images = [img for img in os.listdir(source_dir) if img.endswith('.jpg')]

cat_images = [img for img in all_images if img.startswith('cat')]
dog_images = [img for img in all_images if img.startswith('dog')]

random.shuffle(cat_images)
random.shuffle(dog_images)

split_ratio = 0.8

cat_split = int(len(cat_images) * split_ratio)
dog_split = int(len(dog_images) * split_ratio)

cat_train = cat_images[:cat_split]
cat_val = cat_images[cat_split:]

dog_train = dog_images[:dog_split]
dog_val = dog_images[dog_split:]

def copy_files(file_list, src_dir, dst_dir):
    for fname in file_list:
        shutil.copy(os.path.join(src_dir, fname), os.path.join(dst_dir, fname))

copy_files(cat_train, source_dir, os.path.join(train_dir, 'cats'))
copy_files(cat_val, source_dir, os.path.join(val_dir, 'cats'))
copy_files(dog_train, source_dir, os.path.join(train_dir, 'dogs'))
copy_files(dog_val, source_dir, os.path.join(val_dir, 'dogs'))

print("Done!")
print(f"Train cats: {len(cat_train)}")
print(f"Val cats: {len(cat_val)}")
print(f"Train dogs: {len(dog_train)}")
print(f"Val dogs: {len(dog_val)}")