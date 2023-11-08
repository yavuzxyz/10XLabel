# This file is part of 10XLabel, which is released under the GNU General Public License (GPL).
# See file LICENSE or go to https://www.gnu.org/licenses/gpl-3.0.html for full license details.

import os
import cv2
import numpy as np
import math
def process_images(path, labels_folder, output_folder, rotation_angles=[90, 180, 270]):
    if os.path.isdir(path):  # If path is a directory
        image_files = os.listdir(path)
        images_folder = path
    elif os.path.isfile(path):  # If path is a file
        image_files = [os.path.basename(path)]
        images_folder = os.path.dirname(path)
    else:
        print(f"Invalid path: {path}")
        return
images_folder = "images"
labels_folder = "labels"
output_folder = "output"
if os.path.isdir(images_folder):  # If images_folder is a directory
    image_files = os.listdir(images_folder)
elif os.path.isfile(images_folder):  # If images_folder is a file
    image_files = [os.path.basename(images_folder)]
    images_folder = os.path.dirname(images_folder)
else:
    print(f"Invalid images_folder: {images_folder}")
if os.path.isdir(labels_folder):  # If labels_folder is a directory
    label_files = os.listdir(labels_folder)
elif os.path.isfile(labels_folder):  # If labels_folder is a file
    label_files = [os.path.basename(labels_folder)]
    labels_folder = os.path.dirname(labels_folder)
else:
    print(f"Invalid labels_folder: {labels_folder}")
rotation_angles = [90, 180, 270]
# Kod 1 Başlangıç
# Her bir görüntü dosyası için
for image_file in image_files:
    # Görüntüyü oku
    img = cv2.imread(os.path.join(images_folder, image_file))
    rows, cols, _ = img.shape
    # Görüntüyü döndürme işlemi
    for angle in rotation_angles:
        # Görüntüyü belirtilen açıda döndür
        M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
        img_rotated = cv2.warpAffine(img,M,(cols,rows))
        # Yeni isimle output_folder'a kaydet
        new_img_name = f"rotated{angle}_" + image_file
        cv2.imwrite(os.path.join(output_folder, new_img_name), img_rotated)
        # Etiket dosyasını oku
        with open(os.path.join(labels_folder, image_file.replace('.jpg', '.txt')), 'r') as f:
            labels = f.readlines()
        for i in range(len(labels)):
            # Etiketi parse et
            line = labels[i].strip().split()
            class_id = int(line[0])
            points = np.array(line[1:], dtype=float).reshape(-1, 2)  # reshaping to get pairs of coordinates
            # Döndürme merkezini hesapla
            rot_center_x = cols / 2.0  # resmin genişliğinin yarısı
            rot_center_y = rows / 2.0  # resmin yüksekliğinin yarısı
            # Dönüşüm matrisini oluştur ve açısını belirle
            M = cv2.getRotationMatrix2D((rot_center_x, rot_center_y), angle, 1)
            # Her bir noktayı döndür ve normalize et
            points_rotated = []
            for point in points:
                x, y = point
                x_rot = M[0, 0] * x * cols + M[0, 1] * y * rows + M[0, 2]
                y_rot = M[1, 0] * x * cols + M[1, 1] * y * rows + M[1, 2]
                x_rot /= cols
                y_rot /= rows
                points_rotated.append([x_rot, y_rot])
            # Yeni etiket satırını oluştur
            labels[i] = f"{class_id} " + " ".join(map(str, np.array(points_rotated).flatten())) + "\n"
        # Yeni etiketleri output_folder'a kaydet
        with open(os.path.join(output_folder, new_img_name.replace('.jpg', '.txt')), 'w') as f:
            f.writelines(labels)
    # Görüntüyü aydınlat
    img_light = cv2.convertScaleAbs(img, alpha=1, beta=60)
    new_img_name_light = "light_" + image_file
    cv2.imwrite(os.path.join(output_folder, new_img_name_light), img_light)
    # Gauss gürültüsü ekleyin
    mean = 0
    var = 190 #şiddeti artır
    sigma = var**0.5
    gauss = np.random.normal(mean, sigma, img.shape)  # Gaussian noise
    img_noisy = np.clip(img + gauss, 0, 255).astype(np.uint8)  # Add the Gaussian noise to the image
    new_img_name_noisy = "noisy_" + image_file
    cv2.imwrite(os.path.join(output_folder, new_img_name_noisy), img_noisy)
    # Gölge efekti oluşturun
    shadow = np.ones(img.shape, dtype="uint8") * 255  # create a white image with same size as the original image
    cv2.rectangle(shadow, (int(cols/4), int(rows/4)), (int(3*cols/4), int(3*rows/4)), (170, 170, 170), -1)  # create a gray rectangle on the white image
    img_shadowed = cv2.addWeighted(img, 0.6, shadow, 0.5, 0)  # blend the original image with the shadow image
    new_img_name_shadowed = "shadowed_" + image_file
    cv2.imwrite(os.path.join(output_folder, new_img_name_shadowed), img_shadowed)
    # Etiket dosyalarını oku ve olduğu gibi output_folder'a kaydet
    with open(os.path.join(labels_folder, image_file.replace('.jpg', '.txt')), 'r') as f_in:
        lines = f_in.readlines()
        with open(os.path.join(output_folder, new_img_name_light.replace('.jpg', '.txt')), 'w') as f_out_light:
            f_out_light.writelines(lines)
        with open(os.path.join(output_folder, new_img_name_noisy.replace('.jpg', '.txt')), 'w') as f_out_noisy:
            f_out_noisy.writelines(lines)
        with open(os.path.join(output_folder, new_img_name_shadowed.replace('.jpg', '.txt')), 'w') as f_out_shadowed:
            f_out_shadowed.writelines(lines)
# For each image file
for image_file in image_files:
    # Read the image
    img = cv2.imread(os.path.join(images_folder, image_file))
    rows, cols, _ = img.shape
    aspect_ratio = cols / rows
    # Read the label file
    with open(os.path.join(labels_folder, image_file.replace('.jpg', '.txt')), 'r') as f:
        labels = f.readlines()
    # Cropped
    bboxes = []
    for label in labels:
        numbers = list(map(float, label.strip().split()))
        if len(numbers) == 5:  # Bounding box
            class_id, x_center, y_center, width, height = numbers
            x1 = (x_center - width / 2) * cols
            y1 = (y_center - height / 2) * rows
            x2 = (x_center + width / 2) * cols
            y2 = (y_center + height / 2) * rows
            bboxes.append((x1, y1, x2, y2))
        elif len(numbers) > 5:  # Polygon
            class_id = numbers[0]
            coordinates = numbers[1:]
            for i in range(0, len(coordinates), 2):
                x_center, y_center = coordinates[i], coordinates[i + 1]
                x1 = x_center * cols
                y1 = y_center * rows
                # For polygon, we take every point as a box of size 0
                bboxes.append((x1, y1, x1, y1))
    x1_crop = max(0, min(x1 for x1, y1, x2, y2 in bboxes))
    y1_crop = max(0, min(y1 for x1, y1, x2, y2 in bboxes))
    x2_crop = min(cols, max(x2 for x1, y1, x2, y2 in bboxes))
    y2_crop = min(rows, max(y2 for x1, y1, x2, y2 in bboxes))
    crop_width = x2_crop - x1_crop
    crop_height = y2_crop - y1_crop
    crop_center_x = x1_crop + crop_width / 2
    crop_center_y = y1_crop + crop_height / 2
    if crop_width / crop_height > aspect_ratio:
        # The crop is too wide, adjust height
        crop_height = crop_width / aspect_ratio
    else:
        # The crop is too tall, adjust width
        crop_width = crop_height * aspect_ratio
    x1_crop = max(0, crop_center_x - crop_width / 2)
    y1_crop = max(0, crop_center_y - crop_height / 2)
    x2_crop = min(cols, crop_center_x + crop_width / 2)
    y2_crop = min(rows, crop_center_y + crop_height / 2)
    img_cropped = img[int(y1_crop):int(y2_crop), int(x1_crop):int(x2_crop)]
    new_img_name_cropped = "cropped_" + image_file
    cv2.imwrite(os.path.join(output_folder, new_img_name_cropped), img_cropped)
    labels_cropped = []
    for label in labels:
        numbers = list(map(float, label.strip().split()))
        if len(numbers) == 5:  # Bounding box
            class_id, x_center, y_center, width, height = numbers
            x_center_cropped = (x_center * cols - x1_crop) / (x2_crop - x1_crop)
            y_center_cropped = (y_center * rows - y1_crop) / (y2_crop - y1_crop)
            width_cropped = width * cols / (x2_crop - x1_crop)
            height_cropped = height * rows / (y2_crop - y1_crop)
            labels_cropped.append(f"{int(class_id)} {x_center_cropped} {y_center_cropped} {width_cropped} {height_cropped}\n")
        elif len(numbers) > 5:  # Polygon
            class_id = numbers[0]
            coordinates = numbers[1:]
            coords_cropped = []
            for i in range(0, len(coordinates), 2):
                x_center, y_center = coordinates[i], coordinates[i + 1]
                x_center_cropped = (x_center * cols - x1_crop) / (x2_crop - x1_crop)
                y_center_cropped = (y_center * rows - y1_crop) / (y2_crop - y1_crop)
                coords_cropped.extend([x_center_cropped, y_center_cropped])
            labels_cropped.append(f"{int(class_id)} {' '.join(map(str, coords_cropped))}\n")
    with open(os.path.join(output_folder, new_img_name_cropped.replace('.jpg', '.txt')), 'w') as f:
        f.writelines(labels_cropped)
    # flipped1
    img_flipped1 = cv2.flip(img, -1)
    new_img_name = "flipped1_" + image_file
    cv2.imwrite(os.path.join(output_folder, new_img_name), img_flipped1)
    labels_flipped1 = []
    for i in range(len(labels)):
        # Parse the label
        line = labels[i].strip().split()
        class_id = int(line[0])
        points = np.array(line[1:], dtype=float).reshape(-1, 2)  # reshaping to get pairs of coordinates
        # Flip each point
        points_flipped1 = 1 - points
        labels_flipped1.append(f"{class_id} " + " ".join(map(str, points_flipped1.flatten())) + "\n")
    with open(os.path.join(output_folder, new_img_name.replace('.jpg', '.txt')), 'w') as f:
        f.writelines(labels_flipped1)
    # flipped2
    img_flipped2 = cv2.flip(img, 0)
    new_img_name = "flipped2_" + image_file
    cv2.imwrite(os.path.join(output_folder, new_img_name), img_flipped2)
    labels_flipped2 = []
    for i in range(len(labels)):
        # Parse the label
        line = labels[i].strip().split()
        class_id = int(line[0])
        points = np.array(line[1:], dtype=float).reshape(-1, 2)  # reshaping to get pairs of coordinates
        # Flip the y-coordinates of each point
        points[:, 1] = 1 - points[:, 1]
        labels_flipped2.append(f"{class_id} " + " ".join(map(str, points.flatten())) + "\n")
    with open(os.path.join(output_folder, new_img_name.replace('.jpg', '.txt')), 'w') as f:
        f.writelines(labels_flipped2)
    # zoomedout
    large_img = np.zeros((int(rows * 1.5), int(cols * 1.5), 3), dtype=np.uint8)
    start_row = (large_img.shape[0] - rows) // 2
    start_col = (large_img.shape[1] - cols) // 2
    large_img[start_row:start_row+rows, start_col:start_col+cols] = img
    new_img_name = "zoomedout_" + image_file
    cv2.imwrite(os.path.join(output_folder, new_img_name), large_img)
    labels_zoomedout = []
    for i in range(len(labels)):
        # Parse the label
        line = labels[i].strip().split()
        class_id = int(line[0])
        points = np.array(line[1:], dtype=float).reshape(-1, 2)  # reshaping to get pairs of coordinates
        # Scale the points
        points[:, 0] = (points[:, 0] * cols + start_col) / large_img.shape[1]
        points[:, 1] = (points[:, 1] * rows + start_row) / large_img.shape[0]
        labels_zoomedout.append(f"{class_id} " + " ".join(map(str, points.flatten())) + "\n")
    with open(os.path.join(output_folder, new_img_name.replace('.jpg', '.txt')), 'w') as f:
        f.writelines(labels_zoomedout)