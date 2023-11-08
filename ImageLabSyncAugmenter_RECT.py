# This file is part of 10XLabel, which is released under the GNU General Public License (GPL).
# See file LICENSE or go to https://www.gnu.org/licenses/gpl-3.0.html for full license details.


import os
import cv2
import numpy as np
import math
images_folder = "images"
labels_folder = "labels"
output_folder = "output"
# Check whether images_folder is a directory or a file
if os.path.isdir(images_folder):  # If images_folder is a directory
    image_files = os.listdir(images_folder)
elif os.path.isfile(images_folder):  # If images_folder is a file
    image_files = [os.path.basename(images_folder)]
    images_folder = os.path.dirname(images_folder)
else:
    print(f"Invalid images_folder: {images_folder}")
# Check whether labels_folder is a directory or a file
if os.path.isdir(labels_folder):  # If labels_folder is a directory
    label_files = os.listdir(labels_folder)
elif os.path.isfile(labels_folder):  # If labels_folder is a file
    label_files = [os.path.basename(labels_folder)]
    labels_folder = os.path.dirname(labels_folder)
else:
    print(f"Invalid labels_folder: {labels_folder}")
# Define rotation angles for image augmentation
rotation_angles = [90, 180, 270]
# For each image file
for image_file in image_files:
    # Read the image
    img = cv2.imread(os.path.join(images_folder, image_file))
    rows, cols, _ = img.shape
    # Read the label file
    with open(os.path.join(labels_folder, image_file.replace('.jpg', '.txt')), 'r') as f:
        labels = f.readlines()
    # Rotate the image and label by the specified angle
    for angle in rotation_angles:
        # Rotate the image
        M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
        img_rotated = cv2.warpAffine(img,M,(cols,rows))
        
        # Save the rotated image with a new name
        new_img_name = f"rotated{angle}_" + image_file
        cv2.imwrite(os.path.join(output_folder, new_img_name), img_rotated)
        
        # Rotate the label coordinates
        labels_rotated = []
        for i in range(len(labels)):
            class_id, x_center, y_center, width, height = map(float, labels[i].strip().split())
            rot_center_x = cols / 2.0  # half of the image width
            rot_center_y = rows / 2.0  # half of the image height
            M = cv2.getRotationMatrix2D((rot_center_x, rot_center_y), angle, 1)
            x_center_rot = M[0, 0] * x_center * cols + M[0, 1] * y_center * rows + M[0, 2]
            y_center_rot = M[1, 0] * x_center * cols + M[1, 1] * y_center * rows + M[1, 2]
            box_points = np.array([
                [x_center - width / 2, y_center - height / 2],  # top left corner
                [x_center + width / 2, y_center - height / 2],  # top right corner
                [x_center - width / 2, y_center + height / 2],  # bottom left corner
                [x_center + width / 2, y_center + height / 2]   # bottom right corner
            ]) * np.array([cols, rows])  # convert normalized coordinates to pixels
            rotated_box_points = cv2.transform(box_points.reshape(-1, 1, 2), M)
            width_rot = np.max(rotated_box_points[..., 0]) - np.min(rotated_box_points[..., 0])
            height_rot = np.max(rotated_box_points[..., 1]) - np.min(rotated_box_points[..., 1])
            x_center_rot /= cols
            y_center_rot /= rows
            width_rot /= cols
            height_rot /= rows
            labels_rotated.append(f"{int(class_id)} {x_center_rot} {y_center_rot} {width_rot} {height_rot}\n")

        # Save the new labels for the rotated images
        with open(os.path.join(output_folder, new_img_name.replace('.jpg', '.txt')), 'w') as f:
            f.writelines(labels_rotated)

    # Brighten the image
    img_light = cv2.convertScaleAbs(img, alpha=1, beta=60)
    new_img_name_light = "light_" + image_file
    cv2.imwrite(os.path.join(output_folder, new_img_name_light), img_light)

    # Add Gaussian noise
    mean = 0
    var = 190 #increase the intensity
    sigma = var**0.5
    gauss = np.random.normal(mean, sigma, img.shape)  # Gaussian noise
    img_noisy = np.clip(img + gauss, 0, 255).astype(np.uint8)  # Add the Gaussian noise to the image
    new_img_name_noisy = "noisy_" + image_file
    cv2.imwrite(os.path.join(output_folder, new_img_name_noisy), img_noisy)
    
    # Create shadow effect
    shadow = np.ones(img.shape, dtype="uint8") * 255  # create a white image with same size as the original image
    cv2.rectangle(shadow, (int(cols/4), int(rows/4)), (int(3*cols/4), int(3*rows/4)), (170, 170, 170), -1)  # create a gray rectangle on the white image
    img_shadowed = cv2.addWeighted(img, 0.6, shadow, 0.5, 0)  # blend the original image with the shadow image
    new_img_name_shadowed = "shadowed_" + image_file
    cv2.imwrite(os.path.join(output_folder, new_img_name_shadowed), img_shadowed)

    # Copy the label files as they are
    with open(os.path.join(labels_folder, image_file.replace('.jpg', '.txt')), 'r') as f_in:
        lines = f_in.readlines()
        with open(os.path.join(output_folder, new_img_name_light.replace('.jpg', '.txt')), 'w') as f_out_light:
            f_out_light.writelines(lines)
        with open(os.path.join(output_folder, new_img_name_noisy.replace('.jpg', '.txt')), 'w') as f_out_noisy:
            f_out_noisy.writelines(lines)
        with open(os.path.join(output_folder, new_img_name_shadowed.replace('.jpg', '.txt')), 'w') as f_out_shadowed:
            f_out_shadowed.writelines(lines)

    # Flip the image and label horizontally and vertically
    img_flipped1 = cv2.flip(img, -1)
    new_img_name = "flipped1_" + image_file
    cv2.imwrite(os.path.join(output_folder, new_img_name), img_flipped1)
    labels_flipped1 = []
    for i in range(len(labels)):
        class_id, x_center, y_center, width, height = map(float, labels[i].strip().split())
        x_center_flipped1 = 1 - x_center
        y_center_flipped1 = 1 - y_center
        labels_flipped1.append(f"{int(class_id)} {x_center_flipped1} {y_center_flipped1} {width} {height}\n")
    with open(os.path.join(output_folder, new_img_name.replace('.jpg', '.txt')), 'w') as f:
        f.writelines(labels_flipped1)
    
    # Flip the image and label horizontally
    img_flipped2 = cv2.flip(img, 1)
    new_img_name = "flipped2_" + image_file
    cv2.imwrite(os.path.join(output_folder, new_img_name), img_flipped2)
    labels_flipped2 = []
    for i in range(len(labels)):
        class_id, x_center, y_center, width, height = map(float, labels[i].strip().split())
        x_center_flipped2 = 1 - x_center
        labels_flipped2.append(f"{int(class_id)} {x_center_flipped2} {y_center} {width} {height}\n")
    with open(os.path.join(output_folder, new_img_name.replace('.jpg', '.txt')), 'w') as f:
        f.writelines(labels_flipped2)
        
    # Zoom out the image and adjust the label coordinates
    large_img = np.zeros((int(rows * 1.5), int(cols * 1.5), 3), dtype=np.uint8)
    start_row = (large_img.shape[0] - rows) // 2
    start_col = (large_img.shape[1] - cols) // 2
    large_img[start_row:start_row+rows, start_col:start_col+cols] = img
    new_img_name = "zoomedout_" + image_file
    cv2.imwrite(os.path.join(output_folder, new_img_name), large_img)
    labels_zoomedout = []
    for i in range(len(labels)):
        class_id, x_center, y_center, width, height = map(float, labels[i].strip().split())
        x_center_zoomedout = (x_center * cols + start_col) / large_img.shape[1]
        y_center_zoomedout = (y_center * rows + start_row) / large_img.shape[0]
        width_zoomedout = width * cols / large_img.shape[1]
        height_zoomedout = height * rows / large_img.shape[0]
        labels_zoomedout.append(f"{int(class_id)} {x_center_zoomedout} {y_center_zoomedout} {width_zoomedout} {height_zoomedout}\n")
    with open(os.path.join(output_folder, new_img_name.replace('.jpg', '.txt')), 'w') as f:
        f.writelines(labels_zoomedout)
        
for image_file, label_file in zip(image_files, label_files):
    img = cv2.imread(os.path.join(images_folder, image_file))
    with open(os.path.join(labels_folder, label_file)) as f:
        labels = f.readlines()

    rows, cols, _ = img.shape
    aspect_ratio = cols / rows

    # calculate bounding boxes
    bboxes = []
    for label in labels:
        class_id, x_center, y_center, width, height = map(float, label.strip().split())
        x1 = (x_center - width / 2) * cols
        y1 = (y_center - height / 2) * rows
        x2 = (x_center + width / 2) * cols
        y2 = (y_center + height / 2) * rows
        bboxes.append((x1, y1, x2, y2))

    # calculate cropping area
    x1_crop = max(0, min(x1 for x1, _, _, _ in bboxes))
    y1_crop = max(0, min(y1 for _, y1, _, _ in bboxes))
    x2_crop = min(cols, max(x2 for _, _, x2, _ in bboxes))
    y2_crop = min(rows, max(y2 for _, _, _, y2 in bboxes))

    # calculate cropping area dimensions
    crop_width = x2_crop - x1_crop
    crop_height = y2_crop - y1_crop

    # adjust cropping area dimensions to maintain aspect ratio
    if crop_width < crop_height:
        adjust = (crop_height - crop_width) / 2
        x1_crop = max(0, x1_crop - adjust)
        x2_crop = min(cols, x2_crop + adjust)
    else:
        adjust = (crop_width - crop_height) / 2
        y1_crop = max(0, y1_crop - adjust)
        y2_crop = min(rows, y2_crop + adjust)

    # calculate final cropping area dimensions
    crop_width = x2_crop - x1_crop
    crop_height = y2_crop - y1_crop

    # crop image
    img_cropped = img[int(y1_crop):int(y2_crop), int(x1_crop):int(x2_crop)]
    cv2.imwrite(os.path.join(output_folder, f"cropped_{image_file}"), img_cropped)

    # adjust labels
    labels_cropped = []
    for label in labels:
        class_id, x_center, y_center, width, height = map(float, label.strip().split())
        x_center_cropped = (x_center * cols - x1_crop) / crop_width
        y_center_cropped = (y_center * rows - y1_crop) / crop_height
        width_cropped = width * cols / crop_width
        height_cropped = height * rows / crop_height
        labels_cropped.append(f"{int(class_id)} {x_center_cropped} {y_center_cropped} {width_cropped} {height_cropped}\n")

    with open(os.path.join(output_folder, f"cropped_{label_file}"), 'w') as f:
        f.writelines(labels_cropped)