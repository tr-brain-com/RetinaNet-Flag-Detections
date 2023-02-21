import os
import time
from glob import glob
from os.path import isfile, join

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from keras_retinanet.utils.colors import label_color
from keras_retinanet.utils.image import preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from matplotlib import pyplot as plt
import xml.etree.ElementTree as ET
model_paths = glob('snapshots/resnet50_csv_10.h5')
latest_path = sorted(model_paths)[-1]
print("path:", latest_path)

from keras_retinanet import models

model = models.load_model(latest_path, backbone_name='resnet50')
model = models.convert_model(model)

label_map = {}
for line in open('dataset/flags/maskDetectorClasses.csv'):
  row = line.rstrip().split(',')
  label_map[int(row[1])] = row[0]

print(label_map)

# Write a function to choose one image randomly from your dataset and predict using Trained model.
def show_image_with_predictions(df, threshold):
    # choose a random image
    row = df.sample()
    filepath = row['fileName'].values[0]
    print("filepath:", filepath)


    # get all rows for this image
    df2 = df[df['fileName'] == filepath]
    im = np.array(Image.open(filepath))
    print("im.shape:", im.shape)


    # if there's a PNG it will have alpha channel
    #im = im[:, :, :3]
    """print("im.shape:", im.shape)
    # plot true boxes
    for idx, row in df2.iterrows():
        box = [
            row['xmin'],
            row['ymin'],
            row['xmax'],
            row['ymax'],
        ]
        #print(box)
        #draw_box(im, box, color=(255, 0, 0))

    ### plot predictions ###

    # get predictions"""
    #im = preprocess_image(im)
    imp, scale = resize_image(im)

    boxes, scores, labels = model.predict_on_batch(
        np.expand_dims(imp, axis=0)
    )
    print("start box")
    print(boxes)
    print(scores)
    print(labels)

    # standardize box coordinates
    boxes /= scale
    threshold = threshold
    # loop through each prediction for the input image
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        # scores are sorted so we can quit as soon
        # as we see a score below threshold
        if score < threshold:
            break

        box = box.astype(np.int32)
        color = label_color(label)

        draw_box(im, box, color=color)

        class_name = label_map[label]
        caption = f"{class_name} {score:.3f}"
        draw_caption(im, box, caption)
        score, label = score, label

    plt.axis('off')
    plt.imshow(im)
    plt.show()
    return score, label

plt.rcParams['figure.figsize'] = [20, 10]



search_images_path     = "/home/thecoderman/Downloads/flag_dataset/test/images/"
search_annotation_path = "/home/thecoderman/Downloads/flag_dataset/test/annotations/"
data=pd.DataFrame(columns=['fileName','xmin','ymin','xmax','ymax','class'])

os.getcwd()
#read All files
allfiles = [f for f in os.listdir(search_annotation_path) if isfile(join(search_annotation_path, f))]
for file in allfiles:
    # print(file)
    if (file.split(".")[1] == 'xml'):
        fileName = search_images_path + file.replace(".xml", '.jpg')
        tree = ET.parse(search_annotation_path + file)
        root = tree.getroot()
        for obj in root.iter('object'):
            cls_name = obj.find('name').text
            xml_box = obj.find('bndbox')
            xmin = xml_box.find('xmin').text
            ymin = xml_box.find('ymin').text
            xmax = xml_box.find('xmax').text
            ymax = xml_box.find('ymax').text
            # Append rows in Empty Dataframe by adding dictionaries
            data = data.append(
                {'fileName': fileName, 'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax, 'class': cls_name},
                ignore_index=True)

data.shape

start_time= time.time()
score, label=show_image_with_predictions(data, threshold=0.4)
print(f"süre : {time.time() - start_time}")
print(score, label)
