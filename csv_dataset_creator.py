import numpy as np
import shutil
import pandas as pd
import os, sys, random
import xml.etree.ElementTree as ET
import pandas as pd
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
from PIL import Image
import requests
import urllib
from keras_retinanet.utils.visualization import draw_box, draw_caption , label_color
from keras_retinanet.utils.image import preprocess_image, resize_image

search_images_path     = "/home/thecoderman/Downloads/flag_dataset/train/images/"
search_annotation_path = "/home/thecoderman/Downloads/flag_dataset/train/annotations/"


data=pd.DataFrame(columns=['fileName','xmin','ymin','xmax','ymax','class'])

allfiles = [f for f in listdir(search_annotation_path) if isfile(join(search_annotation_path, f))]

def loadTestData():
    search_annotation_path = "/home/thecoderman/Downloads/flag_dataset/test/annotations/"
    search_images_path = "/home/thecoderman/Downloads/flag_dataset/test/images/"
    data = pd.DataFrame(columns=['fileName', 'xmin', 'ymin', 'xmax', 'ymax', 'class'])

    allfiles = [f for f in listdir(search_annotation_path) if isfile(join(search_annotation_path, f))]

    for file in allfiles:
        # print(file)
        if (file.split(".")[1] == 'xml'):
            fileName = search_images_path + file.replace(".xml",'.jpg')
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
                    {'filename': fileName, 'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax, 'class': cls_name},
                    ignore_index=True)
    return data

with open('dataset/flags/maskDetectorData.csv', 'w') as f:
    for file in allfiles:

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
                f.write(f'{fileName}, {xmin}, {ymin}, {xmax}, {ymax}, {cls_name} \n')

                """data = data.append(
                    {'fileName': fileName, 'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax, 'class': cls_name},
                    ignore_index=True)"""

def show_image_with_boxes(df):
    #rastgele bir resim seç
    filepath = df.sample()['filename'].values[0]
    df2 = df[df['filename'] == filepath]
    im = np.array(Image.open(filepath))

    #Eğer PNG kullanıyorsanız bu şekilde resmi üç kanala indirebilirsiniz.
    #im = im[:, :, :3]

    for idx, row in df2.iterrows():
        box = [
            row['xmin'],
            row['ymin'],
            row['xmax'],
            row['ymax'],
        ]
        print(box)
        draw_box(im, box, color=(255, 0, 0))

    plt.axis('off')
    plt.imshow(im)
    plt.show()


classes = ['flag']
with open('dataset/flags/maskDetectorClasses.csv', 'w') as f:
    for i, class_name in enumerate(classes):
        f.write(f'{class_name},{i}\n')

test_data = loadTestData()

show_image_with_boxes(test_data)