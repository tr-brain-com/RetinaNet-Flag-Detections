import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

def yolo_unnormalize_pascal_voc(x_center, y_center, w, h,  image_w, image_h):
    w = w * image_w
    h = h * image_h
    x1 = ((2 * x_center * image_w) - w)/2
    y1 = ((2 * y_center * image_h) - h)/2
    x2 = x1 + w
    y2 = y1 + h
    return [x1, y1, x2, y2]

def plot_boxes_PIL(boxes, img, color=None, labels=None, line_thickness=None):
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    line_thickness = line_thickness or max(int(min(img.size) / 200), 2)
    for label, box in zip(labels, boxes):
        print(f'box 0 is {box[0]} and box 1 is {box[1]} and box 2 is {box[2]} and box 3 is {box[3]}')
        draw.rectangle(box, width=line_thickness, outline=tuple(color))  # plot
        if label is not None:
            fontsize = max(round(max(img.size) / 40), 12)
            font = ImageFont.truetype('api/source/font/arial.ttf', fontsize)
            txt_width, txt_height = font.getsize(label)
            draw.rectangle([box[0], box[1] - txt_height + 4, box[0] + txt_width, box[1]], fill=tuple(color))
            draw.text((box[0], box[1] - txt_height + 1), label, fill=(255, 255, 255), font=font)
    return np.asarray(img)

def yolo_2_pascal_voc():
    search_images_dir = "/home/thecoderman/Downloads/flag_dataset/images/"
    search_annotation_dir ="/home/thecoderman/Downloads/flag_dataset/labels/"

    target_images_dir = "dataset/flags/images/"
    target_annotation_dir = "dataset/flags/annotations/"


    for dir in os.listdir(search_images_dir):
        active_image_dir = search_images_dir+dir+"/"
        active_target_dir = target_images_dir+dir+"/"

        print(f"starting for {dir} ")
        for a_dir in os.listdir(active_image_dir):
            img_file_name = search_images_dir+a_dir
            xml_file_name = search_annotation_dir+a_dir.split(".")[0]+".xml"
            print(xml_file_name, " ", img_file_name)

yolo_2_pascal_voc()

