__author__ = '__Melika'

import xml.etree.ElementTree as ET
import os, glob, cv2
import pandas as pd


def load_images(path):
    """
        load images from folder

        parameters
        ----------
        path: str
            images folder path

        returns
        -------
        dictionary
            dict of images name and data in the folder
        """
    images = []
    files = glob.glob(os.path.join(path, '*.png'))

    for file in files:
        image = cv2.imread(file)
        images.append(image)
    return images


def load_annotations(path, type):
    """
        load image annotations from folder

        parameters
        ----------
        path: str
            images folder path
        type: str
            annotations type (voc/coco/yolo)

        returns
        -------
        list
            list of bonding boxes and labels
        """
    anns_list = []
    if type == "pascal_voc":
        ext = '*.xml'
        files = glob.glob(os.path.join(path, ext))
        for file in files:
            anns = voc_to_list(file)
            anns_list.append(anns)

    elif type == "coco":
        ext = '*.json'
    elif type == "yolo":
        ext = '*.txt'
        files = glob.glob(os.path.join(path, ext))
        for file in files:
            anns = yolo_to_list(file)
            anns_list.append(anns)

    return anns_list


def load_txt_file(txt_path):
    df = pd.read_csv(txt_path, sep=" ", header=None, names=["class", "x", "y", "w", "h"])
    return df


def visualize(image, boxes):
    xmin, ymin, xmax, ymax, label = boxes[0]
    xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
    print(xmin, ymin, xmax, ymax, label)
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
    # cv2.putText(image, label, (xmin, ymin), (xmax, ymax), cv2.FONT_HERSHEY_SIMPLEX, (255, 0, 0), 2)

    cv2.imshow('image', image)
    cv2.waitKey(0)


def voc_to_list(xml_path):
    """
            convert voc anotation to list

            parameters
            ----------
            xml_path: str
                annotation path

            returns
            -------
            list
                list contain the list of bonding boxes and labels in xml file
            """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    height = int(root.find("size")[0].text)
    width = int(root.find("size")[1].text)
    all_anns = []
    class_labels = []

    for boxes in root.iter('object'):
        filename = root.find('filename').text
        ymin, xmin, ymax, xmax = None, None, None, None

        ymin = int(boxes.find("bndbox/ymin").text)
        xmin = int(boxes.find("bndbox/xmin").text)
        ymax = int(boxes.find("bndbox/ymax").text)
        xmax = int(boxes.find("bndbox/xmax").text)
        label = boxes.find("name").text

        anns = [xmin, ymin, xmax, ymax, label]
        all_anns.append(anns)
        # class_labels.append(label)

    return all_anns


def yolo_to_list(txt_path):
    """
        convert yolo anotation to list

        parameters
        ----------
        txt_path: str
            annotation path

        returns
        -------
        list
            list contain the list of bonding boxes and labels in txt file
    """
    all_anns = []
    data = load_txt_file(txt_path)
    if data.empty:
        return ['empty']
    else:
        ls_data = data.values.tolist()
        for i in range(len(ls_data)):
            data = ls_data[i]
            label = int(data[0])
            x_center = float(data[1])
            y_center = float(data[2])
            width = float(data[3])
            height = float(data[4])

            anns = [label, x_center, y_center, width, height]
            all_anns.append(anns)
            return all_anns


def save_data_in_yolo_format(filename, path, image, annotation):
    """
        save augmented image and annotation in yolo format

        parameters
        ----------
        filename : str
            file path
        path: str
            saving path
        image: array
            augmented image data
        annotation: list
            augmented bounding box data
    """

    image_name = filename.split('/')[-1]
    txt_name = image_name.split('.')[0] + '.txt'

    image_path = os.path.join(path, image_name)
    annotation_path = os.path.join(path, txt_name)
    # resize img
    resized_img = cv2.resize(image, (1280, 720))
    # save augmented image
    if not cv2.imwrite(image_path, resized_img):
        raise Exception('Could not write augmented image')

    # save augmented annotation file
    for i in range(len(annotation)):
        if annotation[i] == 'empty':
            with open(annotation_path, "a") as file:
                file.write('')
        else:
            x_center = annotation[i][1]
            y_center = annotation[i][2]
            width = annotation[i][3]
            height = annotation[i][4]
            label = annotation[i][0]

            anns = str(label) + " " + str(x_center) + " " + str(y_center) + " " + str(width) + " " + str(height)

            with open(annotation_path, "a") as file:
                file.write(anns)
                file.write("\n")
