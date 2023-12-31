__author__ = '__Melika'

from augment_data.Data_augmentation import DataAugmentation
import preprocess as pre
from imgaug import augmenters as iaa
import glob, os,cv2

images_path = ''
save_path = ''
# FDM_source = 'dataset/backgrounds/'
# reference_image = '/test/all/'

# format = 'pascal_voc'
format = 'yolo'


"""
 to set probability for RandomShape method in Data_augmentation class uncomment the code below and set it to desire number
"""
probability = 0.7


def augment(images_path, format):
    """

    Parameters
    ----------
    images_path : string
    format : string

    -------
    save augmented images
    """
    for image in glob.glob(os.path.join(images_path, '*.jpg')):
        txt_name = image.split('.')[0] + '.txt'

        img = cv2.imread(image, cv2.IMREAD_UNCHANGED)
        ann = pre.yolo_to_list(txt_name)

        da = DataAugmentation(img, ann, format, probability)

        # add filters to apply
        # you can customize function parameters
        # da.DustyLook()
        da.RandomShape(num=1, color='blue', shape='circle').RandomShape(num=1, color='blue', shape='rectangle')
        augmented_image = da.__dict__['image']
        augmented_bbox = da.__dict__['bbox']

        # visualize the result after adding filters
        # pre.visualize(augmented_image, augmented_bbox)

        # save augmented data
        pre.save_data_in_yolo_format(image, save_path, augmented_image, augmented_bbox)
        # print(image)
        # print(txt_name)


def three_channel_img(format, image, annotation):
        
        da = DataAugmentation(image, annotation, format, probability)
        # add filters to apply
        # you can customize function parameters
        # da.DustyLook()
        da.RandomShape(num=1, color='blue', shape='circle').RandomShape(num=1, color='blue', shape='rectangle')
        augmented_image = da.__dict__['image']
        augmented_bbox = da.__dict__['bbox']

         # return augmented data
        return augmented_image, augmented_bbox


def four_channel_img(format, image, annotation):

    # convert 4 channel img to 3 channel for color augmentation
    b = image[:,:,0]
    g = image[:,:,1]
    r = image[:,:,2]
    a = image[:,:,3]

    img_merge = cv2.merge((b,g,r))

    da = DataAugmentation(img_merge, annotation, format, probability)

    # add filters to apply
    # you can customize function parameters
    da.DustyLook()
    augmented_image = da.__dict__['image']
    augmented_bbox = da.__dict__['bbox']

    b2 = augmented_image[:,:,0]
    g2 = augmented_image[:,:,1]
    r2 = augmented_image[:,:,2]
    final_img = cv2.merge((b2,g2,r2,a))

    # return augmented data
    return final_img


if __name__ == '__main__':
    # load image and annotation and augment whole image
    augment(images_path, format)
