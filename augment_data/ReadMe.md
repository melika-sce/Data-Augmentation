# Data Augmentation 

## consist of the following

### preprocess.py 
- to load images and annotations from VOC or Yolo format and convert them to specific format that augmentation methods need.


### Data_augmentation.py 
- consist of all Augmentation methods

### custom_augment.py
- consist of three methods: 

1. augment: load images from path and add augmentation to whole image
2. three_channel_img: add augmenation to whole image (use in middle of dataset making code)
3. four_channel_img: add augmentation to only object (use in middle of dataset making code)

### main_fixed_angle.py
code for making syntethic dataset


### To add augmentation on whole image, while making the dataset:
uncomment line 484 on main_fixed_angle.py

	# img_comp, annotations_yolo = three_channel_img('yolo', img_comp, annotations_yolo)



### To add augmentation on objects only, while making the dataset:
uncomment line 267 on main_fixed_angle.py
	
	# obj_img = four_channel_img('yolo', obj_img, [0, 0, 0, 0, 0])


