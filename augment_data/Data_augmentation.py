__author__ = '__Melika'

# Imports
import glob
import albumentations as A
from imgaug import augmenters as iaa
import numpy as np
import cv2
import glob
import os
import random
import matplotlib.pyplot as plt
#from Data_Augmentation.FDM.image-statistics-matching-master import
#import ot


class DataAugmentation:

    def __init__(self, image, bbox, format, p):
        self.image = image
        self.bbox = bbox
        self.format = format
        self.p = p

    # Albumentaion -> Pixel-level transforms
    # the transform will change only the input image and return any other input targets such as masks, bounding boxes, or keypoints unchanged
    def Blur(self, blur_limit=7, always_apply=False, p=1.0):
        """
        Blur the input image using a random-sized kernel.

        parameters
        ----------
            blur_limit: int, [int, int]
                maximum kernel size for blurring the input image. Should be in range [3, inf). Default: (3, 7).
            p: float
                probability of applying the transform. Default: 0.5.

        Targets
        -------
            image

        Image types
        -----------
            uint8, float32
        """
        transform = A.Blur(blur_limit, p)
        augmented_image = transform(image=self.image)['image']
        self.image = augmented_image

        return self

    def Normalize(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0):
        """
        Normalization is applied by the formula: `img = (img - mean * max_pixel_value) / (std * max_pixel_value)`

        parameters
        ----------
            mean: float, list of float
                mean values
            std: float, list of float
                std values
            max_pixel_value: float
                maximum possible pixel value
            p: float
                probability of applying the transform. Default: 1,0.

        Targets
        -------
            image

        Image types
        -----------
            uint8, float32
        """

        transform = A.Normalize(mean, std, max_pixel_value, p)
        augmented_image = transform(image=self.image)['image']
        self.image = augmented_image

        return self

    def RandomGamma(self, gamma_limit=(80, 120), eps=None, always_apply=False, p=1.0):
        """
        parameters
        ----------
            gamma_limit: float or [float, float]
                If gamma_limit is a single float value, the range will be (-gamma_limit, gamma_limit). Default: (80, 120).
            eps: Deprecated
                defualt: None
            p: float
                probability of applying the transform. Default: 0.5.

        Targets
        -------
            image

        Image types
        -----------
            uint8, float32
        """

        transform = A.RandomGamma(gamma_limit, eps, p)
        augmented_image = transform(image=self.image)['image']
        self.image = augmented_image

        return self

    def HueSaturationValue(self, hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, always_apply=False, p=1.0):
        """
        Randomly change hue, saturation and value of the input image.

        parameters
        ---------
            hue_shift_limit: [int, int] or int
                range for changing hue. If hue_shift_limit is a single int, the range will be (-hue_shift_limit, hue_shift_limit). Default: (-20, 20).
            sat_shift_limit: [int, int] or int
                range for changing saturation. If sat_shift_limit is a single int, the range will be (-sat_shift_limit, sat_shift_limit). Default: (-30, 30).
            val_shift_limit: [int, int] or int
                range for changing value. If val_shift_limit is a single int, the range will be (-val_shift_limit, val_shift_limit). Default: (-20, 20).
            p: float 
                probability of applying the transform. Default: 0.5.

        Targets
        -------
            image

        Image types
        -----------
            uint8, float32
        """

        transform = A.HueSaturationValue(hue_shift_limit, sat_shift_limit, val_shift_limit, p)
        augmented_image = transform(image=self.image)['image']
        self.image = augmented_image

        return self

    def RGBShift(self, r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, always_apply=False, p=1.0):
        """
        Randomly shift values for each channel of the input RGB image.

        parameters
        ---------
            r_shift_limit: [int, int] or int
                range for changing values for the red channel. If r_shift_limit is a single
                int, the range will be (-r_shift_limit, r_shift_limit). Default: (-20, 20).
            g_shift_limit: [int, int] or int
                range for changing values for the green channel. If g_shift_limit is a
                single int, the range  will be (-g_shift_limit, g_shift_limit). Default: (-20, 20).
            b_shift_limit: [int, int] or int
                range for changing values for the blue channel. If b_shift_limit is a single
                int, the range will be (-b_shift_limit, b_shift_limit). Default: (-20, 20).
            p: float
                probability of applying the transform. Default: 0.5.
        Targets
        -------
            image

        Image types
        -----------
            uint8, float32
        """

        transform = A.RGBShift(r_shift_limit, g_shift_limit, b_shift_limit, p)
        augmented_image = transform(image=self.image)['image']
        self.image = augmented_image

        return self

    def RandomBrightness(self, limit=0.2, always_apply=False, p=1.0):
        """
        Randomly change brightness of the input image.

        parameters
        ---------
            limit: [float, float] or float
                factor range for changing brightness.
                If limit is a single float, the range will be (-limit, limit). Default: (-0.2, 0.2).
            p: float 
                probability of applying the transform. Default: 0.5.

        Targets
        -------
            image

        Image types
        -----------
            uint8, float32
        """

        transform = A.RandomBrightness(limit, p)
        augmented_image = transform(image=self.image)['image']
        self.image = augmented_image

        return self

    def RandomContrast(self, limit=0.2, always_apply=False, p=1.0):
        """
        Randomly change contrast of the input image.

        parameters
        ---------
            limit: [float, float] or float
                factor range for changing contrast.
                If limit is a single float, the range will be (-limit, limit). Default: (-0.2, 0.2).
            p: float
                probability of applying the transform. Default: 0.5.

        Targets
        -------
            image

        Image types
        -----------
            uint8, float32
        """

        transform = A.RandomContrast(limit, p)
        augmented_image = transform(image=self.image)['image']
        self.image = augmented_image

        return self

    def MotionBlur(self, blur_limit=7, always_apply=False, p=1.0):
        """
        Apply motion blur to the input image using a random-sized kernel.

        parameters
        ---------
            blur_limit: int
                maximum kernel size for blurring the input image.
                Should be in range [3, inf). Default: (3, 7).
            p: float
                probability of applying the transform. Default: 0.5.

        Targets
        -------
            image

        Image types
        -----------
            uint8, float32
        """

        transform = A.MotionBlur(blur_limit, p)
        augmented_image = transform(image=self.image)['image']
        self.image = augmented_image

        return self

    def MedianBlur(self, blur_limit=7, always_apply=False, p=1.0):
        """
        Blur the input image using a median filter with a random aperture linear size.

        parameters
        ---------
            blur_limit: int
                maximum aperture linear size for blurring the input image.
                Must be odd and in range [3, inf). Default: (3, 7).
            p: float
                probability of applying the transform. Default: 0.5.

        Targets
        -------
            image
        
        Image types
        -----------
            uint8, float32
        """

        transform = A.MedianBlur(blur_limit, p)
        augmented_image = transform(image=self.image)['image']
        self.image = augmented_image

        return self

    def GaussianBlur(self, blur_limit=(3, 7), sigma_limit=0, always_apply=False, p=1.0):
        """
        Blur the input image using a Gaussian filter with a random kernel size.

        parameters
        ---------
            blur_limit: int, [int, int]
                maximum Gaussian kernel size for blurring the input image.
                Must be zero or odd and in range [0, inf). If set to 0 it will be computed from sigma
                as `round(sigma * (3 if img.dtype == np.uint8 else 4) * 2 + 1) + 1`.
                If set single value `blur_limit` will be in range (0, blur_limit).
                Default: (3, 7).
            sigma_limit: float, [float, float]
                Gaussian kernel standard deviation. Must be in range [0, inf).
                If set single value `sigma_limit` will be in range (0, sigma_limit).
                If set to 0 sigma will be computed as `sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8`. Default: 0.
            p: float
                probability of applying the transform. Default: 0.5.

        Targets
        -------
            image
        
        Image types
        -----------
            uint8, float32
        """
        transform = A.GaussianBlur(blur_limit, sigma_limit, p)
        augmented_image = transform(image=self.image)['image']
        self.image = augmented_image

        return self

    def GaussNoise(self, var_limit=(10.0, 50.0), mean=0, per_channel=True, always_apply=False, p=1.0):
        """
        Apply gaussian noise to the input image.

        parameters
        ---------
            var_limit: [float, float] or float
                variance range for noise. If var_limit is a single float, the range
                will be (0, var_limit). Default: (10.0, 50.0).
            mean: float
                mean of the noise. Default: 0
            per_channel: bool
                if set to True, noise will be sampled for each channel independently.
                Otherwise, the noise will be sampled once for all channels. Default: True
            p: float
                probability of applying the transform. Default: 0.5.

        Targets
        -------
            image
        
        Image types
        -----------
            uint8, float32
        """

        transform = A.GaussNoise(var_limit, mean, per_channel, p)
        augmented_image = transform(image=self.image)['image']
        self.image = augmented_image

        return self

    def GlassBlur(self, sigma=0.7, max_delta=4, iterations=2, mode='fast', always_apply=False, p=1.0):
        """
        Apply glass noise to the input image.

        parameters
        ---------
            sigma: float
                standard deviation for Gaussian kernel.
            max_delta: int
                max distance between pixels which are swapped.
            iterations: int
                number of repeats.
                Should be in range [1, inf). Default: (2).
            mode: str
                mode of computation: fast or exact. Default: "fast".
            p:float
                probability of applying the transform. Default: 0.5.

        Targets
        -------
            image
        
        Image types
        -----------
            uint8, float32
        Reference:
        |  https://arxiv.org/abs/1903.12261
        |  https://github.com/hendrycks/robustness/blob/master/ImageNet-C/create_c/make_imagenet_c.py
        """
        transform = A.GaussianBlur(sigma, max_delta, iterations, mode, p)
        augmented_image = transform(image=self.image)['image']
        self.image = augmented_image

        return self

    def CLAHE(self, clip_limit=4.0, tile_grid_size=(8, 8), always_apply=False, p=1.0):
        """
        Apply Contrast Limited Adaptive Histogram Equalization to the input image.

        parameters
        ---------
            clip_limit: float or [float, float]
                upper threshold value for contrast limiting.
                If clip_limit is a single float value, the range will be (1, clip_limit). Default: (1, 4).
            tile_grid_size: [int, int]
                size of grid for histogram equalization. Default: (8, 8).
            p: float
                probability of applying the transform. Default: 0.5.

        Targets
        -------
            image
        
        Image types
        -----------
            uint8
        """

        transform = A.CLAHE(clip_limit, tile_grid_size, p)
        augmented_image = transform(image=self.image)['image']
        self.image = augmented_image

        return self

    def ChannelShuffle(self, always_apply=False, p=1.0):
        """
        Randomly rearrange channels of the input RGB image.

        parameters
        ---------
            p: float
                probability of applying the transform. Default: 0.5.

        Targets
        -------
            image
        
        Image types
        -----------
            uint8, float32
        """

        transform = A.ChannelShuffle(p)
        augmented_image = transform(image=self.image)['image']
        self.image = augmented_image

        return self

    def InvertImg(self, always_apply=False, p=1.0):
        """
        Invert the input image by subtracting pixel values from 255.

        parameters
        ---------
            p: float
                probability of applying the transform. Default: 0.5.

        Targets
        -------
            image
        
        Image types
        -----------
            uint8
        """

        transform = A.InvertImg(p)
        augmented_image = transform(image=self.image)['image']
        self.image = augmented_image

        return self

    def ToGray(self, always_apply=False, p=1.0):
        """
        Convert the input RGB image to grayscale. If the mean pixel value for the resulting image is greater
        than 127, invert the resulting grayscale image.

        parameters
        ---------
            p: float
                probability of applying the transform. Default: 0.5.

        Targets
        -------
            image
        
        Image types
        -----------
            uint8, float32
        """

        transform = A.ToGray(p)
        augmented_image = transform(image=self.image)['image']
        self.image = augmented_image

        return self

    def ToSepia(self, always_apply=False, p=1.0):
        """
        Applies sepia filter to the input RGB image

        parameters
        ---------
            p: float
                probability of applying the transform. Default: 0.5.

        Targets
        -------
            image
        
        Image types
        -----------
            uint8, float32
        """

        transform = A.ToSepia(p)
        augmented_image = transform(image=self.image)['image']
        self.image = augmented_image

        return self

    def JpegCompression(self, quality_lower=99, quality_upper=100, always_apply=False, p=1.0):
        """
        Decrease Jpeg compression of an image.

        parameters
        ---------
            quality_lower: float
                lower bound on the jpeg quality. Should be in [0, 100] range
            quality_upper: float
                upper bound on the jpeg quality. Should be in [0, 100] range

        Targets
        -------
            image
        
        Image types
        -----------
            uint8, float32
        """

        transform = A.JpegCompression(quality_lower, quality_upper, p)
        augmented_image = transform(image=self.image)['image']
        self.image = augmented_image

        return self

    def ImageCompression(self, quality_lower=99, quality_upper=100, compression_type=0, always_apply=False, p=1.0):
        """
        Decrease Jpeg, WebP compression of an image.

        parameters
        ---------
            quality_lower: float
                lower bound on the image quality.
                Should be in [0, 100] range for jpeg and [1, 100] for webp.
            quality_upper: float
                upper bound on the image quality.
                Should be in [0, 100] range for jpeg and [1, 100] for webp.
            compression_type: ImageCompressionType
                should be 0 for ImageCompressionType.JPEG or 1 for ImageCompressionType.WEBP.
                Default: 0

        Targets
        -------
            image
        
        Image types
        -----------
            uint8, float32
        """

        transform = A.ImageCompression(quality_lower, quality_upper, compression_type, p)
        augmented_image = transform(image=self.image)['image']
        self.image = augmented_image

        return self

    def ToFloat(self, max_value=None, p=1.0):
        """
        Divide pixel values by `max_value` to get a float32 output array where all values lie in the range [0, 1.0].
        If `max_value` is None the transform will try to infer the maximum value by inspecting the data type of the input
        image.
        
        parameters
        ---------
            max_value: float
                maximum possible input value. Default: None.
            p: float
                probability of applying the transform. Default: 1.0.

        Targets
        -------
            image
        
        Image types
        -----------
            any type
        """

        transform = A.ToFloat(max_value, p)
        augmented_image = transform(image=self.image)['image']
        self.image = augmented_image

        return self

    def FromFloat(self, dtype='uint16', max_value=None, p=1.0):
        """
        Take an input array where all values should lie in the range [0, 1.0], multiply them by `max_value` and then
        cast the resulted value to a type specified by `dtype`. If `max_value` is None the transform will try to infer
        the maximum value for the data type from the `dtype` argument.
        This is the inverse transform for :class:`~albumentations.augmentations.transforms.ToFloat`.

        parameters
        ---------
            max_value: float
                maximum possible input value. Default: None.
            dtype: string or numpy data type
                of the output. See the `'Data types' page from the NumPy docs`_.
                Default: 'uint16'.
            p: float
                probability of applying the transform. Default: 1.0.

        Targets
        -------
            image
        
        Image types
        -----------
            float32
        .. _'Data types' page from the NumPy docs:
        https://docs.scipy.org/doc/numpy/user/basics.types.html
        """

        transform = A.FromFloat(max_value, dtype, p)
        augmented_image = transform(image=self.image)['image']
        self.image = augmented_image

        return self

    def RandomBrightnessContrast(self, brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=True,
                                 always_apply=False, p=1.0):
        """
        Randomly change brightness and contrast of the input image.

        parameters
        ---------
            brightness_limit: [float, float] or float
                factor range for changing brightness.
                If limit is a single float, the range will be (-limit, limit). Default: (-0.2, 0.2).
            contrast_limit: [float, float] or float
                factor range for changing contrast.
                If limit is a single float, the range will be (-limit, limit). Default: (-0.2, 0.2).
            brightness_by_max: bool
                If True adjust contrast by image dtype maximum,
                else adjust contrast by image mean.
            p: float
                probability of applying the transform. Default: 0.5.

        Targets
        -------
            image
        
        Image types
        -----------
            uint8, float32
        """

        transform = A.RandomBrightnessContrast(brightness_limit, contrast_limit, brightness_by_max, p)
        augmented_image = transform(image=self.image)['image']
        self.image = augmented_image

        return self

    def RandomSnow(self, snow_point_lower=0.1, snow_point_upper=0.3, brightness_coeff=2.5, always_apply=False, p=1.0):
        """
        Bleach out some pixel values simulating snow.
        From https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library

        parameters
        ---------
            snow_point_lower: float
                lower_bond of the amount of snow. Should be in [0, 1] range
            snow_point_upper: float
                upper_bond of the amount of snow. Should be in [0, 1] range
            brightness_coeff: float
                larger number will lead to a more snow on the image. Should be >= 0

        Targets
        -------
            image
        
        Image types
        -----------
            uint8, float32
        """

        transform = A.RandomSnow(snow_point_lower, snow_point_upper, brightness_coeff, p)
        augmented_image = transform(image=self.image)['image']
        self.image = augmented_image

        return self

    def RandomRain(self, slant_lower=-10, slant_upper=10, drop_length=20, drop_width=1, drop_color=(200, 200, 200),
                   blur_value=7, brightness_coefficient=0.7, rain_type=None, always_apply=False, p=1.0):
        """
        Adds rain effects.
        From https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library

        parameters
        ---------
            slant_lower: int
                should be in range [-20, 20].
            slant_upper: int
                should be in range [-20, 20].
            drop_length: int
                should be in range [0, 100].
            drop_width: int
                should be in range [1, 5].
            drop_color: list of R G B
                rain lines color.
            blur_value: int
                rainy view are blurry
            brightness_coefficient: float
                rainy days are usually shady. Should be in range [0, 1].
            rain_type: str
                One of [None, "drizzle", "heavy", "torrestial"]

        Targets
        -------
            image
        
        Image types
        -----------
            uint8, float32
        """

        transform = A.RandomRain(slant_lower, slant_upper, drop_length, drop_width, drop_color, blur_value,
                                 brightness_coefficient, rain_type, p)
        augmented_image = transform(image=self.image)['image']
        self.image = augmented_image

        return self

    def RandomFog(self, fog_coef_lower=0.3, fog_coef_upper=1, alpha_coef=0.08, always_apply=False, p=1.0):
        """
        Simulates fog for the image
        From https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library

        parameters
        ---------
            fog_coef_lower: float
                lower limit for fog intensity coefficient. Should be in [0, 1] range.
            fog_coef_upper: float
                upper limit for fog intensity coefficient. Should be in [0, 1] range.
            alpha_coef: float
                transparency of the fog circles. Should be in [0, 1] range.

        Targets
        -------
            image
        
        Image types
        -----------
            uint8, float32
        """

        transform = A.RandomFog(fog_coef_lower, fog_coef_upper, alpha_coef, p)
        augmented_image = transform(image=self.image)['image']
        self.image = augmented_image

        return self

    def RandomSunFlare(self, flare_roi=(0, 0, 1, 0.5), angle_lower=0, angle_upper=1, num_flare_circles_lower=6,
                       num_flare_circles_upper=10, src_radius=400, src_color=(255, 255, 255), always_apply=False,
                       p=1.0):
        """
        Simulates Sun Flare for the image
        From https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library

        parameters
        ---------
            flare_roi: (float, float, float, float) 
                region of the image where flare will
                appear (x_min, y_min, x_max, y_max). All values should be in range [0, 1].
            angle_lower: float
                should be in range [0, `angle_upper`].
            angle_upper: float
                should be in range [`angle_lower`, 1].
            num_flare_circles_lower: int
                lower limit for the number of flare circles.
                Should be in range [0, `num_flare_circles_upper`].
            num_flare_circles_upper:int
                upper limit for the number of flare circles.
                Should be in range [`num_flare_circles_lower`, inf].
            src_radius: int
            src_color: (int, int, int)
                color of the flare

        Targets
        -------
            image
        
        Image types
        -----------
            uint8, float32
        """

        transform = A.RandomSunFlare(flare_roi, angle_upper, num_flare_circles_lower, num_flare_circles_upper,
                                     src_radius, src_color, p)
        augmented_image = transform(image=self.image)['image']
        self.image = augmented_image

        return self

    def RandomShadow(self, shadow_roi=(0, 0.5, 1, 1), num_shadows_lower=1, num_shadows_upper=2, shadow_dimension=5,
                     always_apply=False, p=1.0):
        """
        Simulates shadows for the image
        From https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library

        parameters
        ---------
            shadow_roi: float, float, float, float
                region of the image where shadows
                will appear (x_min, y_min, x_max, y_max). All values should be in range [0, 1].
            num_shadows_lower: int
                Lower limit for the possible number of shadows.
                Should be in range [0, `num_shadows_upper`].
            num_shadows_upper: int
                Lower limit for the possible number of shadows.
                Should be in range [`num_shadows_lower`, inf].
            shadow_dimension: int
                number of edges in the shadow polygons

        Targets
        -------
            image

        Image types
        -----------
            uint8, float32
        """

        transform = A.RandomShadow(shadow_roi, num_shadows_lower, num_shadows_upper, shadow_dimension, p)
        augmented_image = transform(image=self.image)['image']
        self.image = augmented_image

        return self

    def RandomToneCurve(self, scale=0.1, always_apply=False, p=1.0):
        """
        Randomly change the relationship between bright and dark areas of the image by manipulating its tone curve.

        parameters
        ---------
            scale: float
                standard deviation of the normal distribution.
                Used to sample random distances to move two control points that modify the image's curve.
                Values should be in range [0, 1]. Default: 0.1

        Targets
        -------
            image
        
        Image types
        -----------
            uint8
        """

        transform = A.RandomToneCurve(scale, p)
        augmented_image = transform(image=self.image)['image']
        self.image = augmented_image

        return self

    def ISONoise(self, color_shift=(0.01, 0.05), intensity=(0.1, 0.5), always_apply=False, p=1.0):
        """
        Apply camera sensor noise.

        parameters
        ---------
            color_shift: [float, float]
                variance range for color hue change.
                Measured as a fraction of 360 degree Hue angle in HLS colorspace.
            intensity: [float, float]
                Multiplicative factor that control strength
                of color and luminace noise.
            p: float
                probability of applying the transform. Default: 0.5.

        Targets
        -------
            image
        
        Image types
        -----------
            uint8
        """

        transform = A.ISONoise(color_shift, intensity, p)
        augmented_image = transform(image=self.image)['image']
        self.image = augmented_image

        return self

    def Solarize(self, threshold=128, always_apply=False, p=1.0):
        """
        Invert all pixel values above a threshold.

        parameters
        ---------
            threshold: [int, int] or int, or [float, float] or float
                range for solarizing threshold.
                If threshold is a single value, the range will be [threshold, threshold]. Default: 128.
            p: float
                probability of applying the transform. Default: 0.5.

        Targets
        -------
            image

        Image types
        -----------
            any
        """

        transform = A.Solarize(threshold, p)
        augmented_image = transform(image=self.image)['image']
        self.image = augmented_image

        return self

    def Equalize(self, mode='cv', by_channels=True, mask=None, mask_params=(), always_apply=False, p=1.0):
        """Equalize the image histogram.

        parameters
        ---------
            mode: str
                {'cv', 'pil'}. Use OpenCV or Pillow equalization method.
            by_channels: bool
                If True, use equalization by channels separately,
                else convert image to YCbCr representation and use equalization by `Y` channel.
            mask: np.ndarray, callable
                If given, only the pixels selected by
                the mask are included in the analysis. Maybe 1 channel or 3 channel array or callable.
                Function signature must include `image` argument.
            mask_params: list of str
                Params for mask function.

        Targets
        -------
            image
        
        Image types
        -----------
            uint8
        """

        transform = A.Equalize(mode, by_channels, mask, mask_params, p)
        augmented_image = transform(image=self.image)['image']
        self.image = augmented_image

        return self

    def Posterize(self, num_bits=4, always_apply=False, p=1.0):
        """
        Reduce the number of bits for each color channel.

        parameters
        ---------
            num_bits: [int, int] or int, or list of ints [r, g, b], or list of ints [[r1, r1], [g1, g2], [b1, b2]]
                number of high bits.
                If num_bits is a single value, the range will be [num_bits, num_bits].
                Must be in range [0, 8]. Default: 4.
            p: float
                probability of applying the transform. Default: 0.5.

        Targets
        -------
        image
        Image types
        -----------
            uint8
        """

        transform = A.Posterize(num_bits, p)
        augmented_image = transform(image=self.image)['image']
        self.image = augmented_image

        return self

    def Downscale(self, scale_min=0.25, scale_max=0.25, interpolation=0, always_apply=False, p=1.0):
        """
        Decreases image quality by downscaling and upscaling back.

        parameters
        ---------
            scale_min: float
                lower bound on the image scale. Should be < 1.
            scale_max: float
                lower bound on the image scale. Should be .
            interpolation:
                cv2 interpolation method. cv2.INTER_NEAREST by default

        Targets
        -------
            image

        Image types
        -----------
            uint8, float32
        """

        transform = A.Downscale(scale_min, scale_max, interpolation, p)
        augmented_image = transform(image=self.image)['image']
        self.image = augmented_image

        return self

    def MultiplicativeNoise(self, multiplier=(0.9, 1.1), per_channel=False, elementwise=False, always_apply=False,
                            p=1.0):
        """
        Multiply image to random number or array of numbers.

        parameters
        ---------
            multiplier: float or tuple of floats
                If single float image will be multiplied to this number.
                If tuple of float multiplier will be in range `[multiplier[0], multiplier[1])`. Default: (0.9, 1.1).
            per_channel: bool 
                If `False`, same values for all channels will be used.
                If `True` use sample values for each channels. Default False.
            elementwise: bool
                If `False` multiply multiply all pixels in an image with a random value sampled once.
                If `True` Multiply image pixels with values that are pixelwise randomly sampled. Defaule: False.

        Targets
        -------
            image

        Image types
        -----------
            Any
        """

        transform = A.MultiplicativeNoise(multiplier, per_channel, elementwise, p)
        augmented_image = transform(image=self.image)['image']
        self.image = augmented_image

        return augmented_image

    def FancyPCA(self, alpha=0.1, always_apply=False, p=1.0):
        """
        Augment RGB image using FancyPCA from Krizhevsky's paper
        "ImageNet Classification with Deep Convolutional Neural Networks"
        
        parameters
        ---------
            alpha:float
                how much to perturb/scale the eigen vecs and vals.
                scale is samples from gaussian distribution (mu=0, sigma=alpha)

        Targets
        -------
            image

        Image types
        -----------
            3-channel uint8 images only

        Credit:
            http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
            https://deshanadesai.github.io/notes/Fancy-PCA-with-Scikit-Image
            https://pixelatedbrian.github.io/2018-04-29-fancy_pca/
        """

        transform = A.FancyPCA(alpha, p)
        augmented_image = transform(image=self.image)['image']
        self.image = augmented_image

        return self

    def ColorJitter(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, always_apply=False, p=1.0):
        """
        Randomly changes the brightness, contrast, and saturation of an image. Compared to ColorJitter from torchvision,
        this transform gives a little bit different results because Pillow (used in torchvision) and OpenCV (used in
        Albumentations) transform an image to HSV format by different formulas. Another difference - Pillow uses uint8
        overflow, but we use value saturation.

        parameters
        ---------
            brightness: float or tuple of float (min, max)
                How much to jitter brightness.
                brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
                or the given [min, max]. Should be non negative numbers.
            contrast: float or tuple of float (min, max)
                How much to jitter contrast.
                contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
                or the given [min, max]. Should be non negative numbers.
            saturation: float or tuple of float (min, max)
                How much to jitter saturation.
                saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
                or the given [min, max]. Should be non negative numbers.
            hue: float or tuple of float (min, max)
                How much to jitter hue.
                hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
                Should have 0 <= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
        Targets
        -------
            image
        
        Image types
        -----------
            uint8, float32            
        """

        transform = A.ColorJitter(brightness, contrast, saturation, hue, p)
        augmented_image = transform(image=self.image)['image']
        self.image = augmented_image

        return self

    def Sharpen(self, alpha=(0.2, 0.5), lightness=(0.5, 1.0), always_apply=False, p=1.0):
        """
        Sharpen the input image and overlays the result with the original image.

        parameters
        ---------
            alpha: [float, float]
                range to choose the visibility of the sharpened image. At 0, only the original image is
                visible, at 1.0 only its sharpened version is visible. Default: (0.2, 0.5).
            lightness: [float, float]
                range to choose the lightness of the sharpened image. Default: (0.5, 1.0).
            p: float
                probability of applying the transform. Default: 0.5.

        Targets
        ------- ints
            image

        """

        transform = A.Sharpen(alpha, lightness, p)
        augmented_image = transform(image=self.image)['image']
        self.image = augmented_image

        return self

    def Emboss(self, alpha=(0.2, 0.5), strength=(0.2, 0.7), always_apply=False, p=1.0):
        """
        Emboss the input image and overlays the result with the original image.

        parameters
        ---------
            alpha: [float, float]
                range to choose the visibility of the embossed image. At 0, only the original image is
                visible,at 1.0 only its embossed version is visible. Default: (0.2, 0.5).
            strength: [float, float]
                strength range of the embossing. Default: (0.2, 0.7).
            p: float
                probability of applying the transform. Default: 0.5.
    
        Targets
        -------
            image

        """

        transform = A.Emboss(alpha, strength, p)
        augmented_image = transform(image=self.image)['image']
        self.image = augmented_image

        return self

    def Superpixels(self, p_replace=0.1, n_segments=100, max_size=128, interpolation=1, always_apply=False, p=1.0):
        """
        Transform images partially/completely to their superpixel representation.
        This implementation uses skimage's version of the SLIC algorithm.

        parameters
        ---------
            p_replace: float or tuple of float
                Defines for any segment the probability that the pixels within that
                segment are replaced by their average color (otherwise, the pixels are not changed).
                Examples:
                    * A probability of ``0.0`` would mean, that the pixels in no
                    segment are replaced by their average color (image is not
                    changed at all).
                    * A probability of ``0.5`` would mean, that around half of all
                    segments are replaced by their average color.
                    * A probability of ``1.0`` would mean, that all segments are
                    replaced by their average color (resulting in a voronoi
                    image).
                Behaviour based on chosen data types for this parameter:
                    * If a ``float``, then that ``flat`` will always be used.
                    * If ``tuple`` ``(a, b)``, then a random probability will be
                    sampled from the interval ``[a, b]`` per image.
            n_segments: int, or tuple of int
                Rough target number of how many superpixels to generate (the algorithm
                may deviate from this number). Lower value will lead to coarser superpixels.
                Higher values are computationally more intensive and will hence lead to a slowdown
                * If a single ``int``, then that value will always be used as the
                number of segments.
                * If a ``tuple`` ``(a, b)``, then a value from the discrete
                interval ``[a..b]`` will be sampled per image.
            max_size: int or None
                Maximum image size at which the augmentation is performed.
                If the width or height of an image exceeds this value, it will be
                downscaled before the augmentation so that the longest side matches `max_size`.
                This is done to speed up the process. The final output image has the same size as the input image.
                Note that in case `p_replace` is below ``1.0``,
                the down-/upscaling will affect the not-replaced pixels too.
                Use ``None`` to apply no down-/upscaling.
            interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm. Should be one of:
                cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
                Default: cv2.INTER_LINEAR.
            p: float
                probability of applying the transform. Default: 0.5.

        Targets
        -------
            image

        """

        transform = A.Superpixels(p_replace, n_segments, max_size, interpolation, p)
        augmented_image = transform(image=self.image)['image']
        self.image = augmented_image

        return self

    def TemplateTransform(self, templates=0.5, img_weight=0.5, template_weight=0.5, template_transform=None, name=None,
                          always_apply=False, p=1.0):
        """
        Apply blending of input image with specified templates

        parameters
        ---------
            templates: numpy array or list of numpy arrays
                Images as template for transform.
            img_weight: [float, float] or float
                If single float will be used as weight for input image.
                If tuple of float img_weight will be in range `[img_weight[0], img_weight[1])`. Default: 0.5.
            template_weight: [float, float] or float
                If single float will be used as weight for template.
                If tuple of float template_weight will be in range `[template_weight[0], template_weight[1])`.
                Default: 0.5.
            template_transform:
                transformation object which could be applied to template,
                must produce template the same size as input image.
            name: str (Optional)
                Name of transform, used only for deserialization.
            p: float
                probability of applying the transform. Default: 0.5.
        Targets
        -------
            image

        Image types
        -----------
            uint8, float32
        """

        transform = A.TemplateTransform(templates, img_weight, template_transform, name, p)
        augmented_image = transform(image=self.image)['image']
        self.image = augmented_image

        return self

    def RingingOvershoot(self, blur_limit=(7, 15), cutoff=(0.7853981633974483, 1.5707963267948966), always_apply=False,
                         p=1.0):
        """
        Create ringing or overshoot artefacts by conlvolving image with 2D sinc filter.

        parameters
        ---------
            blur_limit: int, [int, int]
                maximum kernel size for sinc filter.
                Should be in range [3, inf). Default: (7, 15).
            cutoff: float, [float, float]
                range to choose the cutoff frequency in radians.
                Should be in range (0, np.pi)
                Default: (np.pi / 4, np.pi / 2).
            p: float
                probability of applying the transform. Default: 0.5.

        Reference:
            dsp.stackexchange.com/questions/58301/2-d-circularly-symmetric-low-pass-filter
            https://arxiv.org/abs/2107.10833

        Targets
        -------
            image

        """

        transform = A.RingingOvershoot(blur_limit, cutoff, p)
        augmented_image = transform(image=self.image)['image']
        self.image = augmented_image

        return self

    def UnsharpMask(self, blur_limit=(3, 7), sigma_limit=0.0, alpha=(0.2, 0.5), threshold=10, always_apply=False,
                    p=1.0):
        """
        Sharpen the input image using Unsharp Masking processing and overlays the result with the original image.

        parameters
        ---------
            blur_limit: int, [int, int]
                maximum Gaussian kernel size for blurring the input image.
                Must be zero or odd and in range [0, inf). If set to 0 it will be computed from sigma
                as `round(sigma * (3 if img.dtype == np.uint8 else 4) * 2 + 1) + 1`.
                If set single value `blur_limit` will be in range (0, blur_limit).
                Default: (3, 7).
            sigma_limit: float, [float, float]
                Gaussian kernel standard deviation. Must be in range [0, inf).
                If set single value `sigma_limit` will be in range (0, sigma_limit).
                If set to 0 sigma will be computed as `sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8`. Default: 0.
            alpha: float, [float, float]
                range to choose the visibility of the sharpened image.
                At 0, only the original image is visible, at 1.0 only its sharpened version is visible.
                Default: (0.2, 0.5).
            threshold: int
                Value to limit sharpening only for areas with high pixel difference between original image
                and it's smoothed version. Higher threshold means less sharpening on flat areas.
                Must be in range [0, 255]. Default: 10.
            p: float
                probability of applying the transform. Default: 0.5.

        Reference
        ---------
            arxiv.org/pdf/2107.10833.pdf

        Targets
        -------
            image

        """

        transform = A.UnsharpMask(blur_limit, sigma_limit, alpha, threshold, p)
        augmented_image = transform(image=self.image)['image']
        self.image = augmented_image

        return self

    def AdvancedBlur(self, blur_limit=(3, 7), sigmaX_limit=(0.2, 1.0), sigmaY_limit=(0.2, 1.0), rotate_limit=90,
                     beta_limit=(0.5, 8.0), noise_limit=(0.9, 1.1), always_apply=False, p=1.0):
        """
        Blur the input image using a Generalized Normal filter with a randomly selected parameters.
        This transform also adds multiplicative noise to generated kernel before convolution.

        parameters
        ---------
            blur_limit:
                maximum Gaussian kernel size for blurring the input image.
                Must be zero or odd and in range [0, inf). If set to 0 it will be computed from sigma
                as `round(sigma * (3 if img.dtype == np.uint8 else 4) * 2 + 1) + 1`.
                If set single value `blur_limit` will be in range (0, blur_limit).
                Default: (3, 7).
            sigmaX_limit:
                Gaussian kernel standard deviation. Must be in range [0, inf).
                If set single value `sigmaX_limit` will be in range (0, sigma_limit).
                If set to 0 sigma will be computed as `sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8`. Default: 0.
            sigmaY_limit:
                Same as `sigmaY_limit` for another dimension.
            rotate_limit:
                Range from which a random angle used to rotate Gaussian kernel is picked.
                If limit is a single int an angle is picked from (-rotate_limit, rotate_limit). Default: (-90, 90).
            beta_limit:
                Distribution shape parameter, 1 is the normal distribution. Values below 1.0 make distribution
                tails heavier than normal, values above 1.0 make it lighter than normal. Default: (0.5, 8.0).
            noise_limit: 
                Multiplicative factor that control strength of kernel noise. Must be positive and preferably
                centered around 1.0. If set single value `noise_limit` will be in range (0, noise_limit).
                Default: (0.75, 1.25).
            p: float
                probability of applying the transform. Default: 0.5.

        Reference
        ---------
            https://arxiv.org/abs/2107.10833

        Targets
        -------
            image

        Image types
        -----------
            uint8, float32
        """

        transform = A.AdvancedBlur(blur_limit, sigmaX_limit, sigmaY_limit, rotate_limit, beta_limit, noise_limit, p)
        augmented_image = transform(image=self.image)['image']
        self.image = augmented_image

        return self

    def ChannelDropout(self, channel_drop_range=(1, 1), fill_value=0, always_apply=False, p=1.0):
        """
        Randomly Drop Channels in the input Image.

        parameters
        ---------
        channel_drop_range: int, int
            range from which we choose the number of channels to drop.
        fill_value: int, float
            pixel value for the dropped channel.
        p: float
            probability of applying the transform. Default: 0.5.
        
        Targets
        -------
            image
        
        Image types
        -----------
            uint8, uint16, unit32, float32
        """

        transform = A.ChannelDropout(channel_drop_range, fill_value, always_apply, p)
        augmented_image = transform(image=self.image)['image']
        self.image = augmented_image

        return self

    # Albumentaion -> Spatial-level transforms
    # If you try to apply a spatial-level transform to an unsupported target, Albumentations will raise an error
    def Affine(self, scale=None, translate_percent=None, translate_px=None, rotate=None, shear=None, interpolation=1,
               mask_interpolation=0, cval=0, cval_mask=0, mode=0, fit_output=False, always_apply=False, p=1.0):
        """
        Augmentation to apply affine transformations to images.
        This is mostly a wrapper around the corresponding classes and functions in OpenCV.
        Affine transformations involve:
            - Translation ("move" image on the x-/y-axis)
            - Rotation
            - Scaling ("zoom" in/out)
            - Shear (move one side of the image, turning a square into a trapezoid)
        All such transformations can create "new" pixels in the image without a defined content, e.g.
        if the image is translated to the left, pixels are created on the right.
        A method has to be defined to deal with these pixel values.
        The parameters `cval` and `mode` of this class deal with this.
        Some transformations involve interpolations between several pixels
        of the input image to generate output pixel values. The parameters `interpolation` and
        `mask_interpolation` deals with the method of interpolation used for this.
        
        parameters
        ---------
            scale: number, tuple of number or dict
                Scaling factor to use, where ``1.0`` denotes "no change" and
                ``0.5`` is zoomed out to ``50`` percent of the original size.
                    * If a single number, then that value will be used for all images.
                    * If a tuple ``(a, b)``, then a value will be uniformly sampled per image from the interval ``[a, b]``.
                    That value will be used identically for both x- and y-axis.
                    * If a dictionary, then it is expected to have the keys ``x`` and/or ``y``.
                    Each of these keys can have the same values as described above.
                    Using a dictionary allows to set different values for the two axis and sampling will then happen
                    *independently* per axis, resulting in samples that differ between the axes.
            translate_percent: None, number, tuple of number or dict
                Translation as a fraction of the image height/width
                (x-translation, y-translation), where ``0`` denotes "no change"
                and ``0.5`` denotes "half of the axis size".
                    * If ``None`` then equivalent to ``0.0`` unless `translate_px` has a value other than ``None``.
                    * If a single number, then that value will be used for all images.
                    * If a tuple ``(a, b)``, then a value will be uniformly sampled per image from the interval ``[a, b]``.
                    That sampled fraction value will be used identically for both x- and y-axis.
                    * If a dictionary, then it is expected to have the keys ``x`` and/or ``y``.
                    Each of these keys can have the same values as described above.
                    Using a dictionary allows to set different values for the two axis and sampling will then happen
                    *independently* per axis, resulting in samples that differ between the axes.
            translate_px: None, int, tuple of int or dict
                Translation in pixels.
                    * If ``None`` then equivalent to ``0`` unless `translate_percent` has a value other than ``None``.
                    * If a single int, then that value will be used for all images.
                    * If a tuple ``(a, b)``, then a value will be uniformly sampled per image from
                    the discrete interval ``[a..b]``. That number will be used identically for both x- and y-axis.
                    * If a dictionary, then it is expected to have the keys ``x`` and/or ``y``.
                    Each of these keys can have the same values as described above.
                    Using a dictionary allows to set different values for the two axis and sampling will then happen
                    *independently* per axis, resulting in samples that differ between the axes.
            rotate: number or tuple of number
                Rotation in degrees (**NOT** radians), i.e. expected value range is
                around ``[-360, 360]``. Rotation happens around the *center* of the image,
                not the top left corner as in some other frameworks.
                    * If a number, then that value will be used for all images.
                    * If a tuple ``(a, b)``, then a value will be uniformly sampled per image from the interval ``[a, b]``
                    and used as the rotation value.
            shear: number, tuple of number or dict
                Shear in degrees (**NOT** radians), i.e. expected value range is
                around ``[-360, 360]``, with reasonable values being in the range of ``[-45, 45]``.
                    * If a number, then that value will be used for all images as
                    the shear on the x-axis (no shear on the y-axis will be done).
                    * If a tuple ``(a, b)``, then two value will be uniformly sampled per image
                    from the interval ``[a, b]`` and be used as the x- and y-shear value.
                    * If a dictionary, then it is expected to have the keys ``x`` and/or ``y``.
                    Each of these keys can have the same values as described above.
                    Using a dictionary allows to set different values for the two axis and sampling will then happen
                    *independently* per axis, resulting in samples that differ between the axes.
            interpolation: int
                OpenCV interpolation flag.
            mask_interpolation: int
                OpenCV interpolation flag.
            cval: number or sequence of number
                The constant value to use when filling in newly created pixels.
                (E.g. translating by 1px to the right will create a new 1px-wide column of pixels
                on the left of the image).
                The value is only used when `mode=constant`. The expected value range is ``[0, 255]`` for ``uint8`` images.
            cval_mask: number or tuple of number
                Same as cval but only for masks.
            mode: int
                OpenCV border flag.
            fit_output: bool
                Whether to modify the affine transformation so that the whole output image is always
                contained in the image plane (``True``) or accept parts of the image being outside
                the image plane (``False``). This can be thought of as first applying the affine transformation
                and then applying a second transformation to "zoom in" on the new image so that it fits the image plane,
                This is useful to avoid corners of the image being outside of the image plane after applying rotations.
                It will however negate translation and scaling.
            p: float
                probability of applying the transform. Default: 0.5.

        Targets
        -------
            image, mask, keypoints, bboxes
        Image types
        -----------
            uint8, float32
        """

        transform = A.Compose(
            [A.Affine(scale, translate_percent, translate_px, rotate, shear, interpolation, mask_interpolation, cval,
                      cval_mask, mode, fit_output, always_apply, p)],
            bbox_params=A.BboxParams(format=self.format))
        transformed = transform(image=self.image, bboxes=self.bbox)
        augmented_image = transformed['image']
        augmented_bbox = transformed['bboxes']

        return self

    def CenterCrop(self, always_apply=False, p=1.0):
        """
        Crop the central part of the input.

        parameters
        ---------
            height: int
                height of the crop.
            width: int
                width of the crop.
            p: float
                probability of applying the transform. Default: 1.

        Targets
        -------
            image, mask, bboxes, keypoints
        Image types
        -----------
            uint8, float32
        Note:
            It is recommended to use uint8 images as input.
            Otherwise the operation will require internal conversion
            float32 -> uint8 -> float32 that causes worse performance.
        """
        height, width, channel = self.image.shape
        transform = A.Compose(
            [A.CenterCrop(height, width, always_apply, p)],
            bbox_params=A.BboxParams(format=self.format))
        transformed = transform(image=self.image, bboxes=self.bbox)
        augmented_image = transformed['image']
        augmented_bbox = transformed['bboxes']
        self.image = augmented_image
        self.bbox = augmented_bbox

        return self

    def CoarseDropout(self, max_holes=8, max_height=8, max_width=8, min_holes=None, min_height=None, min_width=None,
                      fill_value=0, mask_fill_value=None, always_apply=False, p=1.0):
        """
        CoarseDropout of the rectangular regions in the image.

        parameters
        ---------
            max_holes: int 
                Maximum number of regions to zero out.
            max_height: int, float
                Maximum height of the hole.
                If float, it is calculated as a fraction of the image height.
            max_width: int, float
                Maximum width of the hole.
                If float, it is calculated as a fraction of the image width.
            min_holes: int 
                Minimum number of regions to zero out. If `None`,
                `min_holes` is be set to `max_holes`. Default: `None`.
            min_height: int, float
                Minimum height of the hole. Default: None. If `None`,
                `min_height` is set to `max_height`. Default: `None`.
                If float, it is calculated as a fraction of the image height.
            min_width: int, float
                Minimum width of the hole. If `None`, `min_height` is
                set to `max_width`. Default: `None`.
                If float, it is calculated as a fraction of the image width.
            fill_valueint, float, list of int, list of float
                value for dropped pixels.
            mask_fill_valueint, float, list of int, list of float
                fill value for dropped pixels
                in mask. If `None` - mask is not affected. Default: `None`.
        
        Targets
        -------
            image, mask, keypoints
        Image types
        -----------
            uint8, float32
        Reference:
        |  https://arxiv.org/abs/1708.04552
        |  https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py
        |  https://github.com/aleju/imgaug/blob/master/imgaug/augmenters/arithmetic.py
        """

        transform = A.Compose(
            [A.CoarseDropout(max_holes, max_height, max_width, min_holes, min_height, min_width, fill_value,
                             mask_fill_value, always_apply, p)],
            bbox_params=A.BboxParams(format=self.format))
        transformed = transform(image=self.image, bboxes=self.bbox)
        augmented_image = transformed['image']
        augmented_bbox = transformed['bboxes']
        self.image = augmented_image
        self.bbox = augmented_bbox
        self.image = augmented_image
        self.bbox = augmented_bbox

        return self

    def Crop(self, x_min=0, y_min=0, x_max=1024, y_max=1024, always_apply=False, p=1.0):
        """
        Crop region from image.

        parameters
        ---------
            x_min: int
                Minimum upper left x coordinate.
            y_min: int
                Minimum upper left y coordinate.
            x_max: int
                Maximum lower right x coordinate.
            y_max: int
                Maximum lower right y coordinate.

        Targets
        -------
            image, mask, bboxes, keypoints
        Image types
        -----------
            uint8, float32
        """

        transform = A.Compose(
            [A.Crop(x_min, y_min, x_max, y_max, always_apply, p)],
            bbox_params=A.BboxParams(format=self.format))
        transformed = transform(image=self.image, bboxes=self.bbox)
        augmented_image = transformed['image']
        augmented_bbox = transformed['bboxes']
        self.image = augmented_image
        self.bbox = augmented_bbox

        return self

    def CropNonEmptyMaskIfExists(self, ignore_values=None, ignore_channels=None, always_apply=False, p=1.0):
        """
        Crop area with mask if mask is non-empty, else make random crop.

        parameters
        ---------
            height: int
                vertical size of crop in pixels
            width: int
                horizontal size of crop in pixels
            ignore_values: list of int
                values to ignore in mask, `0` values are always ignored
                (e.g. if background value is 5 set `ignore_values=[5]` to ignore)
            ignore_channels: list of int
                channels to ignore in mask
                (e.g. if background is a first channel set `ignore_channels=[0]` to ignore)
            p: float
                probability of applying the transform. Default: 1.0.
        
        Targets
        -------
            image, mask, bboxes, keypoints
        Image types
        -----------
            uint8, float32
        """

        height, width, channel = self.image.shape
        transform = A.Compose(
            [A.CropNonEmptyMaskIfExists(height, width, ignore_values, ignore_channels, always_apply, p)],
            bbox_params=A.BboxParams(format=self.format))
        transformed = transform(image=self.image, bboxes=self.bbox)
        augmented_image = transformed['image']
        augmented_bbox = transformed['bboxes']
        self.image = augmented_image
        self.bbox = augmented_bbox

        return self

    def ElasticTransform(self, alpha=1, sigma=50, alpha_affine=50, interpolation=1, border_mode=4, value=None,
                         mask_value=None, always_apply=False, approximate=False, same_dxdy=False, p=1.0):
        """
        Elastic deformation of images as described in [Simard2003]_ (with modifications).
        Based on https://gist.github.com/ernestum/601cdf56d2b424757de5
        .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
            Convolutional Neural Networks applied to Visual Document Analysis", in
            Proc. of the International Conference on Document Analysis and
            Recognition, 2003.

        parameters
        ---------
            alpha: float            
            sigma: float
                Gaussian filter parameter.
            alpha_affine: float
                The range will be (-alpha_affine, alpha_affine)
            interpolation: OpenCV flag
                flag that is used to specify the interpolation algorithm. Should be one of:
                cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
                Default: cv2.INTER_LINEAR.
            border_mode: OpenCV flag
                flag that is used to specify the pixel extrapolation method. Should be one of:
                cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE, cv2.BORDER_REFLECT, cv2.BORDER_WRAP, cv2.BORDER_REFLECT_101.
                Default: cv2.BORDER_REFLECT_101
            value: int, float, list of ints, list of float
                padding value if border_mode is cv2.BORDER_CONSTANT.
            mask_value: int, float, list of ints, list of float
                padding value if border_mode is cv2.BORDER_CONSTANT applied for masks.
            approximate: bool 
                Whether to smooth displacement map with fixed kernel size.
                                Enabling this option gives ~2X speedup on large images.
            same_dxdy: bool 
                Whether to use same random generated shift for x and y.
                                Enabling this option gives ~2X speedup.
        
        Targets
        -------
            image, mask
        Image types
        -----------
            uint8, float32
        """

        transform = A.Compose(
            [A.ElasticTransform(alpha, sigma, alpha_affine, interpolation, border_mode, value, mask_value, always_apply,
                                approximate, same_dxdy, p)],
            bbox_params=A.BboxParams(format=self.format))
        transformed = transform(image=self.image, bboxes=self.bbox)
        augmented_image = transformed['image']
        augmented_bbox = transformed['bboxes']
        self.image = augmented_image
        self.bbox = augmented_bbox

        return self

    def Flip(self, p=1.0):
        """
        Flip the input either horizontally, vertically or both horizontally and vertically.

        parameters
        ---------
            p: float
            probability of applying the transform. Default: 0.5.

        Targets
        -------
            image, mask, bboxes, keypoints
        Image types
        -----------
            uint8, float32
        """
        transform = A.Compose(
            [A.Flip(p)],
            bbox_params=A.BboxParams(format=self.format))
        transformed = transform(image=self.image, bboxes=self.bbox)
        augmented_image = transformed['image']
        augmented_bbox = transformed['bboxes']
        self.image = augmented_image
        self.bbox = augmented_bbox

        return self

    def GridDistortion(self, num_steps=5, distort_limit=0.3, interpolation=1, border_mode=4, value=None,
                       mask_value=None, always_apply=False, p=1.0):
        """
        parameters
        ---------
            num_steps: int
                count of grid cells on each side.
            distort_limit
                If distort_limit is a single float, the range
                will be (-distort_limit, distort_limit). Default: (-0.03, 0.03).
            interpolation: OpenCV flag
                flag that is used to specify the interpolation algorithm. Should be one of:
                cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
                Default: cv2.INTER_LINEAR.
            border_mode:  OpenCV flag
                flag that is used to specify the pixel extrapolation method. Should be one of:
                cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE, cv2.BORDER_REFLECT, cv2.BORDER_WRAP, cv2.BORDER_REFLECT_101.
                Default: cv2.BORDER_REFLECT_101
            value: int, float, list of ints, list of float
                padding value if border_mode is cv2.BORDER_CONSTANT.
            mask_value: int, float, list of ints, list of float
                padding value if border_mode is cv2.BORDER_CONSTANT applied for masks.

        Targets
        -------
            image, mask
        Image types
        -----------
            uint8, float32
        """

        transform = A.Compose(
            [A.GridDistortion(num_steps, distort_limit, interpolation, border_mode, value, mask_value, always_apply,
                              p)],
            bbox_params=A.BboxParams(format=self.format))
        transformed = transform(image=self.image, bboxes=self.bbox)
        augmented_image = transformed['image']
        augmented_bbox = transformed['bboxes']
        self.image = augmented_image
        self.bbox = augmented_bbox

        return self

    def GridDropout(self, ratio=0.5, unit_size_min=None, unit_size_max=None, holes_number_x=None, holes_number_y=None,
                    shift_x=0, shift_y=0, random_offset=False, fill_value=0, mask_fill_value=None, always_apply=False,
                    p=1.0):
        """
        GridDropout, drops out rectangular regions of an image and the corresponding mask in a grid fashion.

        parameters
        ---------
            ratio: float
                the ratio of the mask holes to the unit_size (same for horizontal and vertical directions).
                Must be between 0 and 1. Default: 0.5.
            unit_size_min: ints
                minimum size of the grid unit. Must be between 2 and the image shorter edge.
                If 'None', holes_number_x and holes_number_y are used to setup the grid. Default: `None`.
            unit_size_max: int 
                maximum size of the grid unit. Must be between 2 and the image shorter edge.
                If 'None', holes_number_x and holes_number_y are used to setup the grid. Default: `None`.
            holes_number_x: int 
                the number of grid units in x direction. Must be between 1 and image width//2.
                If 'None', grid unit width is set as image_width//10. Default: `None`.
            holes_number_y: int 
                the number of grid units in y direction. Must be between 1 and image height//2.
                If `None`, grid unit height is set equal to the grid unit width or image height, whatever is smaller.
            shift_x: int 
                offsets of the grid start in x direction from (0,0) coordinate.
                Clipped between 0 and grid unit_width - hole_width. Default: 0.
            shift_y: int 
                offsets of the grid start in y direction from (0,0) coordinate.
                Clipped between 0 and grid unit height - hole_height. Default: 0.
            random_offset: bool
                weather to offset the grid randomly between 0 and grid unit size - hole size
                If 'True', entered shift_x, shift_y are ignored and set randomly. Default: `False`.
            fill_value: int 
                value for the dropped pixels. Default = 0
            mask_fill_value: int 
                value for the dropped pixels in mask.
                If `None`, transformation is not applied to the mask. Default: `None`.
        
        Targets
        -------
            image, mask
        Image types
        -----------
            uint8, float32
        References:
            https://arxiv.org/abs/2001.04086
        """

        transform = A.Compose(
            [A.GridDropout(ratio, unit_size_min, unit_size_max, holes_number_x, holes_number_y, shift_x, shift_y,
                           random_offset, fill_value, mask_fill_value, always_apply, p)],
            bbox_params=A.BboxParams(format=self.format))
        transformed = transform(image=self.image, bboxes=self.bbox)
        augmented_image = transformed['image']
        augmented_bbox = transformed['bboxes']
        self.image = augmented_image
        self.bbox = augmented_bbox

        return self

    def HorizontalFlip(self, p=1.0):
        """
        Flip the input horizontally around the y-axis.
        
        parameters
        ---------
            p: float
                probability of applying the transform. Default: 0.5.
        
        Targets
        -------
            image, mask, bboxes, keypoints
        Image types
        -----------
            uint8, float32
        """

        transform = A.Compose(
            [A.HorizontalFlip(p)],
            bbox_params=A.BboxParams(format=self.format))
        transformed = transform(image=self.image, bboxes=self.bbox)
        augmented_image = transformed['image']
        augmented_bbox = transformed['bboxes']
        self.image = augmented_image
        self.bbox = augmented_bbox

        return self

    def LongestMaxSize(self, max_size=1024, interpolation=1, always_apply=False, p=1.0):
        """
        Rescale an image so that maximum side is equal to max_size, keeping the aspect ratio of the initial image.

        parameters
        ---------
            max_size: int, list of int
                maximum size of the image after the transformation. When using a list, max size
                will be randomly selected from the values in the list.
            interpolation: OpenCV flag
                interpolation method. Default: cv2.INTER_LINEAR.
            p: float
                probability of applying the transform. Default: 1.

        Targets
        -------
            image, mask, bboxes, keypoints
        Image types
        -----------
            uint8, float32
        """

        transform = A.Compose(
            [A.LongestMaxSize(max_size, interpolation, always_apply, p)],
            bbox_params=A.BboxParams(format=self.format))
        transformed = transform(image=self.image, bboxes=self.bbox)
        augmented_image = transformed['image']
        augmented_bbox = transformed['bboxes']
        self.image = augmented_image
        self.bbox = augmented_bbox

        return self

    def MaskDropout(self, max_objects=1, image_fill_value=0, mask_fill_value=0, always_apply=False, p=1.0):
        """
        Image & mask augmentation that zero out mask and image regions corresponding
        to randomly chosen object instance from mask.
        Mask must be single-channel image, zero values treated as background.
        Image can be any number of channels.
        Inspired by https://www.kaggle.com/c/severstal-steel-defect-detection/discussion/114254

        parameters
        ---------
            max_objects:
                Maximum number of labels that can be zeroed out. Can be tuple, in this case it's [min, max]
            image_fill_value: 
                Fill value to use when filling image.
                Can be 'inpaint' to apply inpaining (works only  for 3-chahnel images)
            mask_fill_value: 
                Fill value to use when filling mask.
        
        Targets
        -------
            image, mask
        Image types
        -----------
            uint8, float32
        """

        transform = A.Compose(
            [A.MaskDropout(max_objects, image_fill_value, mask_fill_value, always_apply, p)],
            bbox_params=A.BboxParams(format=self.format))
        transformed = transform(image=self.image, bboxes=self.bbox)
        augmented_image = transformed['image']
        augmented_bbox = transformed['bboxes']
        self.image = augmented_image
        self.bbox = augmented_bbox

        return self


    def OpticalDistortion(self, distort_limit=0.05, shift_limit=0.05, interpolation=1, border_mode=4, value=None,
                          mask_value=None, always_apply=False, p=1.0):
        """
        parameters
        ---------
            distort_limit: float, [float, float]
                If distort_limit is a single float, the range
                will be (-distort_limit, distort_limit). Default: (-0.05, 0.05).
            shift_limit: float, [float, float]
                If shift_limit is a single float, the range
                will be (-shift_limit, shift_limit). Default: (-0.05, 0.05).
            interpolation: OpenCV flag
                that is used to specify the interpolation algorithm. Should be one of:
                cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
                Default: cv2.INTER_LINEAR.
            border_mode: OpenCV flag
                that is used to specify the pixel extrapolation method. Should be one of:
                cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE, cv2.BORDER_REFLECT, cv2.BORDER_WRAP, cv2.BORDER_REFLECT_101.
                Default: cv2.BORDER_REFLECT_101
            value: int, float, list of ints, list of float
                padding value if border_mode is cv2.BORDER_CONSTANT.
            mask_value: int, float, list of ints, list of float
                padding value if border_mode is cv2.BORDER_CONSTANT applied for masks.
        
        Targets
        -------
            image, mask
        Image types
        -----------
            uint8, float32
        """

        transform = A.Compose(
            [A.OpticalDistortion(distort_limit, shift_limit, interpolation, border_mode, value, mask_value,
                                 always_apply, p)],
            bbox_params=A.BboxParams(format=self.format))
        transformed = transform(image=self.image, bboxes=self.bbox)
        augmented_image = transformed['image']
        augmented_bbox = transformed['bboxes']
        self.image = augmented_image
        self.bbox = augmented_bbox

        return self

    def PadIfNeeded(self, min_height=1024, min_width=1024, pad_height_divisor=None, pad_width_divisor=None,
                    position='center', border_mode=4, value=None, mask_value=None, always_apply=False, p=1.0):
        """Pad side of the image / max if side is less than desired number.

        parameters
        ---------
            min_height: int
                minimal result image height.
            min_width: int
                minimal result image width.
            pad_height_divisor: int
                if not None, ensures image height is dividable by value of this argument.
            pad_width_divisor: int
                if not None, ensures image width is dividable by value of this argument.
            position: Union[str, PositionType] 
                Position of the image. should be PositionType.CENTER or
                PositionType.TOP_LEFT or PositionType.TOP_RIGHT or PositionType.BOTTOM_LEFT or PositionType.BOTTOM_RIGHT.
                Default: PositionType.CENTER.
            border_mode: OpenCV flag
                OpenCV border mode.
            value: int, float, list of int, list of float
                padding value if border_mode is cv2.BORDER_CONSTANT.
            mask_value: int, float, list of int, list of float
                padding value for mask if border_mode is cv2.BORDER_CONSTANT.
            p: float
                probability of applying the transform. Default: 1.0.

        Targets
        -------
            image, mask, bbox, keypoints
        Image types
        -----------
            uint8, float32
        """

        transform = A.Compose(
            [A.PadIfNeeded(min_height, min_width, pad_height_divisor, pad_width_divisor, position, border_mode, value,
                           mask_value, always_apply, p)],
            bbox_params=A.BboxParams(format=self.format))
        transformed = transform(image=self.image, bboxes=self.bbox)
        augmented_image = transformed['image']
        augmented_bbox = transformed['bboxes']
        self.image = augmented_image
        self.bbox = augmented_bbox

        return self

    def Perspective(self, scale=(0.05, 0.1), keep_size=True, pad_mode=0, pad_val=0, mask_pad_val=0, fit_output=False,
                    interpolation=1, always_apply=False, p=1.0):
        """
        Perform a random four point perspective transform of the input.

        parameters
        ---------
            scale: float or [float, float]
                standard deviation of the normal distributions. These are used to sample
                the random distances of the subimage's corners from the full image's corners.
                If scale is a single float value, the range will be (0, scale). Default: (0.05, 0.1).
            keep_size: bool
                Whether to resize image’s back to their original size after applying the perspective
                transform. If set to False, the resulting images may end up having different shapes
                and will always be a list, never an array. Default: True
            pad_mode: OpenCV flag
                OpenCV border mode.
            pad_val: int, float, list of int, list of float
                padding value if border_mode is cv2.BORDER_CONSTANT.
                Default: 0
            mask_pad_val: int, float, list of int, list of float
                padding value for mask
                if border_mode is cv2.BORDER_CONSTANT. Default: 0
            fit_output: bool
                If True, the image plane size and position will be adjusted to still capture
                the whole image after perspective transformation. (Followed by image resizing if keep_size is set to True.)
                Otherwise, parts of the transformed image may be outside of the image plane.
                This setting should not be set to True when using large scale values as it could lead to very large images.
                Default: False
            p: float
                probability of applying the transform. Default: 0.5.

        Targets
        -------
            image, mask, keypoints, bboxes
        Image types
        -----------
            uint8, float32
        """

        transform = A.Compose(
            [A.Perspective(scale, keep_size, pad_mode, pad_val, mask_pad_val, fit_output, interpolation, always_apply,
                           p)],
            bbox_params=A.BboxParams(format=self.format))
        transformed = transform(image=self.image, bboxes=self.bbox)
        augmented_image = transformed['image']
        augmented_bbox = transformed['bboxes']
        self.image = augmented_image
        self.bbox = augmented_bbox

        return self

    def PiecewiseAffine(self, scale=(0.03, 0.05), nb_rows=4, nb_cols=4, interpolation=1, mask_interpolation=0, cval=0,
                        cval_mask=0, mode='constant', absolute_scale=False, always_apply=False,
                        keypoints_threshold=0.01, p=1.0):
        """
        Apply affine transformations that differ between local neighbourhoods.
        This augmentation places a regular grid of points on an image and randomly moves the neighbourhood of these point
        around via affine transformations. This leads to local distortions.
        This is mostly a wrapper around scikit-image's ``PiecewiseAffine``.
        See also ``Affine`` for a similar technique.
        Note:
            This augmenter is very slow. Try to use ``ElasticTransformation`` instead, which is at least 10x faster.
        Note:
            For coordinate-based inputs (keypoints, bounding boxes, polygons, ...),
            this augmenter still has to perform an image-based augmentation,
            which will make it significantly slower and not fully correct for such inputs than other transforms.

        parameters
        ---------
            scale: float, tuple of float
                Each point on the regular grid is moved around via a normal distribution.
                This scale factor is equivalent to the normal distribution's sigma.
                Note that the jitter (how far each point is moved in which direction) is multiplied by the height/width of
                the image if ``absolute_scale=False`` (default), so this scale can be the same for different sized images.
                Recommended values are in the range ``0.01`` to ``0.05`` (weak to strong augmentations).
                    * If a single ``float``, then that value will always be used as the scale.
                    * If a tuple ``(a, b)`` of ``float`` s, then a random value will
                    be uniformly sampled per image from the interval ``[a, b]``.
            nb_rows:int, tuple of int
                Number of rows of points that the regular grid should have.
                Must be at least ``2``. For large images, you might want to pick a higher value than ``4``.
                You might have to then adjust scale to lower values.
                    * If a single ``int``, then that value will always be used as the number of rows.
                    * If a tuple ``(a, b)``, then a value from the discrete interval
                    ``[a..b]`` will be uniformly sampled per image.
            nb_cols: int, tuple of int
                Number of columns. Analogous to `nb_rows`.
            interpolation: int
                The order of interpolation. The order has to be in the range 0-5:
                - 0: Nearest-neighbor
                - 1: Bi-linear (default)
                - 2: Bi-quadratic
                - 3: Bi-cubic
                - 4: Bi-quartic
                - 5: Bi-quintic
            mask_interpolation: int
                same as interpolation but for mask.
            cval: number
                The constant value to use when filling in newly created pixels.
            cval_mask: number
                Same as cval but only for masks.
            mode: str
                {'constant', 'edge', 'symmetric', 'reflect', 'wrap'}, optional
                Points outside the boundaries of the input are filled according
                to the given mode.  Modes match the behaviour of `numpy.pad`.
            absolute_scale: bool 
                Take `scale` as an absolute value rather than a relative value.
            keypoints_threshold: float
                Used as threshold in conversion from distance maps to keypoints.
                The search for keypoints works by searching for the
                argmin (non-inverted) or argmax (inverted) in each channel. This
                parameters contains the maximum (non-inverted) or minimum (inverted) value to accept in order to view a hit
                as a keypoint. Use ``None`` to use no min/max. Default: 0.01
        
        Targets
        -------
            image, mask, keypoints, bboxes
        Image types
        -----------
            uint8, float32
        """

        transform = A.Compose(
            [A.PiecewiseAffine(scale, nb_rows, nb_cols, interpolation, mask_interpolation, cval, cval_mask, mode,
                               absolute_scale, always_apply, keypoints_threshold, p)],
            bbox_params=A.BboxParams(format=self.format))
        transformed = transform(image=self.image, bboxes=self.bbox)
        augmented_image = transformed['image']
        augmented_bbox = transformed['bboxes']
        self.image = augmented_image
        self.bbox = augmented_bbox

        return self

    def PixelDropout(self, dropout_prob=0.01, per_channel=False, drop_value=0, mask_drop_value=None, always_apply=False,
                     p=1.0):
        """
        Set pixels to 0 with some probability.

        parameters
        ---------
            dropout_prob: float
                pixel drop probability. Default: 0.01
            per_channel: bool
                if set to `True` drop mask will be sampled fo each channel,
                otherwise the same mask will be sampled for all channels. Default: False
            drop_value: number or sequence of numbers or None
                Value that will be set in dropped place.
                If set to None value will be sampled randomly, default ranges will be used:
                    - uint8 - [0, 255]
                    - uint16 - [0, 65535]
                    - uint32 - [0, 4294967295]
                    - float, double - [0, 1]
                Default: 0
            mask_drop_value: number or sequence of numbers or None
                Value that will be set in dropped place in masks.
                If set to None masks will be unchanged. Default: 0
            p: float
                probability of applying the transform. Default: 0.5.
        
        Targets
        -------
            image, mask
        Image types
        -----------
            any
        """

        transform = A.Compose(
            [A.PixelDropout(dropout_prob, per_channel, drop_value, mask_drop_value, always_apply, p)],
            bbox_params=A.BboxParams(format=self.format))
        transformed = transform(image=self.image, bboxes=self.bbox)
        augmented_image = transformed['image']
        augmented_bbox = transformed['bboxes']
        self.image = augmented_image
        self.bbox = augmented_bbox

        return self

    def RandomCrop(self, always_apply=False, p=1.0):
        """
        Crop a random part of the input.

        parameters
        ---------
            height: int
                height of the crop.
            width: int
                width of the crop.
            p: float
                probability of applying the transform. Default: 1.
        
        Targets
        -------
            image, mask, bboxes, keypoints
        Image types
        -----------
            uint8, float32
        """
        height, width, channel = self.image.shape
        transform = A.Compose(
            [A.RandomCrop(height, width, always_apply, p)],
            bbox_params=A.BboxParams(format=self.format))
        transformed = transform(image=self.image, bboxes=self.bbox)
        augmented_image = transformed['image']
        augmented_bbox = transformed['bboxes']
        self.image = augmented_image
        self.bbox = augmented_bbox

        return self

    def RandomCropNearBBox(self, max_per_shift=(0.3, 0.3), cropping_box_key='cropping_bbox', always_apply=False, p=1.0):
        """
        Crop bbox from image with random shift by x,y coordinates

        parameters
        ---------
            max_part_shift: float, [float, float]
                Max shift in `height` and `width` dimensions relative
                to `cropping_bbox` dimension.
                If max_part_shift is a single float, the range will be (max_part_shift, max_part_shift).
                Default (0.3, 0.3).
            cropping_box_key: str
                Additional target key for cropping box. Default `cropping_bbox`
            p: float
                probability of applying the transform. Default: 1.
        
        Targets
        -------
            image, mask, bboxes, keypoints
        Image types
        -----------
            uint8, float32
        Examples:
            >>> aug = Compose([RandomCropNearBBox(max_part_shift=(0.1, 0.5), cropping_box_key='test_box')],
            >>>              bbox_params=BboxParams("pascal_voc"))
            >>> result = aug(image=self.image, bboxes=self.bbox, test_box=[0, 5, 10, 20])
        """

        transform = A.Compose(
            [A.RandomCropNearBBox(max_per_shift, cropping_box_key, always_apply, p)],
            bbox_params=A.BboxParams(format=self.format))
        transformed = transform(image=self.image, bboxes=self.bbox)
        augmented_image = transformed['image']
        augmented_bbox = transformed['bboxes']
        self.image = augmented_image
        self.bbox = augmented_bbox

        return self

    def RandomGridShuffle(self, grid=(3, 3), always_apply=False, p=1.0):
        """
        Random shuffle grid's cells on image.

        parameters
        ---------
            grid: [int, int]
                size of grid for splitting image.
        
        Targets
        -------
            image, mask, keypoints
        Image types
        -----------
            uint8, float32
        """

        transform = A.Compose(
            [A.RandomGridShuffle(grid, always_apply, p)],
            bbox_params=A.BboxParams(format=self.format))
        transformed = transform(image=self.image, bboxes=self.bbox)
        augmented_image = transformed['image']
        augmented_bbox = transformed['bboxes']
        self.image = augmented_image
        self.bbox = augmented_bbox

        return self

    def RandomResizedCrop(self, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=1,
                          always_apply=False, p=1.0):
        """
        Torchvision's variant of crop a random part of the input and rescale it to some size.

        parameters
        ---------
            height: int
                height after crop and resize.
            width: int
                width after crop and resize.
            scale: [float, float]
                range of size of the origin size cropped
            ratio: [float, float]
                range of aspect ratio of the origin aspect ratio cropped
            interpolation: OpenCV flag
                flag that is used to specify the interpolation algorithm. Should be one of:
                cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
                Default: cv2.INTER_LINEAR.
            p: float
                probability of applying the transform. Default: 1.
        
        Targets
        -------
            image, mask, bboxes, keypoints
        Image types
        -----------
            uint8, float32
        """
        height, width, channel = self.image.shape
        transform = A.Compose(
            [A.RandomResizedCrop(height, width, scale, ratio, interpolation, always_apply, p)],
            bbox_params=A.BboxParams(format=self.format))
        transformed = transform(image=self.image, bboxes=self.bbox)
        augmented_image = transformed['image']
        augmented_bbox = transformed['bboxes']
        self.image = augmented_image
        self.bbox = augmented_bbox

        return self

    def RandomRotate90(self, p=1.0):
        """
        Randomly rotate the input by 90 degrees zero or more times.

        parameters
        ---------
            p: float
                probability of applying the transform. Default: 0.5.
        
        Targets
        -------
            image, mask, bboxes, keypoints
        Image types
        -----------
            uint8, float32
        """

        transform = A.Compose(
            [A.RandomRotate90(p)],
            bbox_params=A.BboxParams(format=self.format))
        transformed = transform(image=self.image, bboxes=self.bbox)
        augmented_image = transformed['image']
        augmented_bbox = transformed['bboxes']
        self.image = augmented_image
        self.bbox = augmented_bbox

        return self

    def RandomScale(self, scale_limit=0.1, interpolation=1, always_apply=False, p=1.0):
        """
        Randomly resize the input. Output image size is different from the input image size.

        parameters
        ---------
            scale_limit: [float, float] or float
                scaling factor range. If scale_limit is a single float value, the
                range will be (1 - scale_limit, 1 + scale_limit). Default: (0.9, 1.1).
            interpolation: OpenCV flag
                flag that is used to specify the interpolation algorithm. Should be one of:
                cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
                Default: cv2.INTER_LINEAR.
            p: float
                probability of applying the transform. Default: 0.5.
        
        Targets
        -------
            image, mask, bboxes, keypoints
        Image types
        -----------
            uint8, float32
        """

        transform = A.Compose(
            [A.RandomScale(scale_limit, interpolation, always_apply, p)],
            bbox_params=A.BboxParams(format=self.format))
        transformed = transform(image=self.image, bboxes=self.bbox)
        augmented_image = transformed['image']
        augmented_bbox = transformed['bboxes']
        self.image = augmented_image
        self.bbox = augmented_bbox

        return self

    def RandomSizedBBoxSafeCrop(self, erosion_rate=0.0, interpolation=1, always_apply=False, p=1.0):
        """
        Crop a random part of the input and rescale it to some size without loss of bboxes.

        parameters
        ---------
            height: int
                height after crop and resize.
            width: int
                width after crop and resize.
            erosion_rate: float
                erosion rate applied on input image height before crop.
            interpolation: OpenCV flag
                flag that is used to specify the interpolation algorithm. Should be one of:
                cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
                Default: cv2.INTER_LINEAR.
            p: float
                probability of applying the transform. Default: 1.
        
        Targets
        -------
            image, mask, bboxes
        Image types
        -----------
            uint8, float32
        """

        height, width, channel = self.image.shape
        transform = A.Compose(
            [A.RandomSizedBBoxSafeCrop(height, width, erosion_rate, interpolation, always_apply, p)],
            bbox_params=A.BboxParams(format=self.format))
        transformed = transform(image=self.image, bboxes=self.bbox)
        augmented_image = transformed['image']
        augmented_bbox = transformed['bboxes']
        self.image = augmented_image
        self.bbox = augmented_bbox

        return self

    def RandomSizedCrop(self, min_max_height=10, w2h_ratio=1.0, interpolation=1, always_apply=False, p=1.0):
        """
        Crop a random part of the input and rescale it to some size.

        parameters
        ---------
            min_max_height: [int, int]
                crop size limits.
            height: int
                height after crop and resize.
            width: int
                width after crop and resize.
            w2h_ratio: float
                aspect ratio of crop.
            interpolation: OpenCV flag
                flag that is used to specify the interpolation algorithm. Should be one of:
                cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
                Default: cv2.INTER_LINEAR.
            p: float
                probability of applying the transform. Default: 1.
        Targets
        -------
            image, mask, bboxes, keypoints
        Image types
        -----------
            uint8, float32
        """

        height, width, channel = self.image.shape
        transform = A.Compose(
            [A.RandomSizedCrop(min_max_height, height, width, w2h_ratio, interpolation, always_apply, p)],
            bbox_params=A.BboxParams(format=self.format))
        transformed = transform(image=self.image, bboxes=self.bbox)
        augmented_image = transformed['image']
        augmented_bbox = transformed['bboxes']
        self.image = augmented_image
        self.bbox = augmented_bbox

        return self

    def Resize(self, interpolation=1, always_apply=False, p=1):
        """
        Resize the input to the given height and width.

        parameters
        ---------
            height: int
                desired height of the output.
            width: int
                desired width of the output.
            interpolation: OpenCV flag
                flag that is used to specify the interpolation algorithm. Should be one of:
                cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
                Default: cv2.INTER_LINEAR.
            p: float
                probability of applying the transform. Default: 1.
        
        Targets
        -------
            image, mask, bboxes, keypoints
        Image types
        -----------
            uint8, float32
        """

        height, width, channel = self.image.shape
        transform = A.Compose(
            [A.Resize(height, width, interpolation, always_apply, p)],
            bbox_params=A.BboxParams(format=self.format))
        transformed = transform(image=self.image, bboxes=self.bbox)
        augmented_image = transformed['image']
        augmented_bbox = transformed['bboxes']
        self.image = augmented_image
        self.bbox = augmented_bbox

        return self

    def Rotate(self, limit=90, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, p=1.0):
        """
        Rotate the input by an angle selected randomly from the uniform distribution.

        parameters
        ---------
            limit: [int, int] or int
                range from which a random angle is picked. If limit is a single int
                an angle is picked from (-limit, limit). Default: (-90, 90)
            interpolation: OpenCV flag
                flag that is used to specify the interpolation algorithm. Should be one of:
                cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
                Default: cv2.INTER_LINEAR.
            border_mode: OpenCV flag
                flag that is used to specify the pixel extrapolation method. Should be one of:
                cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE, cv2.BORDER_REFLECT, cv2.BORDER_WRAP, cv2.BORDER_REFLECT_101.
                Default: cv2.BORDER_REFLECT_101
            value: int, float, list of ints, list of float
                padding value if border_mode is cv2.BORDER_CONSTANT.
            mask_value: int, float, list of ints, list of float
                padding value if border_mode is cv2.BORDER_CONSTANT applied for masks.
            p: float
                probability of applying the transform. Default: 0.5.
        
        Targets
        -------
            image, mask, bboxes, keypoints
        Image types
        -----------
            uint8, float32
        """

        transform = A.Compose(
            [A.Rotate(limit, interpolation, border_mode, value, mask_value, always_apply, p)],
            bbox_params=A.BboxParams(format=self.format))
        transformed = transform(image=self.image, bboxes=self.bbox)
        augmented_image = transformed['image']
        augmented_bbox = transformed['bboxes']
        self.image = augmented_image
        self.bbox = augmented_bbox

        return self

    def SafeRotate(self, limit=90, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False,
                   p=1.0):
        """
        Rotate the input inside the input's frame by an angle selected randomly from the uniform distribution.
        The resulting image may have artifacts in it. After rotation, the image may have a different aspect ratio, and
        after resizing, it returns to its original shape with the original aspect ratio of the image. For these reason we
        may see some artifacts.

        parameters
        ---------
            limit: [int, int] or int
                range from which a random angle is picked. If limit is a single int
                an angle is picked from (-limit, limit). Default: (-90, 90)
            interpolation: OpenCV flag 
                flag that is used to specify the interpolation algorithm. Should be one of:
                cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
                Default: cv2.INTER_LINEAR.
            border_mode: OpenCV flag
                flag that is used to specify the pixel extrapolation method. Should be one of:
                cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE, cv2.BORDER_REFLECT, cv2.BORDER_WRAP, cv2.BORDER_REFLECT_101.
                Default: cv2.BORDER_REFLECT_101
            value: int, float, list of ints, list of float
                padding value if border_mode is cv2.BORDER_CONSTANT.
            mask_value: int, float, list of ints, list of float
                padding value if border_mode is cv2.BORDER_CONSTANT applied for masks.
            p: float
                probability of applying the transform. Default: 0.5.

        Targets
        -------
            image, mask, bboxes, keypoints
        Image types
        -----------
            uint8, float32
        """

        transform = A.Compose(
            [A.SafeRotate(limit, interpolation, border_mode, value, mask_value, always_apply, p)],
            bbox_params=A.BboxParams(format=self.format))
        transformed = transform(image=self.image, bboxes=self.bbox)
        augmented_image = transformed['image']
        augmented_bbox = transformed['bboxes']
        self.image = augmented_image
        self.bbox = augmented_bbox

        return self

    def ShiftScaleRotate(self, shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, interpolation=1, border_mode=4,
                         value=None, mask_value=None, shift_limit_x=None, shift_limit_y=None, always_apply=False,
                         p=1.0):
        """
        Randomly apply affine transforms: translate, scale and rotate the input.

        parameters
        ---------
            shift_limit: [float, float] or float    
                shift factor range for both height and width. If shift_limit
                is a single float value, the range will be (-shift_limit, shift_limit). Absolute values for lower and
                upper bounds should lie in range [0, 1]. Default: (-0.0625, 0.0625).
            scale_limit: [float, float] or float    
                scaling factor range. If scale_limit is a single float value, the
                range will be (-scale_limit, scale_limit). Default: (-0.1, 0.1).
            rotate_limit: [int, int] or int
                rotation range. If rotate_limit is a single int value, the
                range will be (-rotate_limit, rotate_limit). Default: (-45, 45).
            interpolation: OpenCV flag
                flag that is used to specify the interpolation algorithm. Should be one of:
                cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
                Default: cv2.INTER_LINEAR.
            border_mode: OpenCV flag
                flag that is used to specify the pixel extrapolation method. Should be one of:
                cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE, cv2.BORDER_REFLECT, cv2.BORDER_WRAP, cv2.BORDER_REFLECT_101.
                Default: cv2.BORDER_REFLECT_101
            value: int, float, list of int, list of float
                padding value if border_mode is cv2.BORDER_CONSTANT.
            mask_value: int, float, list of int, list of float
                padding value if border_mode is cv2.BORDER_CONSTANT applied for masks.
            shift_limit_x: [float, float] or float  
                shift factor range for width. If it is set then this value
                instead of shift_limit will be used for shifting width.  If shift_limit_x is a single float value,
                the range will be (-shift_limit_x, shift_limit_x). Absolute values for lower and upper bounds should lie in
                the range [0, 1]. Default: None.
            shift_limit_y: [float, float] or float  
                shift factor range for height. If it is set then this value
                instead of shift_limit will be used for shifting height.  If shift_limit_y is a single float value,
                the range will be (-shift_limit_y, shift_limit_y). Absolute values for lower and upper bounds should lie
                in the range [0, 1]. Default: None.
            p: float
                probability of applying the transform. Default: 0.5.
        
        Targets
        -------
            image, mask, keypoints
        Image types
        -----------
            uint8, float32
        """

        transform = A.Compose(
            [A.ShiftScaleRotate(shift_limit, scale_limit, rotate_limit, interpolation, border_mode, value, mask_value,
                                shift_limit_x, shift_limit_y, always_apply, p)],
            bbox_params=A.BboxParams(format=self.format))
        transformed = transform(image=self.image, bboxes=self.bbox)
        augmented_image = transformed['image']
        augmented_bbox = transformed['bboxes']
        self.image = augmented_image
        self.bbox = augmented_bbox

        return self

    def SmallestMaxSize(self, max_size=1024, interpolation=1, always_apply=False, p=1):
        """
        Rescale an image so that minimum side is equal to max_size, keeping the aspect ratio of the initial image.

        parameters
        ---------
            max_size: int, list of int
                maximum size of smallest side of the image after the transformation. When using a
                list, max size will be randomly selected from the values in the list.
            interpolation: OpenCV flag
                interpolation method. Default: cv2.INTER_LINEAR.
            p: float 
                probability of applying the transform. Default: 1.
        
        Targets
        -------
            image, mask, bboxes, keypoints
        Image types
        -----------
            uint8, float32
        """

        transform = A.Compose(
            [A.SmallestMaxSize(max_size, interpolation, always_apply, p)],
            bbox_params=A.BboxParams(format=self.format))
        transformed = transform(image=self.image, bboxes=self.bbox)
        augmented_image = transformed['image']
        augmented_bbox = transformed['bboxes']
        self.image = augmented_image
        self.bbox = augmented_bbox

        return self

    def Transpose(self, p=1.0):
        """
        Transpose the input by swapping rows and columns.

        parameters
        ---------
            p: float
                probability of applying the transform. Default: 0.5.
        
        Targets
        -------
            image, mask, bboxes, keypoints
        Image types
        -----------
            uint8, float32
        """

        transform = A.Compose(
            [A.Transpose(p)],
            bbox_params=A.BboxParams(format=self.format))
        transformed = transform(image=self.image, bboxes=self.bbox)
        augmented_image = transformed['image']
        augmented_bbox = transformed['bboxes']
        self.image = augmented_image
        self.bbox = augmented_bbox

        return self

    def VerticalFlip(self, p=1.0):
        """
        Flip the input vertically around the x-axis.

        parameters
        ---------
            p: float
                probability of applying the transform. Default: 0.5.
        
        Targets
        -------
            image, mask, bboxes, keypoints
        Image types
        -----------
            uint8, float32
        """

        transform = A.Compose(
            [A.VerticalFlip(p)],
            bbox_params=A.BboxParams(format=self.format))
        transformed = transform(image=self.image, bboxes=self.bbox)
        augmented_image = transformed['image']
        augmented_bbox = transformed['bboxes']
        self.image = augmented_image
        self.bbox = augmented_bbox

        return self

    # ImageAug transforms
    def Add(self, value=(-20, 20), per_channel=False, seed=None, name=None, random_state='deprecated',
            deterministic='deprecated'):
        """
        Add a value to all pixels in an image.

        Parameters
        ----------
        value : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
            Value to add to all pixels.

                * If a number, exactly that value will always be used.
                * If a tuple ``(a, b)``, then a value from the discrete
                interval ``[a..b]`` will be sampled per image.
                * If a list, then a random value will be sampled from that list
                per image.
                * If a ``StochasticParameter``, then a value will be sampled per
                image from that parameter.

        per_channel : bool or float or imgaug.parameters.StochasticParameter, optional
            Whether to use (imagewise) the same sample(s) for all
            channels (``False``) or to sample value(s) for each channel (``True``).
            Setting this to ``True`` will therefore lead to different
            transformations per image *and* channel, otherwise only per image.
            If this value is a float ``p``, then for ``p`` percent of all images
            `per_channel` will be treated as ``True``.
            If it is a ``StochasticParameter`` it is expected to produce samples
            with values between ``0.0`` and ``1.0``, where values ``>0.5`` will
            lead to per-channel behaviour (i.e. same as ``True``).

        seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        name : None or str, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            Old name for parameter `seed`.
            Its usage will not yet cause a deprecation warning,
            but it is still recommended to use `seed` now.
            Outdated since 0.4.0.

        deterministic : bool, optional
            Deprecated since 0.4.0.
            See method ``to_deterministic()`` for an alternative and for
            details about what the "deterministic mode" actually does.
        """

        transform = iaa.Add(value, per_channel, seed, name, random_state, deterministic)
        augmented_image = transform(image=self.image)
        self.image = augmented_image

        return self

    def AddElementwise(self, value=(-20, 20), per_channel=False, seed=None, name=None, random_state='deprecated',
                       deterministic='deprecated'):
        """
        Add to the pixels of images values that are pixelwise randomly sampled.

        While the ``Add`` Augmenter samples one value to add *per image* (and
        optionally per channel), this augmenter samples different values per image
        and *per pixel* (and optionally per channel), i.e. intensities of
        neighbouring pixels may be increased/decreased by different amounts.

        **Supported dtypes**:

        See :func:`~imgaug.augmenters.arithmetic.add_elementwise`.

        Parameters
        ----------
        value : int or tuple of int or list of int or imgaug.parameters.StochasticParameter, optional
            Value to add to the pixels.

                * If an int, exactly that value will always be used.
                * If a tuple ``(a, b)``, then values from the discrete interval
                ``[a..b]`` will be sampled per image and pixel.
                * If a list of integers, a random value will be sampled from the
                list per image and pixel.
                * If a ``StochasticParameter``, then values will be sampled per
                image and pixel from that parameter.

        per_channel : bool or float or imgaug.parameters.StochasticParameter, optional
            Whether to use (imagewise) the same sample(s) for all
            channels (``False``) or to sample value(s) for each channel (``True``).
            Setting this to ``True`` will therefore lead to different
            transformations per image *and* channel, otherwise only per image.
            If this value is a float ``p``, then for ``p`` percent of all images
            `per_channel` will be treated as ``True``.
            If it is a ``StochasticParameter`` it is expected to produce samples
            with values between ``0.0`` and ``1.0``, where values ``>0.5`` will
            lead to per-channel behaviour (i.e. same as ``True``).

        seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        name : None or str, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            Old name for parameter `seed`.
            Its usage will not yet cause a deprecation warning,
            but it is still recommended to use `seed` now.
            Outdated since 0.4.0.

        deterministic : bool, optional
            Deprecated since 0.4.0.
            See method ``to_deterministic()`` for an alternative and for
            details about what the "deterministic mode" actually does.
            """

        transform = iaa.AddElementwise(value, per_channel, seed, name, random_state, deterministic)
        augmented_image = transform(image=self.image)
        self.image = augmented_image

        return self

    def AdditiveLaplaceNoise(self, loc=0, scale=(0, 15), per_channel=False, seed=None, name=None,
                             random_state='deprecated', deterministic='deprecated'):
        """
        Add noise sampled from laplace distributions elementwise to images.

        The laplace distribution is similar to the gaussian distribution, but
        puts more weight on the long tail. Hence, this noise will add more
        outliers (very high/low values). It is somewhere between gaussian noise and
        salt and pepper noise.

        Values of around ``255 * 0.05`` for `scale` lead to visible noise (for
        ``uint8``).
        Values of around ``255 * 0.10`` for `scale` lead to very visible
        noise (for ``uint8``).
        It is recommended to usually set `per_channel` to ``True``.

        This augmenter samples and adds noise elementwise, i.e. it can add
        different noise values to neighbouring pixels and is comparable
        to ``AddElementwise``.

        **Supported dtypes**:

        See :class:`~imgaug.augmenters.arithmetic.AddElementwise`.

        Parameters
        ----------
        loc : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
            Mean of the laplace distribution that generates the noise.

                * If a number, exactly that value will always be used.
                * If a tuple ``(a, b)``, a random value from the interval
                ``[a, b]`` will be sampled per image.
                * If a list, then a random value will be sampled from that list per
                image.
                * If a ``StochasticParameter``, a value will be sampled from the
                parameter per image.

        scale : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
            Standard deviation of the laplace distribution that generates the noise.
            Must be ``>=0``. If ``0`` then only `loc` will be used.
            Recommended to be around ``255*0.05``.

                * If a number, exactly that value will always be used.
                * If a tuple ``(a, b)``, a random value from the interval
                ``[a, b]`` will be sampled per image.
                * If a list, then a random value will be sampled from that list per
                image.
                * If a ``StochasticParameter``, a value will be sampled from the
                parameter per image.

        per_channel : bool or float or imgaug.parameters.StochasticParameter, optional
            Whether to use (imagewise) the same sample(s) for all
            channels (``False``) or to sample value(s) for each channel (``True``).
            Setting this to ``True`` will therefore lead to different
            transformations per image *and* channel, otherwise only per image.
            If this value is a float ``p``, then for ``p`` percent of all images
            `per_channel` will be treated as ``True``.
            If it is a ``StochasticParameter`` it is expected to produce samples
            with values between ``0.0`` and ``1.0``, where values ``>0.5`` will
            lead to per-channel behaviour (i.e. same as ``True``).

        seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        name : None or str, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            Old name for parameter `seed`.
            Its usage will not yet cause a deprecation warning,
            but it is still recommended to use `seed` now.
            Outdated since 0.4.0.

        deterministic : bool, optional
            Deprecated since 0.4.0.
            See method ``to_deterministic()`` for an alternative and for
            details about what the "deterministic mode" actually does.
            """
        need_to_run = random.random() < self.p
        if need_to_run:
            transform = iaa.AdditiveLaplaceNoise(loc, scale, per_channel, seed, name, random_state, deterministic)
            augmented_image = transform(image=self.image)
            self.image = augmented_image
        else:
            pass

        return self

    def AdditivePoissonNoise(self, lam=(0.0, 15.0), per_channel=False, seed=None, name=None, random_state='deprecated',
                             deterministic='deprecated'):
        """
        Add noise sampled from poisson distributions elementwise to images.

        Poisson noise is comparable to gaussian noise, as e.g. generated via
        ``AdditiveGaussianNoise``. As poisson distributions produce only positive
        numbers, the sign of the sampled values are here randomly flipped.

        Values of around ``10.0`` for `lam` lead to visible noise (for ``uint8``).
        Values of around ``20.0`` for `lam` lead to very visible noise (for
        ``uint8``).
        It is recommended to usually set `per_channel` to ``True``.

        This augmenter samples and adds noise elementwise, i.e. it can add
        different noise values to neighbouring pixels and is comparable
        to ``AddElementwise``.

        **Supported dtypes**:

        See :class:`~imgaug.augmenters.arithmetic.AddElementwise`.

        Parameters
        ----------
        lam : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
            Lambda parameter of the poisson distribution. Must be ``>=0``.
            Recommended values are around ``0.0`` to ``10.0``.

                * If a number, exactly that value will always be used.
                * If a tuple ``(a, b)``, a random value from the interval
                ``[a, b]`` will be sampled per image.
                * If a list, then a random value will be sampled from that list
                per image.
                * If a ``StochasticParameter``, a value will be sampled from the
                parameter per image.

        per_channel : bool or float or imgaug.parameters.StochasticParameter, optional
            Whether to use (imagewise) the same sample(s) for all
            channels (``False``) or to sample value(s) for each channel (``True``).
            Setting this to ``True`` will therefore lead to different
            transformations per image *and* channel, otherwise only per image.
            If this value is a float ``p``, then for ``p`` percent of all images
            `per_channel` will be treated as ``True``.
            If it is a ``StochasticParameter`` it is expected to produce samples
            with values between ``0.0`` and ``1.0``, where values ``>0.5`` will
            lead to per-channel behaviour (i.e. same as ``True``).

        seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        name : None or str, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            Old name for parameter `seed`.
            Its usage will not yet cause a deprecation warning,
            but it is still recommended to use `seed` now.
            Outdated since 0.4.0.

        deterministic : bool, optional
            Deprecated since 0.4.0.
            See method ``to_deterministic()`` for an alternative and for
            details about what the "deterministic mode" actually does.
        """

        transform = iaa.AdditivePoissonNoise(lam, per_channel, seed, name, random_state, deterministic)
        augmented_image = transform(image=self.image)
        self.image = augmented_image

        return self

    def Multiply(self, mul=(0.8, 1.2), per_channel=False, seed=None, name=None, random_state='deprecated',
                 deterministic='deprecated'):
        """
        Multiply all pixels in an image with a random value sampled once per image.

        This augmenter can be used to make images lighter or darker.

        **Supported dtypes**:

        See :func:`~imgaug.augmenters.arithmetic.multiply_scalar`.

        Parameters
        ----------
        mul : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
            The value with which to multiply the pixel values in each image.

                * If a number, then that value will always be used.
                * If a tuple ``(a, b)``, then a value from the interval ``[a, b]``
                will be sampled per image and used for all pixels.
                * If a list, then a random value will be sampled from that list per
                image.
                * If a ``StochasticParameter``, then that parameter will be used to
                sample a new value per image.

        per_channel : bool or float or imgaug.parameters.StochasticParameter, optional
            Whether to use (imagewise) the same sample(s) for all
            channels (``False``) or to sample value(s) for each channel (``True``).
            Setting this to ``True`` will therefore lead to different
            transformations per image *and* channel, otherwise only per image.
            If this value is a float ``p``, then for ``p`` percent of all images
            `per_channel` will be treated as ``True``.
            If it is a ``StochasticParameter`` it is expected to produce samples
            with values between ``0.0`` and ``1.0``, where values ``>0.5`` will
            lead to per-channel behaviour (i.e. same as ``True``).

        seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        name : None or str, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            Old name for parameter `seed`.
            Its usage will not yet cause a deprecation warning,
            but it is still recommended to use `seed` now.
            Outdated since 0.4.0.

        deterministic : bool, optional
            Deprecated since 0.4.0.
            See method ``to_deterministic()`` for an alternative and for
            details about what the "deterministic mode" actually does.
        """

        transform = iaa.Multiply(mul, per_channel, seed, name, random_state, deterministic)
        augmented_image = transform(image=self.image)
        self.image = augmented_image

        return self

    def MultiplyElementwise(self, mul=(0.8, 1.2), per_channel=False, seed=None, name=None, random_state='deprecated',
                            deterministic='deprecated'):
        """
        Multiply image pixels with values that are pixelwise randomly sampled.

        While the ``Multiply`` Augmenter uses a constant multiplier *per
        image* (and optionally channel), this augmenter samples the multipliers
        to use per image and *per pixel* (and optionally per channel).

        **Supported dtypes**:

        See :func:`~imgaug.augmenters.arithmetic.multiply_elementwise`.

        Parameters
        ----------
        mul : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
            The value with which to multiply pixel values in the image.

                * If a number, then that value will always be used.
                * If a tuple ``(a, b)``, then a value from the interval ``[a, b]``
                will be sampled per image and pixel.
                * If a list, then a random value will be sampled from that list
                per image and pixel.
                * If a ``StochasticParameter``, then that parameter will be used to
                sample a new value per image and pixel.

        per_channel : bool or float or imgaug.parameters.StochasticParameter, optional
            Whether to use (imagewise) the same sample(s) for all
            channels (``False``) or to sample value(s) for each channel (``True``).
            Setting this to ``True`` will therefore lead to different
            transformations per image *and* channel, otherwise only per image.
            If this value is a float ``p``, then for ``p`` percent of all images
            `per_channel` will be treated as ``True``.
            If it is a ``StochasticParameter`` it is expected to produce samples
            with values between ``0.0`` and ``1.0``, where values ``>0.5`` will
            lead to per-channel behaviour (i.e. same as ``True``).

        seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        name : None or str, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            Old name for parameter `seed`.
            Its usage will not yet cause a deprecation warning,
            but it is still recommended to use `seed` now.
            Outdated since 0.4.0.

        deterministic : bool, optional
            Deprecated since 0.4.0.
            See method ``to_deterministic()`` for an alternative and for
            details about what the "deterministic mode" actually does.
        """

        transform = iaa.MultiplyElementwise(mul, per_channel, seed, name, random_state, deterministic)
        augmented_image = transform(image=self.image)
        self.image = augmented_image

        return self

    def Cutout(self, nb_iterations=1, position='uniform', size=0.2, squared=True, fill_mode='constant', cval=128,
               fill_per_channel=False, seed=None, name=None, random_state='deprecated', deterministic='deprecated'):
        """
        Fill one or more rectangular areas in an image using a fill mode.

        See paper "Improved Regularization of Convolutional Neural Networks with
        Cutout" by DeVries and Taylor.

        In contrast to the paper, this implementation also supports replacing
        image sub-areas with gaussian noise, random intensities or random RGB
        colors. It also supports non-squared areas. While the paper uses
        absolute pixel values for the size and position, this implementation
        uses relative values, which seems more appropriate for mixed-size
        datasets. The position parameter furthermore allows more flexibility, e.g.
        gaussian distributions around the center.

        .. note::

            This augmenter affects only image data. Other datatypes (e.g.
            segmentation map pixels or keypoints within the filled areas)
            are not affected.

        .. note::

            Gaussian fill mode will assume that float input images contain values
            in the interval ``[0.0, 1.0]`` and hence sample values from a
            gaussian within that interval, i.e. from ``N(0.5, std=0.5/3)``.

        Added in 0.4.0.

        **Supported dtypes**:

        See :func:`~imgaug.augmenters.arithmetic.cutout_`.

        Parameters
        ----------
        nb_iterations : int or tuple of int or list of int or imgaug.parameters.StochasticParameter, optional
            How many rectangular areas to fill.

                * If ``int``: Exactly that many areas will be filled on all images.
                * If ``tuple`` ``(a, b)``: A value from the interval ``[a, b]``
                will be sampled per image.
                * If ``list``: A random value will be sampled from that ``list``
                per image.
                * If ``StochasticParameter``: That parameter will be used to
                sample ``(B,)`` values per batch of ``B`` images.

        position : {'uniform', 'normal', 'center', 'left-top', 'left-center', 'left-bottom', 'center-top', 'center-center', 'center-bottom', 'right-top', 'right-center', 'right-bottom'} or tuple of float or StochasticParameter or tuple of StochasticParameter, optional
            Defines the position of each area to fill.
            Analogous to the definition in e.g.
            :class:`~imgaug.augmenters.size.CropToFixedSize`.
            Usually, ``uniform`` (anywhere in the image) or ``normal`` (anywhere
            in the image with preference around the center) are sane values.

        size : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
            The size of the rectangle to fill as a fraction of the corresponding
            image size, i.e. with value range ``[0.0, 1.0]``. The size is sampled
            independently per image axis.

                * If ``number``: Exactly that size is always used.
                * If ``tuple`` ``(a, b)``: A value from the interval ``[a, b]``
                will be sampled per area and axis.
                * If ``list``: A random value will be sampled from that ``list``
                per area and axis.
                * If ``StochasticParameter``: That parameter will be used to
                sample ``(N, 2)`` values per batch, where ``N`` is the total
                number of areas to fill within the whole batch.

        squared : bool or float or imgaug.parameters.StochasticParameter, optional
            Whether to generate only squared areas cutout areas or allow
            rectangular ones. If this evaluates to a true-like value, the
            first value from `size` will be converted to absolute pixels and used
            for both axes.

            If this value is a float ``p``, then for ``p`` percent of all areas
            to be filled `per_channel` will be treated as ``True``.
            If it is a ``StochasticParameter`` it is expected to produce samples
            with values between ``0.0`` and ``1.0``, where values ``>0.5`` will
            lead to per-channel behaviour (i.e. same as ``True``).

        fill_mode : str or list of str or imgaug.parameters.StochasticParameter, optional
            Mode to use in order to fill areas. Corresponds to ``mode`` parameter
            in some other augmenters. Valid strings for the mode are:

                * ``contant``: Fill each area with a single value.
                * ``gaussian``: Fill each area with gaussian noise.

            Valid datatypes are:

                * If ``str``: Exactly that mode will alaways be used.
                * If ``list``: A random value will be sampled from that ``list``
                per area.
                * If ``StochasticParameter``: That parameter will be used to
                sample ``(N,)`` values per batch, where ``N`` is the total number
                of areas to fill within the whole batch.

        cval : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
            The value to use (i.e. the color) to fill areas if `fill_mode` is
            ```constant``.

                * If ``number``: Exactly that value is used for all areas
                and channels.
                * If ``tuple`` ``(a, b)``: A value from the interval ``[a, b]``
                will be sampled per area (and channel if ``per_channel=True``).
                * If ``list``: A random value will be sampled from that ``list``
                per area (and channel if ``per_channel=True``).
                * If ``StochasticParameter``: That parameter will be used to
                sample ``(N, Cmax)`` values per batch, where ``N`` is the total
                number of areas to fill within the whole batch and ``Cmax``
                is the maximum number of channels in any image (usually ``3``).
                If ``per_channel=False``, only the first value of the second
                axis is used.

        fill_per_channel : bool or float or imgaug.parameters.StochasticParameter, optional
            Whether to fill each area in a channelwise fashion (``True``) or
            not (``False``).
            The behaviour per fill mode is:

                * ``constant``: Whether to fill all channels with the same value
                (i.e, grayscale) or different values (i.e. usually RGB color).
                * ``gaussian``: Whether to sample once from a gaussian and use the
                values for all channels (i.e. grayscale) or to sample
                channelwise (i.e. RGB colors)

            If this value is a float ``p``, then for ``p`` percent of all areas
            to be filled `per_channel` will be treated as ``True``.
            If it is a ``StochasticParameter`` it is expected to produce samples
            with values between ``0.0`` and ``1.0``, where values ``>0.5`` will
            lead to per-channel behaviour (i.e. same as ``True``).

        name : None or str, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        deterministic : bool, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.bit_generator.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.
        """

        transform = iaa.Cutout(nb_iterations, position, size, squared, fill_mode, cval, fill_per_channel, seed, name,
                               random_state, deterministic)
        augmented_image = transform(image=self.image)
        self.image = augmented_image

        return self

    def Dropout(self, p=(0.0, 0.05), per_channel=False, seed=None, name=None, random_state='deprecated',
                deterministic='deprecated'):
        """
        Set a fraction of pixels in images to zero.

        **Supported dtypes**:

        See :class:`~imgaug.augmenters.arithmetic.MultiplyElementwise`.

        Parameters
        ----------
        p : float or tuple of float or imgaug.parameters.StochasticParameter, optional
            The probability of any pixel being dropped (i.e. to set it to zero).

                * If a float, then that value will be used for all images. A value
                of ``1.0`` would mean that all pixels will be dropped
                and ``0.0`` that no pixels will be dropped. A value of ``0.05``
                corresponds to ``5`` percent of all pixels being dropped.
                * If a tuple ``(a, b)``, then a value ``p`` will be sampled from
                the interval ``[a, b]`` per image and be used as the pixel's
                dropout probability.
                * If a list, then a value will be sampled from that list per
                batch and used as the probability.
                * If a ``StochasticParameter``, then this parameter will be used to
                determine per pixel whether it should be *kept* (sampled value
                of ``>0.5``) or shouldn't be kept (sampled value of ``<=0.5``).
                If you instead want to provide the probability as a stochastic
                parameter, you can usually do ``imgaug.parameters.Binomial(1-p)``
                to convert parameter `p` to a 0/1 representation.

        per_channel : bool or float or imgaug.parameters.StochasticParameter, optional
            Whether to use (imagewise) the same sample(s) for all
            channels (``False``) or to sample value(s) for each channel (``True``).
            Setting this to ``True`` will therefore lead to different
            transformations per image *and* channel, otherwise only per image.
            If this value is a float ``p``, then for ``p`` percent of all images
            `per_channel` will be treated as ``True``.
            If it is a ``StochasticParameter`` it is expected to produce samples
            with values between ``0.0`` and ``1.0``, where values ``>0.5`` will
            lead to per-channel behaviour (i.e. same as ``True``).

        seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        name : None or str, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            Old name for parameter `seed`.
            Its usage will not yet cause a deprecation warning,
            but it is still recommended to use `seed` now.
            Outdated since 0.4.0.

        deterministic : bool, optional
            Deprecated since 0.4.0.
            See method ``to_deterministic()`` for an alternative and for
            details about what the "deterministic mode" actually does.
        """

        transform = iaa.Dropout(p, per_channel, seed, name, random_state, deterministic)
        augmented_image = transform(image=self.image)
        self.image = augmented_image

        return self

    def Dropout2d(self, p=0.1, nb_keep_channels=1, seed=None, name=None, random_state='deprecated',
                  deterministic='deprecated'):
        """
        Drop random channels from images.

        For image data, dropped channels will be filled with zeros.

        .. note::

            This augmenter may also set the arrays of heatmaps and segmentation
            maps to zero and remove all coordinate-based data (e.g. it removes
            all bounding boxes on images that were filled with zeros).
            It does so if and only if *all* channels of an image are dropped.
            If ``nb_keep_channels >= 1`` then that never happens.

        Added in 0.4.0.

        **Supported dtypes**:

            * ``uint8``: yes; fully tested
            * ``uint16``: yes; tested
            * ``uint32``: yes; tested
            * ``uint64``: yes; tested
            * ``int8``: yes; tested
            * ``int16``: yes; tested
            * ``int32``: yes; tested
            * ``int64``: yes; tested
            * ``float16``: yes; tested
            * ``float32``: yes; tested
            * ``float64``: yes; tested
            * ``float128``: yes; tested
            * ``bool``: yes; tested

        Parameters
        ----------
        p : float or tuple of float or imgaug.parameters.StochasticParameter, optional
            The probability of any channel to be dropped (i.e. set to zero).

                * If a ``float``, then that value will be used for all channels.
                A value of ``1.0`` would mean, that all channels will be dropped.
                A value of ``0.0`` would lead to no channels being dropped.
                * If a tuple ``(a, b)``, then a value ``p`` will be sampled from
                the interval ``[a, b)`` per batch and be used as the dropout
                probability.
                * If a list, then a value will be sampled from that list per
                batch and used as the probability.
                * If a ``StochasticParameter``, then this parameter will be used to
                determine per channel whether it should be *kept* (sampled value
                of ``>=0.5``) or shouldn't be kept (sampled value of ``<0.5``).
                If you instead want to provide the probability as a stochastic
                parameter, you can usually do ``imgaug.parameters.Binomial(1-p)``
                to convert parameter `p` to a 0/1 representation.

        nb_keep_channels : int
            Minimum number of channels to keep unaltered in all images.
            E.g. a value of ``1`` means that at least one channel in every image
            will not be dropped, even if ``p=1.0``. Set to ``0`` to allow dropping
            all channels.

        seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        name : None or str, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            Old name for parameter `seed`.
            Its usage will not yet cause a deprecation warning,
            but it is still recommended to use `seed` now.
            Outdated since 0.4.0.

        deterministic : bool, optional
            Deprecated since 0.4.0.
            See method ``to_deterministic()`` for an alternative and for
            details about what the "deterministic mode" actually does.
        """

        transform = iaa.Dropout2d(p, nb_keep_channels, seed, name, random_state, deterministic)
        augmented_image = transform(image=self.image)
        self.image = augmented_image

        return self

    def TotalDropout(self, p=1, seed=None, name=None, random_state='deprecated', deterministic='deprecated'):
        """
        Drop all channels of a defined fraction of all images.

        For image data, all components of dropped images will be filled with zeros.

        .. note::

            This augmenter also sets the arrays of heatmaps and segmentation
            maps to zero and removes all coordinate-based data (e.g. it removes
            all bounding boxes on images that were filled with zeros).

        Added in 0.4.0.

        **Supported dtypes**:

            * ``uint8``: yes; fully tested
            * ``uint16``: yes; tested
            * ``uint32``: yes; tested
            * ``uint64``: yes; tested
            * ``int8``: yes; tested
            * ``int16``: yes; tested
            * ``int32``: yes; tested
            * ``int64``: yes; tested
            * ``float16``: yes; tested
            * ``float32``: yes; tested
            * ``float64``: yes; tested
            * ``float128``: yes; tested
            * ``bool``: yes; tested

        Parameters
        ----------
        p : float or tuple of float or imgaug.parameters.StochasticParameter, optional
            The probability of an image to be filled with zeros.

                * If ``float``: The value will be used for all images.
                A value of ``1.0`` would mean that all images will be set to zero.
                A value of ``0.0`` would lead to no images being set to zero.
                * If ``tuple`` ``(a, b)``: A value ``p`` will be sampled from
                the interval ``[a, b)`` per batch and be used as the dropout
                probability.
                * If a list, then a value will be sampled from that list per
                batch and used as the probability.
                * If ``StochasticParameter``: The parameter will be used to
                determine per image whether it should be *kept* (sampled value
                of ``>=0.5``) or shouldn't be kept (sampled value of ``<0.5``).
                If you instead want to provide the probability as a stochastic
                parameter, you can usually do ``imgaug.parameters.Binomial(1-p)``
                to convert parameter `p` to a 0/1 representation.

        seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        name : None or str, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            Old name for parameter `seed`.
            Its usage will not yet cause a deprecation warning,
            but it is still recommended to use `seed` now.
            Outdated since 0.4.0.

        deterministic : bool, optional
            Deprecated since 0.4.0.
            See method ``to_deterministic()`` for an alternative and for
            details about what the "deterministic mode" actually does.
        """

        transform = iaa.TotalDropout(p, seed, name, random_state, deterministic)
        augmented_image = transform(image=self.image)
        self.image = augmented_image

        return self

    def ReplaceElementwise(self, mask, replacement, per_channel=False, seed=None, name=None, random_state='deprecated',
                           deterministic='deprecated'):
        """
        Replace pixels in an image with new values.

        **Supported dtypes**:

        See :func:`~imgaug.augmenters.arithmetic.replace_elementwise_`.

        Parameters
        ----------
        mask : float or tuple of float or list of float or imgaug.parameters.StochasticParameter
            Mask that indicates the pixels that are supposed to be replaced.
            The mask will be binarized using a threshold of ``0.5``. A value
            of ``1`` then indicates a pixel that is supposed to be replaced.

                * If this is a float, then that value will be used as the
                probability of being a ``1`` in the mask (sampled per image and
                pixel) and hence being replaced.
                * If a tuple ``(a, b)``, then the probability will be uniformly
                sampled per image from the interval ``[a, b]``.
                * If a list, then a random value will be sampled from that list
                per image and pixel.
                * If a ``StochasticParameter``, then this parameter will be used to
                sample a mask per image.

        replacement : number or tuple of number or list of number or imgaug.parameters.StochasticParameter
            The replacement to use at all locations that are marked as ``1`` in
            the mask.

                * If this is a number, then that value will always be used as the
                replacement.
                * If a tuple ``(a, b)``, then the replacement will be sampled
                uniformly per image and pixel from the interval ``[a, b]``.
                * If a list, then a random value will be sampled from that list
                per image and pixel.
                * If a ``StochasticParameter``, then this parameter will be used
                sample replacement values per image and pixel.

        per_channel : bool or float or imgaug.parameters.StochasticParameter, optional
            Whether to use (imagewise) the same sample(s) for all
            channels (``False``) or to sample value(s) for each channel (``True``).
            Setting this to ``True`` will therefore lead to different
            transformations per image *and* channel, otherwise only per image.
            If this value is a float ``p``, then for ``p`` percent of all images
            `per_channel` will be treated as ``True``.
            If it is a ``StochasticParameter`` it is expected to produce samples
            with values between ``0.0`` and ``1.0``, where values ``>0.5`` will
            lead to per-channel behaviour (i.e. same as ``True``).

        seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        name : None or str, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            Old name for parameter `seed`.
            Its usage will not yet cause a deprecation warning,
            but it is still recommended to use `seed` now.
            Outdated since 0.4.0.

        deterministic : bool, optional
            Deprecated since 0.4.0.
            See method ``to_deterministic()`` for an alternative and for
            details about what the "deterministic mode" actually does.
        """

        transform = iaa.ReplaceElementwise(mask, replacement, per_channel, seed, name, random_state, deterministic)
        augmented_image = transform(image=self.image)
        self.image = augmented_image

        return self

    def ImpulseNoise(self, p=(0.0, 0.03), seed=None, name=None, random_state='deprecated', deterministic='deprecated'):
        """
        Add impulse noise to images.

        This is identical to ``SaltAndPepper``, except that `per_channel` is
        always set to ``True``.

        **Supported dtypes**:

        See :class:`~imgaug.augmenters.arithmetic.SaltAndPepper`.

        Parameters
        ----------
        p : float or tuple of float or list of float or imgaug.parameters.StochasticParameter, optional
            Probability of replacing a pixel to impulse noise.

                * If a float, then that value will always be used as the
                probability.
                * If a tuple ``(a, b)``, then a probability will be sampled
                uniformly per image from the interval ``[a, b]``.
                * If a list, then a random value will be sampled from that list
                per image.
                * If a ``StochasticParameter``, then a image-sized mask will be
                sampled from that parameter per image. Any value ``>0.5`` in
                that mask will be replaced with impulse noise noise.

        seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        name : None or str, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            Old name for parameter `seed`.
            Its usage will not yet cause a deprecation warning,
            but it is still recommended to use `seed` now.
            Outdated since 0.4.0.

        deterministic : bool, optional
            Deprecated since 0.4.0.
            See method ``to_deterministic()`` for an alternative and for
            details about what the "deterministic mode" actually does.
        """

        transform = iaa.ImpulseNoise(p, seed, name, random_state, deterministic)
        augmented_image = transform(image=self.image)
        self.image = augmented_image

        return self

    def SaltAndPepper(self, p=(0.0, 0.03), per_channel=False, seed=None, name=None, random_state='deprecated',
                      deterministic='deprecated'):
        """
        Replace pixels in images with salt/pepper noise (white/black-ish colors).

        **Supported dtypes**:

        See :class:`~imgaug.augmenters.arithmetic.ReplaceElementwise`.

        Parameters
        ----------
        p : float or tuple of float or list of float or imgaug.parameters.StochasticParameter, optional
            Probability of replacing a pixel to salt/pepper noise.

                * If a float, then that value will always be used as the
                probability.
                * If a tuple ``(a, b)``, then a probability will be sampled
                uniformly per image from the interval ``[a, b]``.
                * If a list, then a random value will be sampled from that list
                per image.
                * If a ``StochasticParameter``, then a image-sized mask will be
                sampled from that parameter per image. Any value ``>0.5`` in
                that mask will be replaced with salt and pepper noise.

        per_channel : bool or float or imgaug.parameters.StochasticParameter, optional
            Whether to use (imagewise) the same sample(s) for all
            channels (``False``) or to sample value(s) for each channel (``True``).
            Setting this to ``True`` will therefore lead to different
            transformations per image *and* channel, otherwise only per image.
            If this value is a float ``p``, then for ``p`` percent of all images
            `per_channel` will be treated as ``True``.
            If it is a ``StochasticParameter`` it is expected to produce samples
            with values between ``0.0`` and ``1.0``, where values ``>0.5`` will
            lead to per-channel behaviour (i.e. same as ``True``).

        seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        name : None or str, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            Old name for parameter `seed`.
            Its usage will not yet cause a deprecation warning,
            but it is still recommended to use `seed` now.
            Outdated since 0.4.0.

        deterministic : bool, optional
            Deprecated since 0.4.0.
            See method ``to_deterministic()`` for an alternative and for
            details about what the "deterministic mode" actually does.
        """

        transform = iaa.SaltAndPepper(p, per_channel, seed, name, random_state, deterministic)
        augmented_image = transform(image=self.image)
        self.image = augmented_image

        return self

    def CoarseSaltAndPepper(self, p=(0.02, 0.1), size_px=None, size_percent=None, per_channel=False, min_size=3,
                            seed=None, name=None, random_state='deprecated', deterministic='deprecated'):
        """
        Replace rectangular areas in images with white/black-ish pixel noise.

        This adds salt and pepper noise (noisy white-ish and black-ish pixels) to
        rectangular areas within the image. Note that this means that within these
        rectangular areas the color varies instead of each rectangle having only
        one color.

        See also the similar ``CoarseDropout``.

        TODO replace dtype support with uint8 only, because replacement is
            geared towards that value range

        **Supported dtypes**:

        See :class:`~imgaug.augmenters.arithmetic.ReplaceElementwise`.

        Parameters
        ----------
        p : float or tuple of float or list of float or imgaug.parameters.StochasticParameter, optional
            Probability of changing a pixel to salt/pepper noise.

                * If a float, then that value will always be used as the
                probability.
                * If a tuple ``(a, b)``, then a probability will be sampled
                uniformly per image from the interval ``[a, b]``.
                * If a list, then a random value will be sampled from that list
                per image.
                * If a ``StochasticParameter``, then a lower-resolution mask will
                be sampled from that parameter per image. Any value ``>0.5`` in
                that mask will denote a spatial location that is to be replaced
                by salt and pepper noise.

        size_px : int or tuple of int or imgaug.parameters.StochasticParameter, optional
            The size of the lower resolution image from which to sample the
            replacement mask in absolute pixel dimensions.
            Note that this means that *lower* values of this parameter lead to
            *larger* areas being replaced (as any pixel in the lower resolution
            image will correspond to a larger area at the original resolution).

                * If ``None`` then `size_percent` must be set.
                * If an integer, then that size will always be used for both height
                and width. E.g. a value of ``3`` would lead to a ``3x3`` mask,
                which is then upsampled to ``HxW``, where ``H`` is the image size
                and ``W`` the image width.
                * If a tuple ``(a, b)``, then two values ``M``, ``N`` will be
                sampled from the discrete interval ``[a..b]``. The mask
                will then be generated at size ``MxN`` and upsampled to ``HxW``.
                * If a ``StochasticParameter``, then this parameter will be used to
                determine the sizes. It is expected to be discrete.

        size_percent : float or tuple of float or imgaug.parameters.StochasticParameter, optional
            The size of the lower resolution image from which to sample the
            replacement mask *in percent* of the input image.
            Note that this means that *lower* values of this parameter lead to
            *larger* areas being replaced (as any pixel in the lower resolution
            image will correspond to a larger area at the original resolution).

                * If ``None`` then `size_px` must be set.
                * If a float, then that value will always be used as the percentage
                of the height and width (relative to the original size). E.g. for
                value ``p``, the mask will be sampled from ``(p*H)x(p*W)`` and
                later upsampled to ``HxW``.
                * If a tuple ``(a, b)``, then two values ``m``, ``n`` will be
                sampled from the interval ``(a, b)`` and used as the size
                fractions, i.e the mask size will be ``(m*H)x(n*W)``.
                * If a ``StochasticParameter``, then this parameter will be used to
                sample the percentage values. It is expected to be continuous.

        per_channel : bool or float or imgaug.parameters.StochasticParameter, optional
            Whether to use (imagewise) the same sample(s) for all
            channels (``False``) or to sample value(s) for each channel (``True``).
            Setting this to ``True`` will therefore lead to different
            transformations per image *and* channel, otherwise only per image.
            If this value is a float ``p``, then for ``p`` percent of all images
            `per_channel` will be treated as ``True``.
            If it is a ``StochasticParameter`` it is expected to produce samples
            with values between ``0.0`` and ``1.0``, where values ``>0.5`` will
            lead to per-channel behaviour (i.e. same as ``True``).

        min_size : int, optional
            Minimum height and width of the low resolution mask. If
            `size_percent` or `size_px` leads to a lower value than this,
            `min_size` will be used instead. This should never have a value of
            less than ``2``, otherwise one may end up with a ``1x1`` low resolution
            mask, leading easily to the whole image being replaced.

        seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        name : None or str, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            Old name for parameter `seed`.
            Its usage will not yet cause a deprecation warning,
            but it is still recommended to use `seed` now.
            Outdated since 0.4.0.

        deterministic : bool, optional
            Deprecated since 0.4.0.
            See method ``to_deterministic()`` for an alternative and for
            details about what the "deterministic mode" actually does.
        """

        transform = iaa.CoarseSaltAndPepper(p, size_px, size_percent, per_channel, min_size, seed, name, random_state,
                                            deterministic)
        augmented_image = transform(image=self.image)
        self.image = augmented_image

        return self

    def Salt(self, p=(0.0, 0.03), per_channel=False, seed=None, name=None, random_state='deprecated',
             deterministic='deprecated'):
        """
        Replace pixels in images with salt noise, i.e. white-ish pixels.

        This augmenter is similar to ``SaltAndPepper``, but adds no pepper noise to
        images.

        **Supported dtypes**:

        See :class:`~imgaug.augmenters.arithmetic.ReplaceElementwise`.

        Parameters
        ----------
        p : float or tuple of float or list of float or imgaug.parameters.StochasticParameter, optional
            Probability of replacing a pixel with salt noise.

                * If a float, then that value will always be used as the
                probability.
                * If a tuple ``(a, b)``, then a probability will be sampled
                uniformly per image from the interval ``[a, b]``.
                * If a list, then a random value will be sampled from that list
                per image.
                * If a ``StochasticParameter``, then a image-sized mask will be
                sampled from that parameter per image. Any value ``>0.5`` in
                that mask will be replaced with salt noise.

        per_channel : bool or float or imgaug.parameters.StochasticParameter, optional
            Whether to use (imagewise) the same sample(s) for all
            channels (``False``) or to sample value(s) for each channel (``True``).
            Setting this to ``True`` will therefore lead to different
            transformations per image *and* channel, otherwise only per image.
            If this value is a float ``p``, then for ``p`` percent of all images
            `per_channel` will be treated as ``True``.
            If it is a ``StochasticParameter`` it is expected to produce samples
            with values between ``0.0`` and ``1.0``, where values ``>0.5`` will
            lead to per-channel behaviour (i.e. same as ``True``).

        seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        name : None or str, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            Old name for parameter `seed`.
            Its usage will not yet cause a deprecation warning,
            but it is still recommended to use `seed` now.
            Outdated since 0.4.0.

        deterministic : bool, optional
            Deprecated since 0.4.0.
            See method ``to_deterministic()`` for an alternative and for
            details about what the "deterministic mode" actually does.
        """

        transform = iaa.Salt(p, per_channel, seed, name, random_state, deterministic)
        augmented_image = transform(image=self.image)
        self.image = augmented_image

        return self

    def CoarseSalt(self, p=(0.02, 0.1), size_px=None, size_percent=None, per_channel=False, min_size=3, seed=None,
                   name=None, random_state='deprecated', deterministic='deprecated'):
        """
        Replace rectangular areas in images with white-ish pixel noise.

        See also the similar ``CoarseSaltAndPepper``.

        **Supported dtypes**:

        See :class:`~imgaug.augmenters.arithmetic.ReplaceElementwise`.

        Parameters
        ----------
        p : float or tuple of float or list of float or imgaug.parameters.StochasticParameter, optional
            Probability of changing a pixel to salt noise.

                * If a float, then that value will always be used as the
                probability.
                * If a tuple ``(a, b)``, then a probability will be sampled
                uniformly per image from the interval ``[a, b]``.
                * If a list, then a random value will be sampled from that list
                per image.
                * If a ``StochasticParameter``, then a lower-resolution mask will
                be sampled from that parameter per image. Any value ``>0.5`` in
                that mask will denote a spatial location that is to be replaced
                by salt noise.

        size_px : int or tuple of int or imgaug.parameters.StochasticParameter, optional
            The size of the lower resolution image from which to sample the
            replacement mask in absolute pixel dimensions.
            Note that this means that *lower* values of this parameter lead to
            *larger* areas being replaced (as any pixel in the lower resolution
            image will correspond to a larger area at the original resolution).

                * If ``None`` then `size_percent` must be set.
                * If an integer, then that size will always be used for both height
                and width. E.g. a value of ``3`` would lead to a ``3x3`` mask,
                which is then upsampled to ``HxW``, where ``H`` is the image size
                and ``W`` the image width.
                * If a tuple ``(a, b)``, then two values ``M``, ``N`` will be
                sampled from the discrete interval ``[a..b]``. The mask
                will then be generated at size ``MxN`` and upsampled to ``HxW``.
                * If a ``StochasticParameter``, then this parameter will be used to
                determine the sizes. It is expected to be discrete.

        size_percent : float or tuple of float or imgaug.parameters.StochasticParameter, optional
            The size of the lower resolution image from which to sample the
            replacement mask *in percent* of the input image.
            Note that this means that *lower* values of this parameter lead to
            *larger* areas being replaced (as any pixel in the lower resolution
            image will correspond to a larger area at the original resolution).

                * If ``None`` then `size_px` must be set.
                * If a float, then that value will always be used as the percentage
                of the height and width (relative to the original size). E.g. for
                value ``p``, the mask will be sampled from ``(p*H)x(p*W)`` and
                later upsampled to ``HxW``.
                * If a tuple ``(a, b)``, then two values ``m``, ``n`` will be
                sampled from the interval ``(a, b)`` and used as the size
                fractions, i.e the mask size will be ``(m*H)x(n*W)``.
                * If a ``StochasticParameter``, then this parameter will be used to
                sample the percentage values. It is expected to be continuous.

        per_channel : bool or float or imgaug.parameters.StochasticParameter, optional
            Whether to use (imagewise) the same sample(s) for all
            channels (``False``) or to sample value(s) for each channel (``True``).
            Setting this to ``True`` will therefore lead to different
            transformations per image *and* channel, otherwise only per image.
            If this value is a float ``p``, then for ``p`` percent of all images
            `per_channel` will be treated as ``True``.
            If it is a ``StochasticParameter`` it is expected to produce samples
            with values between ``0.0`` and ``1.0``, where values ``>0.5`` will
            lead to per-channel behaviour (i.e. same as ``True``).

        min_size : int, optional
            Minimum height and width of the low resolution mask. If
            `size_percent` or `size_px` leads to a lower value than this,
            `min_size` will be used instead. This should never have a value of
            less than ``2``, otherwise one may end up with a ``1x1`` low resolution
            mask, leading easily to the whole image being replaced.

        seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        name : None or str, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            Old name for parameter `seed`.
            Its usage will not yet cause a deprecation warning,
            but it is still recommended to use `seed` now.
            Outdated since 0.4.0.

        deterministic : bool, optional
            Deprecated since 0.4.0.
            See method ``to_deterministic()`` for an alternative and for
            details about what the "deterministic mode" actually does.
        """

        transform = iaa.CoarseSalt(p, size_px, size_percent, per_channel, min_size, seed, name, random_state,
                                   deterministic)
        augmented_image = transform(image=self.image)
        self.image = augmented_image

        return self

    def Pepper(self, p=(0.0, 0.05), per_channel=False, seed=None, name=None, random_state='deprecated',
               deterministic='deprecated'):
        """
        Replace pixels in images with pepper noise, i.e. black-ish pixels.

        This augmenter is similar to ``SaltAndPepper``, but adds no salt noise to
        images.

        This augmenter is similar to ``Dropout``, but slower and the black pixels
        are not uniformly black.

        **Supported dtypes**:

        See :class:`~imgaug.augmenters.arithmetic.ReplaceElementwise`.

        Parameters
        ----------
        p : float or tuple of float or list of float or imgaug.parameters.StochasticParameter, optional
            Probability of replacing a pixel with pepper noise.

                * If a float, then that value will always be used as the
                probability.
                * If a tuple ``(a, b)``, then a probability will be sampled
                uniformly per image from the interval ``[a, b]``.
                * If a list, then a random value will be sampled from that list
                per image.
                * If a ``StochasticParameter``, then a image-sized mask will be
                sampled from that parameter per image. Any value ``>0.5`` in
                that mask will be replaced with pepper noise.

        per_channel : bool or float or imgaug.parameters.StochasticParameter, optional
            Whether to use (imagewise) the same sample(s) for all
            channels (``False``) or to sample value(s) for each channel (``True``).
            Setting this to ``True`` will therefore lead to different
            transformations per image *and* channel, otherwise only per image.
            If this value is a float ``p``, then for ``p`` percent of all images
            `per_channel` will be treated as ``True``.
            If it is a ``StochasticParameter`` it is expected to produce samples
            with values between ``0.0`` and ``1.0``, where values ``>0.5`` will
            lead to per-channel behaviour (i.e. same as ``True``).

        seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        name : None or str, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            Old name for parameter `seed`.
            Its usage will not yet cause a deprecation warning,
            but it is still recommended to use `seed` now.
            Outdated since 0.4.0.

        deterministic : bool, optional
            Deprecated since 0.4.0.
            See method ``to_deterministic()`` for an alternative and for
            details about what the "deterministic mode" actually does.
        """

        transform = iaa.Pepper(p, per_channel, seed, name, random_state, deterministic)
        augmented_image = transform(image=self.image)
        self.image = augmented_image

        return self

    def CoarsePepper(self, p=(0.02, 0.1), size_px=None, size_percent=None, per_channel=False, min_size=3, seed=None,
                     name=None, random_state='deprecated', deterministic='deprecated'):
        """
        Replace rectangular areas in images with black-ish pixel noise.

        **Supported dtypes**:

        See :class:`~imgaug.augmenters.arithmetic.ReplaceElementwise`.

        Parameters
        ----------
        p : float or tuple of float or list of float or imgaug.parameters.StochasticParameter, optional
            Probability of changing a pixel to pepper noise.

                * If a float, then that value will always be used as the
                probability.
                * If a tuple ``(a, b)``, then a probability will be sampled
                uniformly per image from the interval ``[a, b]``.
                * If a list, then a random value will be sampled from that list
                per image.
                * If a ``StochasticParameter``, then a lower-resolution mask will
                be sampled from that parameter per image. Any value ``>0.5`` in
                that mask will denote a spatial location that is to be replaced
                by pepper noise.

        size_px : int or tuple of int or imgaug.parameters.StochasticParameter, optional
            The size of the lower resolution image from which to sample the
            replacement mask in absolute pixel dimensions.
            Note that this means that *lower* values of this parameter lead to
            *larger* areas being replaced (as any pixel in the lower resolution
            image will correspond to a larger area at the original resolution).

                * If ``None`` then `size_percent` must be set.
                * If an integer, then that size will always be used for both height
                and width. E.g. a value of ``3`` would lead to a ``3x3`` mask,
                which is then upsampled to ``HxW``, where ``H`` is the image size
                and ``W`` the image width.
                * If a tuple ``(a, b)``, then two values ``M``, ``N`` will be
                sampled from the discrete interval ``[a..b]``. The mask
                will then be generated at size ``MxN`` and upsampled to ``HxW``.
                * If a ``StochasticParameter``, then this parameter will be used to
                determine the sizes. It is expected to be discrete.

        size_percent : float or tuple of float or imgaug.parameters.StochasticParameter, optional
            The size of the lower resolution image from which to sample the
            replacement mask *in percent* of the input image.
            Note that this means that *lower* values of this parameter lead to
            *larger* areas being replaced (as any pixel in the lower resolution
            image will correspond to a larger area at the original resolution).

                * If ``None`` then `size_px` must be set.
                * If a float, then that value will always be used as the percentage
                of the height and width (relative to the original size). E.g. for
                value ``p``, the mask will be sampled from ``(p*H)x(p*W)`` and
                later upsampled to ``HxW``.
                * If a tuple ``(a, b)``, then two values ``m``, ``n`` will be
                sampled from the interval ``(a, b)`` and used as the size
                fractions, i.e the mask size will be ``(m*H)x(n*W)``.
                * If a ``StochasticParameter``, then this parameter will be used to
                sample the percentage values. It is expected to be continuous.

        per_channel : bool or float or imgaug.parameters.StochasticParameter, optional
            Whether to use (imagewise) the same sample(s) for all
            channels (``False``) or to sample value(s) for each channel (``True``).
            Setting this to ``True`` will therefore lead to different
            transformations per image *and* channel, otherwise only per image.
            If this value is a float ``p``, then for ``p`` percent of all images
            `per_channel` will be treated as ``True``.
            If it is a ``StochasticParameter`` it is expected to produce samples
            with values between ``0.0`` and ``1.0``, where values ``>0.5`` will
            lead to per-channel behaviour (i.e. same as ``True``).

        min_size : int, optional
            Minimum size of the low resolution mask, both width and height. If
            `size_percent` or `size_px` leads to a lower value than this, `min_size`
            will be used instead. This should never have a value of less than 2,
            otherwise one may end up with a ``1x1`` low resolution mask, leading
            easily to the whole image being replaced.

        seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        name : None or str, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            Old name for parameter `seed`.
            Its usage will not yet cause a deprecation warning,
            but it is still recommended to use `seed` now.
            Outdated since 0.4.0.

        deterministic : bool, optional
            Deprecated since 0.4.0.
            See method ``to_deterministic()`` for an alternative and for
            details about what the "deterministic mode" actually does.
        """

        transform = iaa.CoarsePepper(p, size_px, size_percent, per_channel, min_size, seed, name, random_state,
                                     deterministic)
        augmented_image = transform(image=self.image)
        self.image = augmented_image

        return self

    def Cartoon(self, blur_ksize=(1, 5), segmentation_size=(0.8, 1.2), saturation=(1.5, 2.5),
                edge_prevalence=(0.9, 1.1), from_colorspace='RGB', seed=None, name=None, random_state='deprecated',
                deterministic='deprecated'):
        """
        Convert the style of images to a more cartoonish one.

        This augmenter was primarily designed for images with a size of ``200``
        to ``800`` pixels. Smaller or larger images may cause issues.

        Note that the quality of the results can currently not compete with
        learned style transfer, let alone human-made images. A lack of detected
        edges or also too many detected edges are probably the most significant
        drawbacks.

        Added in 0.4.0.

        **Supported dtypes**:

        See :func:`~imgaug.augmenters.artistic.stylize_cartoon`.

        Parameters
        ----------
        blur_ksize : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
            Median filter kernel size.
            See :func:`~imgaug.augmenters.artistic.stylize_cartoon` for details.

                * If ``number``: That value will be used for all images.
                * If ``tuple (a, b) of number``: A random value will be uniformly
                sampled per image from the interval ``[a, b)``.
                * If ``list``: A random value will be picked per image from the
                ``list``.
                * If ``StochasticParameter``: The parameter will be queried once
                per batch for ``(N,)`` values, where ``N`` is the number of
                images.

        segmentation_size : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
            Mean-Shift segmentation size multiplier.
            See :func:`~imgaug.augmenters.artistic.stylize_cartoon` for details.

                * If ``number``: That value will be used for all images.
                * If ``tuple (a, b) of number``: A random value will be uniformly
                sampled per image from the interval ``[a, b)``.
                * If ``list``: A random value will be picked per image from the
                ``list``.
                * If ``StochasticParameter``: The parameter will be queried once
                per batch for ``(N,)`` values, where ``N`` is the number of
                images.

        saturation : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
            Saturation multiplier.
            See :func:`~imgaug.augmenters.artistic.stylize_cartoon` for details.

                * If ``number``: That value will be used for all images.
                * If ``tuple (a, b) of number``: A random value will be uniformly
                sampled per image from the interval ``[a, b)``.
                * If ``list``: A random value will be picked per image from the
                ``list``.
                * If ``StochasticParameter``: The parameter will be queried once
                per batch for ``(N,)`` values, where ``N`` is the number of
                images.

        edge_prevalence : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
            Multiplier for the prevalence of edges.
            See :func:`~imgaug.augmenters.artistic.stylize_cartoon` for details.

                * If ``number``: That value will be used for all images.
                * If ``tuple (a, b) of number``: A random value will be uniformly
                sampled per image from the interval ``[a, b)``.
                * If ``list``: A random value will be picked per image from the
                ``list``.
                * If ``StochasticParameter``: The parameter will be queried once
                per batch for ``(N,)`` values, where ``N`` is the number of
                images.

        from_colorspace : str, optional
            The source colorspace. Use one of ``imgaug.augmenters.color.CSPACE_*``.
            Defaults to ``RGB``.

        seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        name : None or str, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            Old name for parameter `seed`.
            Its usage will not yet cause a deprecation warning,
            but it is still recommended to use `seed` now.
            Outdated since 0.4.0.

        deterministic : bool, optional
            Deprecated since 0.4.0.
            See method ``to_deterministic()`` for an alternative and for
            details about what the "deterministic mode" actually does.
        """

        transform = iaa.Cartoon(blur_ksize, segmentation_size, saturation, edge_prevalence, from_colorspace, seed, name,
                                random_state, deterministic)
        augmented_image = transform(image=self.image)
        self.image = augmented_image

        return self

    def BlendAlpha(self, factor=(0.0, 1.0), foreground=None, background=None, per_channel=False, seed=None, name=None,
                   random_state='deprecated', deterministic='deprecated'):
        """
        Alpha-blend two image sources using an alpha/opacity value.

        The two image sources can be imagined as branches.
        If a source is not given, it is automatically the same as the input.
        Let ``FG`` be the foreground branch and ``BG`` be the background branch.
        Then the result images are defined as ``factor * FG + (1-factor) * BG``,
        where ``factor`` is an overlay factor.

        .. note::

            It is not recommended to use ``BlendAlpha`` with augmenters
            that change the geometry of images (e.g. horizontal flips, affine
            transformations) if you *also* want to augment coordinates (e.g.
            keypoints, polygons, ...), as it is unclear which of the two
            coordinate results (foreground or background branch) should be used
            as the coordinates after augmentation.

            Currently, if ``factor >= 0.5`` (per image), the results of the
            foreground branch are used as the new coordinates, otherwise the
            results of the background branch.

        Added in 0.4.0. (Before that named `Alpha`.)

        **Supported dtypes**:

        See :func:`~imgaug.augmenters.blend.blend_alpha`.

        Parameters
        ----------
        factor : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
            Opacity of the results of the foreground branch. Values close to
            ``0.0`` mean that the results from the background branch (see
            parameter `background`) make up most of the final image.

                * If float, then that value will be used for all images.
                * If tuple ``(a, b)``, then a random value from the interval
                ``[a, b]`` will be sampled per image.
                * If a list, then a random value will be picked from that list per
                image.
                * If ``StochasticParameter``, then that parameter will be used to
                sample a value per image.

        foreground : None or imgaug.augmenters.meta.Augmenter or iterable of imgaug.augmenters.meta.Augmenter, optional
            Augmenter(s) that make up the foreground branch.
            High alpha values will show this branch's results.

                * If ``None``, then the input images will be reused as the output
                of the foreground branch.
                * If ``Augmenter``, then that augmenter will be used as the branch.
                * If iterable of ``Augmenter``, then that iterable will be
                converted into a ``Sequential`` and used as the augmenter.

        background : None or imgaug.augmenters.meta.Augmenter or iterable of imgaug.augmenters.meta.Augmenter, optional
            Augmenter(s) that make up the background branch.
            Low alpha values will show this branch's results.

                * If ``None``, then the input images will be reused as the output
                of the background branch.
                * If ``Augmenter``, then that augmenter will be used as the branch.
                * If iterable of ``Augmenter``, then that iterable will be
                converted into a ``Sequential`` and used as the augmenter.

        per_channel : bool or float or imgaug.parameters.StochasticParameter, optional
            Whether to use the same factor for all channels (``False``)
            or to sample a new value for each channel (``True``).
            If this value is a float ``p``, then for ``p`` percent of all images
            `per_channel` will be treated as True, otherwise as False.

        seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        name : None or str, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            Old name for parameter `seed`.
            Its usage will not yet cause a deprecation warning,
            but it is still recommended to use `seed` now.
            Outdated since 0.4.0.

        deterministic : bool, optional
            Deprecated since 0.4.0.
            See method ``to_deterministic()`` for an alternative and for
            details about what the "deterministic mode" actually does.
        """

        transform = iaa.BlendAlpha(factor, foreground, background, per_channel, seed, name, random_state, deterministic)
        augmented_image = transform(image=self.image)
        self.image = augmented_image

        return self

    def BlendAlphaMask(self, mask_generator, foreground=None, background=None, seed=None, name=None,
                       random_state='deprecated', deterministic='deprecated'):
        """
        Alpha-blend two image sources using non-binary masks generated per image.

        This augmenter queries for each image a mask generator to generate
        a ``(H,W)`` or ``(H,W,C)`` channelwise mask ``[0.0, 1.0]``, where
        ``H`` is the image height and ``W`` the width.
        The mask will then be used to alpha-blend pixel- and possibly channel-wise
        between a foreground branch of augmenters and a background branch.
        (Both branches default to the identity operation if not provided.)

        See also :class:`~imgaug.augmenters.blend.BlendAlpha`.

        .. note::

            It is not recommended to use ``BlendAlphaMask`` with augmenters
            that change the geometry of images (e.g. horizontal flips, affine
            transformations) if you *also* want to augment coordinates (e.g.
            keypoints, polygons, ...), as it is unclear which of the two
            coordinate results (foreground or background branch) should be used
            as the final output coordinates after augmentation.

            Currently, for keypoints the results of the
            foreground and background branch will be mixed. That means that for
            each coordinate the augmented result will be picked from the
            foreground or background branch based on the average alpha mask value
            at the corresponding spatial location.

            For bounding boxes, line strings and polygons, either all objects
            (on an image) of the foreground or all of the background branch will
            be used, based on the average over the whole alpha mask.

        Added in 0.4.0.

        **Supported dtypes**:

        See :func:`~imgaug.augmenters.blend.blend_alpha`.

        Parameters
        ----------
        mask_generator : IBatchwiseMaskGenerator
            A generator that will be queried per image to generate a mask.

        foreground : None or imgaug.augmenters.meta.Augmenter or iterable of imgaug.augmenters.meta.Augmenter, optional
            Augmenter(s) that make up the foreground branch.
            High alpha values will show this branch's results.

                * If ``None``, then the input images will be reused as the output
                of the foreground branch (i.e. identity function).
                * If ``Augmenter``, then that augmenter will be used as the branch.
                * If iterable of ``Augmenter``, then that iterable will be
                converted into a ``Sequential`` and used as the augmenter.

        background : None or imgaug.augmenters.meta.Augmenter or iterable of imgaug.augmenters.meta.Augmenter, optional
            Augmenter(s) that make up the background branch.
            Low alpha values will show this branch's results.

                * If ``None``, then the input images will be reused as the output
                of the background branch (i.e. identity function).
                * If ``Augmenter``, then that augmenter will be used as the branch.
                * If iterable of ``Augmenter``, then that iterable will be
                converted into a ``Sequential`` and used as the augmenter.

        seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        name : None or str, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            Old name for parameter `seed`.
            Its usage will not yet cause a deprecation warning,
            but it is still recommended to use `seed` now.
            Outdated since 0.4.0.

        deterministic : bool, optional
            Deprecated since 0.4.0.
            See method ``to_deterministic()`` for an alternative and for
            details about what the "deterministic mode" actually does.
        """

        transform = iaa.BlendAlphaMask(mask_generator, foreground, background, seed, name, random_state, deterministic)
        augmented_image = transform(image=self.image)
        self.image = augmented_image

        return self

    def BlendAlphaElementwise(self, factor=(0.0, 1.0), foreground=None, background=None, per_channel=False, seed=None,
                              name=None, random_state='deprecated', deterministic='deprecated'):
        """
        Alpha-blend two image sources using alpha/opacity values sampled per pixel.

        This is the same as :class:`BlendAlpha`, except that the opacity factor is
        sampled once per *pixel* instead of once per *image* (or a few times per
        image, if ``BlendAlpha.per_channel`` is set to ``True``).

        See :class:`BlendAlpha` for more details.

        This class is a wrapper around
        :class:`~imgaug.augmenters.blend.BlendAlphaMask`.

        .. note::

            Avoid using augmenters as children that affect pixel locations (e.g.
            horizontal flips). See
            :class:`~imgaug.augmenters.blend.BlendAlphaMask` for details.

        Added in 0.4.0. (Before that named `AlphaElementwise`.)

        **Supported dtypes**:

        See :class:`~imgaug.augmenters.blend.BlendAlphaMask`.

        Parameters
        ----------
        factor : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
            Opacity of the results of the foreground branch. Values close to
            ``0.0`` mean that the results from the background branch (see
            parameter `background`) make up most of the final image.

                * If float, then that value will be used for all images.
                * If tuple ``(a, b)``, then a random value from the interval
                ``[a, b]`` will be sampled per image.
                * If a list, then a random value will be picked from that list per
                image.
                * If ``StochasticParameter``, then that parameter will be used to
                sample a value per image.

        foreground : None or imgaug.augmenters.meta.Augmenter or iterable of imgaug.augmenters.meta.Augmenter, optional
            Augmenter(s) that make up the foreground branch.
            High alpha values will show this branch's results.

                * If ``None``, then the input images will be reused as the output
                of the foreground branch.
                * If ``Augmenter``, then that augmenter will be used as the branch.
                * If iterable of ``Augmenter``, then that iterable will be
                converted into a ``Sequential`` and used as the augmenter.

        background : None or imgaug.augmenters.meta.Augmenter or iterable of imgaug.augmenters.meta.Augmenter, optional
            Augmenter(s) that make up the background branch.
            Low alpha values will show this branch's results.

                * If ``None``, then the input images will be reused as the output
                of the background branch.
                * If ``Augmenter``, then that augmenter will be used as the branch.
                * If iterable of ``Augmenter``, then that iterable will be
                converted into a ``Sequential`` and used as the augmenter.

        per_channel : bool or float, optional
            Whether to use the same factor for all channels (``False``)
            or to sample a new value for each channel (``True``).
            If this value is a float ``p``, then for ``p`` percent of all images
            `per_channel` will be treated as True, otherwise as False.

        seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        name : None or str, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            Old name for parameter `seed`.
            Its usage will not yet cause a deprecation warning,
            but it is still recommended to use `seed` now.
            Outdated since 0.4.0.

        deterministic : bool, optional
            Deprecated since 0.4.0.
            See method ``to_deterministic()`` for an alternative and for
            details about what the "deterministic mode" actually does.
        """

        transform = iaa.BlendAlphaElementwise(factor, foreground, background, per_channel, seed, name, random_state,
                                              deterministic)
        augmented_image = transform(image=self.image)
        self.image = augmented_image

        return self

    def BlendAlphaSimplexNoise(self, foreground=None, background=None, per_channel=False, size_px_max=(2, 16),
                               upscale_method=None, iterations=(1, 3), aggregation_method='max', sigmoid=True,
                               sigmoid_thresh=None, seed=None, name=None, random_state='deprecated',
                               deterministic='deprecated'):
        """
        Alpha-blend two image sources using simplex noise alpha masks.

        The alpha masks are sampled using a simplex noise method, roughly creating
        connected blobs of 1s surrounded by 0s. If nearest neighbour
        upsampling is used, these blobs can be rectangular with sharp edges.

        Added in 0.4.0. (Before that named `SimplexNoiseAlpha`.)

        **Supported dtypes**:

        See :class:`~imgaug.augmenters.blend.BlendAlphaElementwise`.

        Parameters
        ----------
        foreground : None or imgaug.augmenters.meta.Augmenter or iterable of imgaug.augmenters.meta.Augmenter, optional
            Augmenter(s) that make up the foreground branch.
            High alpha values will show this branch's results.

                * If ``None``, then the input images will be reused as the output
                of the foreground branch.
                * If ``Augmenter``, then that augmenter will be used as the branch.
                * If iterable of ``Augmenter``, then that iterable will be
                converted into a ``Sequential`` and used as the augmenter.

        background : None or imgaug.augmenters.meta.Augmenter or iterable of imgaug.augmenters.meta.Augmenter, optional
            Augmenter(s) that make up the background branch.
            Low alpha values will show this branch's results.

                * If ``None``, then the input images will be reused as the output
                of the background branch.
                * If ``Augmenter``, then that augmenter will be used as the branch.
                * If iterable of ``Augmenter``, then that iterable will be
                converted into a ``Sequential`` and used as the augmenter.

        per_channel : bool or float, optional
            Whether to use the same factor for all channels (``False``)
            or to sample a new value for each channel (``True``).
            If this value is a float ``p``, then for ``p`` percent of all images
            `per_channel` will be treated as ``True``, otherwise as ``False``.

        size_px_max : int or tuple of int or list of int or imgaug.parameters.StochasticParameter, optional
            The simplex noise is always generated in a low resolution environment.
            This parameter defines the maximum size of that environment (in
            pixels). The environment is initialized at the same size as the input
            image and then downscaled, so that no side exceeds `size_px_max`
            (aspect ratio is kept).

                * If int, then that number will be used as the size for all
                iterations.
                * If tuple of two ``int`` s ``(a, b)``, then a value will be
                sampled per iteration from the discrete interval ``[a..b]``.
                * If a list of ``int`` s, then a value will be picked per iteration
                at random from that list.
                * If a ``StochasticParameter``, then a value will be sampled from
                that parameter per iteration.

        upscale_method : None or imgaug.ALL or str or list of str or imgaug.parameters.StochasticParameter, optional
            After generating the noise maps in low resolution environments, they
            have to be upscaled to the input image size. This parameter controls
            the upscaling method.

                * If ``None``, then either ``nearest`` or ``linear`` or ``cubic``
                is picked. Most weight is put on ``linear``, followed by
                ``cubic``.
                * If ``imgaug.ALL``, then either ``nearest`` or ``linear`` or
                ``area`` or ``cubic`` is picked per iteration (all same
                probability).
                * If a string, then that value will be used as the method (must be
                ``nearest`` or ``linear`` or ``area`` or ``cubic``).
                * If list of string, then a random value will be picked from that
                list per iteration.
                * If ``StochasticParameter``, then a random value will be sampled
                from that parameter per iteration.

        iterations : int or tuple of int or list of int or imgaug.parameters.StochasticParameter, optional
            How often to repeat the simplex noise generation process per image.

                * If ``int``, then that number will be used as the iterations for
                all images.
                * If tuple of two ``int`` s ``(a, b)``, then a value will be
                sampled per image from the discrete interval ``[a..b]``.
                * If a list of ``int`` s, then a value will be picked per image at
                random from that list.
                * If a ``StochasticParameter``, then a value will be sampled from
                that parameter per image.

        aggregation_method : imgaug.ALL or str or list of str or imgaug.parameters.StochasticParameter, optional
            The noise maps (from each iteration) are combined to one noise map
            using an aggregation process. This parameter defines the method used
            for that process. Valid methods are ``min``, ``max`` or ``avg``,
            where ``min`` combines the noise maps by taking the (elementwise)
            minimum over all iteration's results, ``max`` the (elementwise)
            maximum and ``avg`` the (elementwise) average.

                * If ``imgaug.ALL``, then a random value will be picked per image
                from the valid ones.
                * If a string, then that value will always be used as the method.
                * If a list of string, then a random value will be picked from
                that list per image.
                * If a ``StochasticParameter``, then a random value will be
                sampled from that paramter per image.

        sigmoid : bool or number, optional
            Whether to apply a sigmoid function to the final noise maps, resulting
            in maps that have more extreme values (close to 0.0 or 1.0).

                * If ``bool``, then a sigmoid will always (``True``) or never
                (``False``) be applied.
                * If a number ``p`` with ``0<=p<=1``, then a sigmoid will be
                applied to ``p`` percent of all final noise maps.

        sigmoid_thresh : None or number or tuple of number or imgaug.parameters.StochasticParameter, optional
            Threshold of the sigmoid, when applied. Thresholds above zero
            (e.g. ``5.0``) will move the saddle point towards the right, leading
            to more values close to 0.0.

                * If ``None``, then ``Normal(0, 5.0)`` will be used.
                * If number, then that threshold will be used for all images.
                * If tuple of two numbers ``(a, b)``, then a random value will
                be sampled per image from the interval ``[a, b]``.
                * If ``StochasticParameter``, then a random value will be sampled
                from that parameter per image.

        seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        name : None or str, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            Old name for parameter `seed`.
            Its usage will not yet cause a deprecation warning,
            but it is still recommended to use `seed` now.
            Outdated since 0.4.0.

        deterministic : bool, optional
            Deprecated since 0.4.0.
            See method ``to_deterministic()`` for an alternative and for
            details about what the "deterministic mode" actually does.
        """

        transform = iaa.BlendAlphaSimplexNoise(foreground, background, per_channel, size_px_max, upscale_method,
                                               iterations, aggregation_method, sigmoid, sigmoid_thresh, seed=None,
                                               name=None, random_state='deprecated', deterministic='deprecated')
        augmented_image = transform(image=self.image)
        self.image = augmented_image

        return self

    def BlendAlphaFrequencyNoise(self, exponent=(-4, 4), foreground=None, background=None, per_channel=False,
                                 size_px_max=(4, 16), upscale_method=None, iterations=(1, 3),
                                 aggregation_method=['avg', 'max'], sigmoid=0.5, sigmoid_thresh=None, seed=None,
                                 name=None, random_state='deprecated', deterministic='deprecated'):
        """
        Alpha-blend two image sources using frequency noise masks.

        The alpha masks are sampled using frequency noise of varying scales,
        which can sometimes create large connected blobs of ``1`` s surrounded
        by ``0`` s and other times results in smaller patterns. If nearest
        neighbour upsampling is used, these blobs can be rectangular with sharp
        edges.

        Added in 0.4.0. (Before that named `FrequencyNoiseAlpha`.)

        **Supported dtypes**:

        See :class:`~imgaug.augmenters.blend.BlendAlphaElementwise`.

        Parameters
        ----------
        exponent : number or tuple of number of list of number or imgaug.parameters.StochasticParameter, optional
            Exponent to use when scaling in the frequency domain.
            Sane values are in the range ``-4`` (large blobs) to ``4`` (small
            patterns). To generate cloud-like structures, use roughly ``-2``.

                * If number, then that number will be used as the exponent for all
                iterations.
                * If tuple of two numbers ``(a, b)``, then a value will be sampled
                per iteration from the interval ``[a, b]``.
                * If a list of numbers, then a value will be picked per iteration
                at random from that list.
                * If a ``StochasticParameter``, then a value will be sampled from
                that parameter per iteration.

        foreground : None or imgaug.augmenters.meta.Augmenter or iterable of imgaug.augmenters.meta.Augmenter, optional
            Augmenter(s) that make up the foreground branch.
            High alpha values will show this branch's results.

                * If ``None``, then the input images will be reused as the output
                of the foreground branch.
                * If ``Augmenter``, then that augmenter will be used as the branch.
                * If iterable of ``Augmenter``, then that iterable will be
                converted into a ``Sequential`` and used as the augmenter.

        background : None or imgaug.augmenters.meta.Augmenter or iterable of imgaug.augmenters.meta.Augmenter, optional
            Augmenter(s) that make up the background branch.
            Low alpha values will show this branch's results.

                * If ``None``, then the input images will be reused as the output
                of the background branch.
                * If ``Augmenter``, then that augmenter will be used as the branch.
                * If iterable of ``Augmenter``, then that iterable will be
                converted into a ``Sequential`` and used as the augmenter.

        per_channel : bool or float, optional
            Whether to use the same factor for all channels (``False``)
            or to sample a new value for each channel (``True``).
            If this value is a float ``p``, then for ``p`` percent of all images
            `per_channel` will be treated as ``True``, otherwise as ``False``.

        size_px_max : int or tuple of int or list of int or imgaug.parameters.StochasticParameter, optional
            The noise is generated in a low resolution environment.
            This parameter defines the maximum size of that environment (in
            pixels). The environment is initialized at the same size as the input
            image and then downscaled, so that no side exceeds `size_px_max`
            (aspect ratio is kept).

                * If ``int``, then that number will be used as the size for all
                iterations.
                * If tuple of two ``int`` s ``(a, b)``, then a value will be
                sampled per iteration from the discrete interval ``[a..b]``.
                * If a list of ``int`` s, then a value will be picked per
                iteration at random from that list.
                * If a ``StochasticParameter``, then a value will be sampled from
                that parameter per iteration.

        upscale_method : None or imgaug.ALL or str or list of str or imgaug.parameters.StochasticParameter, optional
            After generating the noise maps in low resolution environments, they
            have to be upscaled to the input image size. This parameter controls
            the upscaling method.

                * If ``None``, then either ``nearest`` or ``linear`` or ``cubic``
                is picked. Most weight is put on ``linear``, followed by
                ``cubic``.
                * If ``imgaug.ALL``, then either ``nearest`` or ``linear`` or
                ``area`` or ``cubic`` is picked per iteration (all same
                probability).
                * If string, then that value will be used as the method (must be
                ``nearest`` or ``linear`` or ``area`` or ``cubic``).
                * If list of string, then a random value will be picked from that
                list per iteration.
                * If ``StochasticParameter``, then a random value will be sampled
                from that parameter per iteration.

        iterations : int or tuple of int or list of int or imgaug.parameters.StochasticParameter, optional
            How often to repeat the simplex noise generation process per
            image.

                * If ``int``, then that number will be used as the iterations for
                all images.
                * If tuple of two ``int`` s ``(a, b)``, then a value will be
                sampled per image from the discrete interval ``[a..b]``.
                * If a list of ``int`` s, then a value will be picked per image at
                random from that list.
                * If a ``StochasticParameter``, then a value will be sampled from
                that parameter per image.

        aggregation_method : imgaug.ALL or str or list of str or imgaug.parameters.StochasticParameter, optional
            The noise maps (from each iteration) are combined to one noise map
            using an aggregation process. This parameter defines the method used
            for that process. Valid methods are ``min``, ``max`` or ``avg``,
            where 'min' combines the noise maps by taking the (elementwise) minimum
            over all iteration's results, ``max`` the (elementwise) maximum and
            ``avg`` the (elementwise) average.

                * If ``imgaug.ALL``, then a random value will be picked per image
                from the valid ones.
                * If a string, then that value will always be used as the method.
                * If a list of string, then a random value will be picked from
                that list per image.
                * If a ``StochasticParameter``, then a random value will be sampled
                from that parameter per image.

        sigmoid : bool or number, optional
            Whether to apply a sigmoid function to the final noise maps, resulting
            in maps that have more extreme values (close to ``0.0`` or ``1.0``).

                * If ``bool``, then a sigmoid will always (``True``) or never
                (``False``) be applied.
                * If a number ``p`` with ``0<=p<=1``, then a sigmoid will be applied to
                ``p`` percent of all final noise maps.

        sigmoid_thresh : None or number or tuple of number or imgaug.parameters.StochasticParameter, optional
            Threshold of the sigmoid, when applied. Thresholds above zero
            (e.g. ``5.0``) will move the saddle point towards the right, leading to
            more values close to ``0.0``.

                * If ``None``, then ``Normal(0, 5.0)`` will be used.
                * If number, then that threshold will be used for all images.
                * If tuple of two numbers ``(a, b)``, then a random value will
                be sampled per image from the range ``[a, b]``.
                * If ``StochasticParameter``, then a random value will be sampled
                from that parameter per image.

        seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        name : None or str, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            Old name for parameter `seed`.
            Its usage will not yet cause a deprecation warning,
            but it is still recommended to use `seed` now.
            Outdated since 0.4.0.

        deterministic : bool, optional
            Deprecated since 0.4.0.
            See method ``to_deterministic()`` for an alternative and for
            details about what the "deterministic mode" actually does.
        """

        transform = iaa.BlendAlphaFrequencyNoise(exponent, foreground, background, per_channel, size_px_max,
                                                 upscale_method, iterations, aggregation_method, sigmoid,
                                                 sigmoid_thresh, seed, name, random_state, deterministic)
        augmented_image = transform(image=self.image)
        self.image = augmented_image

        return self

    def BlendAlphaSomeColors(self, foreground=None, background=None, nb_bins=(5, 15), smoothness=(0.1, 0.3),
                             alpha=[0.0, 1.0], rotation_deg=(0, 360), from_colorspace='RGB', seed=None, name=None,
                             random_state='deprecated', deterministic='deprecated'):
        """
        Blend images from two branches using colorwise masks.

        This class generates masks that "mark" a few colors and replace the
        pixels within these colors with the results of the foreground branch.
        The remaining pixels are replaced with the results of the background
        branch (usually the identity function). That allows to e.g. selectively
        grayscale a few colors, while keeping other colors unchanged.

        This class is a thin wrapper around
        :class:`~imgaug.augmenters.blend.BlendAlphaMask` together with
        :class:`~imgaug.augmenters.blend.SomeColorsMaskGen`.

        .. note::

            The underlying mask generator will produce an ``AssertionError`` for
            batches that contain no images.

        .. note::

            Avoid using augmenters as children that affect pixel locations (e.g.
            horizontal flips). See
            :class:`~imgaug.augmenters.blend.BlendAlphaMask` for details.

        Added in 0.4.0.

        **Supported dtypes**:

        See :func:`~imgaug.augmenters.color.change_colorspaces_`.

        Parameters
        ----------
        foreground : None or imgaug.augmenters.meta.Augmenter or iterable of imgaug.augmenters.meta.Augmenter, optional
            Augmenter(s) that make up the foreground branch.
            High alpha values will show this branch's results.

                * If ``None``, then the input images will be reused as the output
                of the foreground branch.
                * If ``Augmenter``, then that augmenter will be used as the branch.
                * If iterable of ``Augmenter``, then that iterable will be
                converted into a ``Sequential`` and used as the augmenter.

        background : None or imgaug.augmenters.meta.Augmenter or iterable of imgaug.augmenters.meta.Augmenter, optional
            Augmenter(s) that make up the background branch.
            Low alpha values will show this branch's results.

                * If ``None``, then the input images will be reused as the output
                of the background branch.
                * If ``Augmenter``, then that augmenter will be used as the branch.
                * If iterable of ``Augmenter``, then that iterable will be
                converted into a ``Sequential`` and used as the augmenter.

        nb_bins : int or tuple of int or list of int or imgaug.parameters.StochasticParameter, optional
            See :class:`~imgaug.augmenters.blend.SomeColorsMaskGen`.

        smoothness : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
            See :class:`~imgaug.augmenters.blend.SomeColorsMaskGen`.

        alpha : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
            See :class:`~imgaug.augmenters.blend.SomeColorsMaskGen`.

        rotation_deg : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
            See :class:`~imgaug.augmenters.blend.SomeColorsMaskGen`.

        from_colorspace : str, optional
            See :class:`~imgaug.augmenters.blend.SomeColorsMaskGen`.

        seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        name : None or str, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            Old name for parameter `seed`.
            Its usage will not yet cause a deprecation warning,
            but it is still recommended to use `seed` now.
            Outdated since 0.4.0.

        deterministic : bool, optional
            Deprecated since 0.4.0.
            See method ``to_deterministic()`` for an alternative and for
            details about what the "deterministic mode" actually does.
        """

        transform = iaa.BlendAlphaSomeColors(foreground, background, nb_bins, smoothness, alpha, rotation_deg,
                                             from_colorspace, seed, name, random_state, deterministic)
        augmented_image = transform(image=self.image)
        self.image = augmented_image

        return self

    def BlendAlphaHorizontalLinearGradient(self, foreground=None, background=None, min_value=(0.0, 0.2),
                                           max_value=(0.8, 1.0), start_at=(0.0, 0.2), end_at=(0.8, 1.0), seed=None,
                                           name=None, random_state='deprecated', deterministic='deprecated'):
        """
        Blend images from two branches along a horizontal linear gradient.

        This class generates a horizontal linear gradient mask (i.e. usually a
        mask with low values on the left and high values on the right) and
        alphas-blends between foreground and background branch using that
        mask.

        This class is a thin wrapper around
        :class:`~imgaug.augmenters.blend.BlendAlphaMask` together with
        :class:`~imgaug.augmenters.blend.HorizontalLinearGradientMaskGen`.

        .. note::

            Avoid using augmenters as children that affect pixel locations (e.g.
            horizontal flips). See
            :class:`~imgaug.augmenters.blend.BlendAlphaMask` for details.

        Added in 0.4.0.

        **Supported dtypes**:

        See :class:`~imgaug.augmenters.blend.BlendAlphaMask`.

        Parameters
        ----------
        foreground : None or imgaug.augmenters.meta.Augmenter or iterable of imgaug.augmenters.meta.Augmenter, optional
            Augmenter(s) that make up the foreground branch.
            High alpha values will show this branch's results.

                * If ``None``, then the input images will be reused as the output
                of the foreground branch.
                * If ``Augmenter``, then that augmenter will be used as the branch.
                * If iterable of ``Augmenter``, then that iterable will be
                converted into a ``Sequential`` and used as the augmenter.

        background : None or imgaug.augmenters.meta.Augmenter or iterable of imgaug.augmenters.meta.Augmenter, optional
            Augmenter(s) that make up the background branch.
            Low alpha values will show this branch's results.

                * If ``None``, then the input images will be reused as the output
                of the background branch.
                * If ``Augmenter``, then that augmenter will be used as the branch.
                * If iterable of ``Augmenter``, then that iterable will be
                converted into a ``Sequential`` and used as the augmenter.

        min_value : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
            See :class:`~imgaug.augmenters.blend.HorizontalLinearGradientMaskGen`.

        max_value : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
            See :class:`~imgaug.augmenters.blend.HorizontalLinearGradientMaskGen`.

        start_at : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
            See :class:`~imgaug.augmenters.blend.HorizontalLinearGradientMaskGen`.

        end_at : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
            See :class:`~imgaug.augmenters.blend.HorizontalLinearGradientMaskGen`.

        seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        name : None or str, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            Old name for parameter `seed`.
            Its usage will not yet cause a deprecation warning,
            but it is still recommended to use `seed` now.
            Outdated since 0.4.0.

        deterministic : bool, optional
            Deprecated since 0.4.0.
            See method ``to_deterministic()`` for an alternative and for
            details about what the "deterministic mode" actually does.
        """

        transform = iaa.BlendAlphaHorizontalLinearGradient(foreground, background, min_value, max_value, start_at,
                                                           end_at, seed, name, random_state, deterministic)
        augmented_image = transform(image=self.image)
        self.image = augmented_image

        return self

    def BlendAlphaVerticalLinearGradient(self, foreground=None, background=None, min_value=(0.0, 0.2),
                                         max_value=(0.8, 1.0), start_at=(0.0, 0.2), end_at=(0.8, 1.0), seed=None,
                                         name=None, random_state='deprecated', deterministic='deprecated'):
        """
        Blend images from two branches along a vertical linear gradient.

        This class generates a vertical linear gradient mask (i.e. usually a
        mask with low values on the left and high values on the right) and
        alphas-blends between foreground and background branch using that
        mask.

        This class is a thin wrapper around
        :class:`~imgaug.augmenters.blend.BlendAlphaMask` together with
        :class:`~imgaug.augmenters.blend.VerticalLinearGradientMaskGen`.

        .. note::

            Avoid using augmenters as children that affect pixel locations (e.g.
            horizontal flips). See
            :class:`~imgaug.augmenters.blend.BlendAlphaMask` for details.

        Added in 0.4.0.

        **Supported dtypes**:

        See :class:`~imgaug.augmenters.blend.BlendAlphaMask`.

        Parameters
        ----------
        foreground : None or imgaug.augmenters.meta.Augmenter or iterable of imgaug.augmenters.meta.Augmenter, optional
            Augmenter(s) that make up the foreground branch.
            High alpha values will show this branch's results.

                * If ``None``, then the input images will be reused as the output
                of the foreground branch.
                * If ``Augmenter``, then that augmenter will be used as the branch.
                * If iterable of ``Augmenter``, then that iterable will be
                converted into a ``Sequential`` and used as the augmenter.

        background : None or imgaug.augmenters.meta.Augmenter or iterable of imgaug.augmenters.meta.Augmenter, optional
            Augmenter(s) that make up the background branch.
            Low alpha values will show this branch's results.

                * If ``None``, then the input images will be reused as the output
                of the background branch.
                * If ``Augmenter``, then that augmenter will be used as the branch.
                * If iterable of ``Augmenter``, then that iterable will be
                converted into a ``Sequential`` and used as the augmenter.

        min_value : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
            See :class:`~imgaug.augmenters.blend.VerticalLinearGradientMaskGen`.

        max_value : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
            See :class:`~imgaug.augmenters.blend.VerticalLinearGradientMaskGen`.

        start_at : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
            See :class:`~imgaug.augmenters.blend.VerticalLinearGradientMaskGen`.

        end_at : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
            See :class:`~imgaug.augmenters.blend.VerticalLinearGradientMaskGen`.

        seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        name : None or str, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            Old name for parameter `seed`.
            Its usage will not yet cause a deprecation warning,
            but it is still recommended to use `seed` now.
            Outdated since 0.4.0.

        deterministic : bool, optional
            Deprecated since 0.4.0.
            See method ``to_deterministic()`` for an alternative and for
            details about what the "deterministic mode" actually does.
        """

        transform = iaa.BlendAlphaVerticalLinearGradient(foreground, background, min_value, max_value, start_at, end_at,
                                                         seed, name, random_state, deterministic)
        augmented_image = transform(image=self.image)
        self.image = augmented_image

        return self

    def BlendAlphaRegularGrid(self, nb_rows, nb_cols, foreground=None, background=None, alpha=[0.0, 1.0], seed=None,
                              name=None, random_state='deprecated', deterministic='deprecated'):
        """
        Blend images from two branches according to a regular grid.

        This class generates for each image a mask that splits the image into a
        grid-like pattern of ``H`` rows and ``W`` columns. Each cell is then
        filled with an alpha value, sampled randomly per cell.

        The difference to :class:`AlphaBlendCheckerboard` is that this class
        samples random alpha values per grid cell, while in the checkerboard the
        alpha values follow a fixed pattern.

        This class is a thin wrapper around
        :class:`~imgaug.augmenters.blend.BlendAlphaMask` together with
        :class:`~imgaug.augmenters.blend.RegularGridMaskGen`.

        .. note::

            Avoid using augmenters as children that affect pixel locations (e.g.
            horizontal flips). See
            :class:`~imgaug.augmenters.blend.BlendAlphaMask` for details.

        Added in 0.4.0.

        **Supported dtypes**:

        See :class:`~imgaug.augmenters.blend.BlendAlphaMask`.

        Parameters
        ----------
        nb_rows : int or tuple of int or list of int or imgaug.parameters.StochasticParameter
            Number of rows of the checkerboard.
            See :class:`~imgaug.augmenters.blend.CheckerboardMaskGen` for details.

        nb_cols : int or tuple of int or list of int or imgaug.parameters.StochasticParameter
            Number of columns of the checkerboard. Analogous to `nb_rows`.
            See :class:`~imgaug.augmenters.blend.CheckerboardMaskGen` for details.

        foreground : None or imgaug.augmenters.meta.Augmenter or iterable of imgaug.augmenters.meta.Augmenter, optional
            Augmenter(s) that make up the foreground branch.
            High alpha values will show this branch's results.

                * If ``None``, then the input images will be reused as the output
                of the foreground branch.
                * If ``Augmenter``, then that augmenter will be used as the branch.
                * If iterable of ``Augmenter``, then that iterable will be
                converted into a ``Sequential`` and used as the augmenter.

        background : None or imgaug.augmenters.meta.Augmenter or iterable of imgaug.augmenters.meta.Augmenter, optional
            Augmenter(s) that make up the background branch.
            Low alpha values will show this branch's results.

                * If ``None``, then the input images will be reused as the output
                of the background branch.
                * If ``Augmenter``, then that augmenter will be used as the branch.
                * If iterable of ``Augmenter``, then that iterable will be
                converted into a ``Sequential`` and used as the augmenter.

        alpha : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
            Alpha value of each cell.

            * If ``number``: Exactly that value will be used for all images.
            * If ``tuple`` ``(a, b)``: A random value will be uniformly sampled
            per image from the interval ``[a, b]``.
            * If ``list``: A random value will be picked per image from that list.
            * If ``StochasticParameter``: That parameter will be queried once
            per batch for ``(N,)`` values -- one per image.

        seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        name : None or str, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            Old name for parameter `seed`.
            Its usage will not yet cause a deprecation warning,
            but it is still recommended to use `seed` now.
            Outdated since 0.4.0.

        deterministic : bool, optional
            Deprecated since 0.4.0.
            See method ``to_deterministic()`` for an alternative and for
            details about what the "deterministic mode" actually does.
        """

        transform = iaa.BlendAlphaRegularGrid(nb_rows, nb_cols, foreground, background, alpha, seed, name, random_state,
                                              deterministic)
        augmented_image = transform(image=self.image)
        self.image = augmented_image

        return self

    def BlendAlphaCheckerboard(self, nb_rows, nb_cols, foreground=None, background=None, seed=None, name=None,
                               random_state='deprecated', deterministic='deprecated'):
        """
        Blend images from two branches according to a checkerboard pattern.

        This class generates for each image a mask following a checkboard layout of
        ``H`` rows and ``W`` columns. Each cell is then filled with either
        ``1.0`` or ``0.0``. The cell at the top-left is always ``1.0``. Its right
        and bottom neighbour cells are ``0.0``. The 4-neighbours of any cell always
        have a value opposite to the cell's value (``0.0`` vs. ``1.0``).

        This class is a thin wrapper around
        :class:`~imgaug.augmenters.blend.BlendAlphaMask` together with
        :class:`~imgaug.augmenters.blend.CheckerboardMaskGen`.

        .. note::

            Avoid using augmenters as children that affect pixel locations (e.g.
            horizontal flips). See
            :class:`~imgaug.augmenters.blend.BlendAlphaMask` for details.

        Added in 0.4.0.

        **Supported dtypes**:

        See :class:`~imgaug.augmenters.blend.BlendAlphaMask`.

        Parameters
        ----------
        nb_rows : int or tuple of int or list of int or imgaug.parameters.StochasticParameter
            Number of rows of the checkerboard.
            See :class:`~imgaug.augmenters.blend.CheckerboardMaskGen` for details.

        nb_cols : int or tuple of int or list of int or imgaug.parameters.StochasticParameter
            Number of columns of the checkerboard. Analogous to `nb_rows`.
            See :class:`~imgaug.augmenters.blend.CheckerboardMaskGen` for details.

        foreground : None or imgaug.augmenters.meta.Augmenter or iterable of imgaug.augmenters.meta.Augmenter, optional
            Augmenter(s) that make up the foreground branch.
            High alpha values will show this branch's results.

                * If ``None``, then the input images will be reused as the output
                of the foreground branch.
                * If ``Augmenter``, then that augmenter will be used as the branch.
                * If iterable of ``Augmenter``, then that iterable will be
                converted into a ``Sequential`` and used as the augmenter.

        background : None or imgaug.augmenters.meta.Augmenter or iterable of imgaug.augmenters.meta.Augmenter, optional
            Augmenter(s) that make up the background branch.
            Low alpha values will show this branch's results.

                * If ``None``, then the input images will be reused as the output
                of the background branch.
                * If ``Augmenter``, then that augmenter will be used as the branch.
                * If iterable of ``Augmenter``, then that iterable will be
                converted into a ``Sequential`` and used as the augmenter.

        seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        name : None or str, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            Old name for parameter `seed`.
            Its usage will not yet cause a deprecation warning,
            but it is still recommended to use `seed` now.
            Outdated since 0.4.0.

        deterministic : bool, optional
            Deprecated since 0.4.0.
            See method ``to_deterministic()`` for an alternative and for
            details about what the "deterministic mode" actually does.
        """

        transform = iaa.BlendAlphaCheckerboard(nb_rows, nb_cols, foreground, background, seed, name, random_state,
                                               deterministic)
        augmented_image = transform(image=self.image)
        self.image = augmented_image

        return self

    def AverageBlur(self, k=(1, 7), seed=None, name=None, random_state='deprecated', deterministic='deprecated'):
        """
        Blur an image by computing simple means over neighbourhoods.

        The padding behaviour around the image borders is cv2's
        ``BORDER_REFLECT_101``.

        **Supported dtypes**:

            * ``uint8``: yes; fully tested
            * ``uint16``: yes; tested
            * ``uint32``: no (1)
            * ``uint64``: no (2)
            * ``int8``: yes; tested (3)
            * ``int16``: yes; tested
            * ``int32``: no (4)
            * ``int64``: no (5)
            * ``float16``: yes; tested (6)
            * ``float32``: yes; tested
            * ``float64``: yes; tested
            * ``float128``: no
            * ``bool``: yes; tested (7)

            - (1) rejected by ``cv2.blur()``
            - (2) loss of resolution in ``cv2.blur()`` (result is ``int32``)
            - (3) ``int8`` is mapped internally to ``int16``, ``int8`` itself
                leads to cv2 error "Unsupported combination of source format
                (=1), and buffer format (=4) in function 'getRowSumFilter'" in
                ``cv2``
            - (4) results too inaccurate
            - (5) loss of resolution in ``cv2.blur()`` (result is ``int32``)
            - (6) ``float16`` is mapped internally to ``float32``
            - (7) ``bool`` is mapped internally to ``float32``

        Parameters
        ----------
        k : int or tuple of int or tuple of tuple of int or imgaug.parameters.StochasticParameter or tuple of StochasticParameter, optional
            Kernel size to use.

                * If a single ``int``, then that value will be used for the height
                and width of the kernel.
                * If a tuple of two ``int`` s ``(a, b)``, then the kernel size will
                be sampled from the interval ``[a..b]``.
                * If a tuple of two tuples of ``int`` s ``((a, b), (c, d))``,
                then per image a random kernel height will be sampled from the
                interval ``[a..b]`` and a random kernel width will be sampled
                from the interval ``[c..d]``.
                * If a ``StochasticParameter``, then ``N`` samples will be drawn
                from that parameter per ``N`` input images, each representing
                the kernel size for the n-th image.
                * If a tuple ``(a, b)``, where either ``a`` or ``b`` is a tuple,
                then ``a`` and ``b`` will be treated according to the rules
                above. This leads to different values for height and width of
                the kernel.

        seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        name : None or str, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            Old name for parameter `seed`.
            Its usage will not yet cause a deprecation warning,
            but it is still recommended to use `seed` now.
            Outdated since 0.4.0.

        deterministic : bool, optional
            Deprecated since 0.4.0.
            See method ``to_deterministic()`` for an alternative and for
            details about what the "deterministic mode" actually does.
        """

        transform = iaa.AverageBlur(k, seed, name, random_state, deterministic)
        augmented_image = transform(image=self.image)
        self.image = augmented_image

        return self

    def BilateralBlur(self, d=(1, 9), sigma_color=(10, 250), sigma_space=(10, 250), seed=None, name=None,
                      random_state='deprecated', deterministic='deprecated'):
        """
        Blur/Denoise an image using a bilateral filter.

        Bilateral filters blur homogenous and textured areas, while trying to
        preserve edges.

        See
        http://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html#bilateralfilter
        for more information regarding the parameters.

        **Supported dtypes**:

            * ``uint8``: yes; not tested
            * ``uint16``: ?
            * ``uint32``: ?
            * ``uint64``: ?
            * ``int8``: ?
            * ``int16``: ?
            * ``int32``: ?
            * ``int64``: ?
            * ``float16``: ?
            * ``float32``: ?
            * ``float64``: ?
            * ``float128``: ?
            * ``bool``: ?

        Parameters
        ----------
        d : int or tuple of int or list of int or imgaug.parameters.StochasticParameter, optional
            Diameter of each pixel neighborhood with value range ``[1 .. inf)``.
            High values for `d` lead to significantly worse performance. Values
            equal or less than ``10`` seem to be good. Use ``<5`` for real-time
            applications.

                * If a single ``int``, then that value will be used for the
                diameter.
                * If a tuple of two ``int`` s ``(a, b)``, then the diameter will
                be a value sampled from the interval ``[a..b]``.
                * If a list, then a random value will be sampled from that list
                per image.
                * If a ``StochasticParameter``, then ``N`` samples will be drawn
                from that parameter per ``N`` input images, each representing
                the diameter for the n-th image. Expected to be discrete.

        sigma_color : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
            Filter sigma in the color space with value range ``[1, inf)``. A
            large value of the parameter means that farther colors within the
            pixel neighborhood (see `sigma_space`) will be mixed together,
            resulting in larger areas of semi-equal color.

                * If a single ``int``, then that value will be used for the
                diameter.
                * If a tuple of two ``int`` s ``(a, b)``, then the diameter will
                be a value sampled from the interval ``[a, b]``.
                * If a list, then a random value will be sampled from that list
                per image.
                * If a ``StochasticParameter``, then ``N`` samples will be drawn
                from that parameter per ``N`` input images, each representing
                the diameter for the n-th image. Expected to be discrete.

        sigma_space : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
            Filter sigma in the coordinate space with value range ``[1, inf)``. A
            large value of the parameter means that farther pixels will influence
            each other as long as their colors are close enough (see
            `sigma_color`).

                * If a single ``int``, then that value will be used for the
                diameter.
                * If a tuple of two ``int`` s ``(a, b)``, then the diameter will
                be a value sampled from the interval ``[a, b]``.
                * If a list, then a random value will be sampled from that list
                per image.
                * If a ``StochasticParameter``, then ``N`` samples will be drawn
                from that parameter per ``N`` input images, each representing
                the diameter for the n-th image. Expected to be discrete.

        seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        name : None or str, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            Old name for parameter `seed`.
            Its usage will not yet cause a deprecation warning,
            but it is still recommended to use `seed` now.
            Outdated since 0.4.0.

        deterministic : bool, optional
            Deprecated since 0.4.0.
            See method ``to_deterministic()`` for an alternative and for
            details about what the "deterministic mode" actually does.
        """

        transform = iaa.BilateralBlur(d, sigma_color, sigma_space, seed, name, random_state, deterministic)
        augmented_image = transform(image=self.image)
        self.image = augmented_image

        return self

    def RandAugment(self, n=2, m=(6, 12), cval=128, seed=None, name=None, random_state='deprecated',
                    deterministic='deprecated'):
        """
        Apply RandAugment to inputs as described in the corresponding paper.

        See paper::

            Cubuk et al.

            RandAugment: Practical automated data augmentation with a reduced
            search space

        .. note::

            The paper contains essentially no hyperparameters for the individual
            augmentation techniques. The hyperparameters used here come mostly
            from the official code repository, which however seems to only contain
            code for CIFAR10 and SVHN, not for ImageNet. So some guesswork was
            involved and a few of the hyperparameters were also taken from
            https://github.com/ildoonet/pytorch-randaugment/blob/master/RandAugment/augmentations.py .

            This implementation deviates from the code repository for all PIL
            enhance operations. In the repository these use a factor of
            ``0.1 + M*1.8/M_max``, which would lead to a factor of ``0.1`` for the
            weakest ``M`` of ``M=0``. For e.g. ``Brightness`` that would result in
            a basically black image. This definition is fine for AutoAugment (from
            where the code and hyperparameters are copied), which optimizes
            each transformation's ``M`` individually, but not for RandAugment,
            which uses a single fixed ``M``. We hence redefine these
            hyperparameters to ``1.0 + S * M * 0.9/M_max``, where ``S`` is
            randomly either ``1`` or ``-1``.

            We also note that it is not entirely clear which transformations
            were used in the ImageNet experiments. The paper lists some
            transformations in Figure 2, but names others in the text too (e.g.
            crops, flips, cutout). While Figure 2 lists the Identity function,
            this transformation seems to not appear in the repository (and in fact,
            the function ``randaugment(N, M)`` doesn't seem to exist in the
            repository either). So we also make a best guess here about what
            transformations might have been used.

        .. warning::

            This augmenter only works with image data, not e.g. bounding boxes.
            The used PIL-based affine transformations are not yet able to
            process non-image data. (This augmenter uses PIL-based affine
            transformations to ensure that outputs are as similar as possible
            to the paper's implementation.)

        Added in 0.4.0.

        **Supported dtypes**:

        minimum of (
            :class:`~imgaug.augmenters.flip.Fliplr`,
            :class:`~imgaug.augmenters.size.KeepSizeByResize`,
            :class:`~imgaug.augmenters.size.Crop`,
            :class:`~imgaug.augmenters.meta.Sequential`,
            :class:`~imgaug.augmenters.meta.SomeOf`,
            :class:`~imgaug.augmenters.meta.Identity`,
            :class:`~imgaug.augmenters.pillike.Autocontrast`,
            :class:`~imgaug.augmenters.pillike.Equalize`,
            :class:`~imgaug.augmenters.arithmetic.Invert`,
            :class:`~imgaug.augmenters.pillike.Affine`,
            :class:`~imgaug.augmenters.pillike.Posterize`,
            :class:`~imgaug.augmenters.pillike.Solarize`,
            :class:`~imgaug.augmenters.pillike.EnhanceColor`,
            :class:`~imgaug.augmenters.pillike.EnhanceContrast`,
            :class:`~imgaug.augmenters.pillike.EnhanceBrightness`,
            :class:`~imgaug.augmenters.pillike.EnhanceSharpness`,
            :class:`~imgaug.augmenters.arithmetic.Cutout`,
            :class:`~imgaug.augmenters.pillike.FilterBlur`,
            :class:`~imgaug.augmenters.pillike.FilterSmooth`
        )

        Parameters
        ----------
        n : int or tuple of int or list of int or imgaug.parameters.StochasticParameter or None, optional
            Parameter ``N`` in the paper, i.e. number of transformations to apply.
            The paper suggests ``N=2`` for ImageNet.
            See also parameter ``n`` in :class:`~imgaug.augmenters.meta.SomeOf`
            for more details.

            Note that horizontal flips (p=50%) and crops are always applied. This
            parameter only determines how many of the other transformations
            are applied per image.


        m : int or tuple of int or list of int or imgaug.parameters.StochasticParameter or None, optional
            Parameter ``M`` in the paper, i.e. magnitude/severity/strength of the
            applied transformations in interval ``[0 .. 30]`` with ``M=0`` being
            the weakest. The paper suggests for ImageNet ``M=9`` in case of
            ResNet-50 and ``M=28`` in case of EfficientNet-B7.
            This implementation uses a default value of ``(6, 12)``, i.e. the
            value is uniformly sampled per image from the interval ``[6 .. 12]``.
            This ensures greater diversity of transformations than using a single
            fixed value.

            * If ``int``: That value will always be used.
            * If ``tuple`` ``(a, b)``: A random value will be uniformly sampled per
            image from the discrete interval ``[a .. b]``.
            * If ``list``: A random value will be picked from the list per image.
            * If ``StochasticParameter``: For ``B`` images in a batch, ``B`` values
            will be sampled per augmenter (provided the augmenter is dependent
            on the magnitude).

        cval : number or tuple of number or list of number or imgaug.ALL or imgaug.parameters.StochasticParameter, optional
            The constant value to use when filling in newly created pixels.
            See parameter `fillcolor` in
            :class:`~imgaug.augmenters.pillike.Affine` for details.

            The paper's repository uses an RGB value of ``125, 122, 113``.
            This implementation uses a single intensity value of ``128``, which
            should work better for cases where input images don't have exactly
            ``3`` channels or come from a different dataset than used by the
            paper.

        seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        name : None or str, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            Old name for parameter `seed`.
            Its usage will not yet cause a deprecation warning,
            but it is still recommended to use `seed` now.
            Outdated since 0.4.0.

        deterministic : bool, optional
            Deprecated since 0.4.0.
            See method ``to_deterministic()`` for an alternative and for
            details about what the "deterministic mode" actually does.
        """
        transform = iaa.RandAugment(n, m, cval, seed, name, random_state, deterministic)
        augmented_image = transform(image=self.image)
        self.image = augmented_image

        return self

    def WithColorspace(self, to_colorspace, from_colorspace='RGB', children=None, seed=None, name=None,
                       random_state='deprecated', deterministic='deprecated'):
        """
        Apply child augmenters within a specific colorspace.

        This augumenter takes a source colorspace A and a target colorspace B
        as well as children C. It changes images from A to B, then applies the
        child augmenters C and finally changes the colorspace back from B to A.
        See also ChangeColorspace() for more.

        **Supported dtypes**:

        See :func:`~imgaug.augmenters.color.change_colorspaces_`.

        Parameters
        ----------
        to_colorspace : str
            See :func:`~imgaug.augmenters.color.change_colorspace_`.

        from_colorspace : str, optional
            See :func:`~imgaug.augmenters.color.change_colorspace_`.

        children : imgaug.augmenters.meta.Augmenter or list of imgaug.augmenters.meta.Augmenter or None, optional
            One or more augmenters to apply to converted images.

        seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        name : None or str, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            Old name for parameter `seed`.
            Its usage will not yet cause a deprecation warning,
            but it is still recommended to use `seed` now.
            Outdated since 0.4.0.

        deterministic : bool, optional
            Deprecated since 0.4.0.
            See method ``to_deterministic()`` for an alternative and for
            details about what the "deterministic mode" actually does.
        """

        transform = iaa.WithColorspace(to_colorspace, from_colorspace, children, seed, name, random_state,
                                       deterministic)
        augmented_image = transform(image=self.image)
        self.image = augmented_image

        return self

    def WithBrightnessChannels(self, children=None, to_colorspace=['YCrCb', 'HSV', 'HLS', 'Lab', 'Luv', 'YUV'],
                               from_colorspace='RGB', seed=None, name=None, random_state='deprecated',
                               deterministic='deprecated'):
        """
        Augmenter to apply child augmenters to brightness-related image channels.

        This augmenter first converts an image to a random colorspace containing a
        brightness-related channel (e.g. ``V`` in ``HSV``), then extracts that
        channel and applies its child augmenters to this one channel. Afterwards,
        it reintegrates the augmented channel into the full image and converts
        back to the input colorspace.

        Added in 0.4.0.

        **Supported dtypes**:

        See :func:`~imgaug.augmenters.color.change_colorspaces_`.

        Parameters
        ----------
        children : imgaug.augmenters.meta.Augmenter or list of imgaug.augmenters.meta.Augmenter or None, optional
            One or more augmenters to apply to the brightness channels.
            They receive images with a single channel and have to modify these.

        to_colorspace : imgaug.ALL or str or list of str or imgaug.parameters.StochasticParameter, optional
            Colorspace in which to extract the brightness-related channels.
            Currently, ``imgaug.augmenters.color.CSPACE_YCrCb``, ``CSPACE_HSV``,
            ``CSPACE_HLS``, ``CSPACE_Lab``, ``CSPACE_Luv``, ``CSPACE_YUV``,
            ``CSPACE_CIE`` are supported.

                * If ``imgaug.ALL``: Will pick imagewise a random colorspace from
                all supported colorspaces.
                * If ``str``: Will always use this colorspace.
                * If ``list`` or ``str``: Will pick imagewise a random colorspace
                from this list.
                * If :class:`~imgaug.parameters.StochasticParameter`:
                A parameter that will be queried once per batch to generate
                all target colorspaces. Expected to return strings matching the
                ``CSPACE_*`` constants.

        from_colorspace : str, optional
            See :func:`~imgaug.augmenters.color.change_colorspace_`.

        seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        name : None or str, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            Old name for parameter `seed`.
            Its usage will not yet cause a deprecation warning,
            but it is still recommended to use `seed` now.
            Outdated since 0.4.0.

        deterministic : bool, optional
            Deprecated since 0.4.0.
            See method ``to_deterministic()`` for an alternative and for
            details about what the "deterministic mode" actually does.
        """

        transform = iaa.WithBrightnessChannels(children, to_colorspace, from_colorspace, seed, name, random_state,
                                               deterministic)
        augmented_image = transform(image=self.image)
        self.image = augmented_image

        return self

    def MultiplyAndAddToBrightness(self, mul=(0.7, 1.3), add=(-30, 30),
                                   to_colorspace=['YCrCb', 'HSV', 'HLS', 'Lab', 'Luv', 'YUV'], from_colorspace='RGB',
                                   random_order=True, seed=None, name=None, random_state='deprecated',
                                   deterministic='deprecated'):
        """
        Multiply and add to the brightness channels of input images.

        This is a wrapper around :class:`WithBrightnessChannels` and hence
        performs internally the same projection to random colorspaces.

        Added in 0.4.0.

        **Supported dtypes**:

        See :class:`~imgaug.augmenters.color.WithBrightnessChannels`.

        Parameters
        ----------
        mul : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
            See :class:`~imgaug.augmenters.airthmetic.Multiply`.

        add : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
            See :class:`~imgaug.augmenters.airthmetic.Add`.

        to_colorspace : imgaug.ALL or str or list of str or imgaug.parameters.StochasticParameter, optional
            See :class:`~imgaug.augmenters.color.WithBrightnessChannels`.

        from_colorspace : str, optional
            See :class:`~imgaug.augmenters.color.WithBrightnessChannels`.

        random_order : bool, optional
            Whether to apply the add and multiply operations in random
            order (``True``). If ``False``, this augmenter will always first
            multiply and then add.

        seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        name : None or str, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            Old name for parameter `seed`.
            Its usage will not yet cause a deprecation warning,
            but it is still recommended to use `seed` now.
            Outdated since 0.4.0.

        deterministic : bool, optional
            Deprecated since 0.4.0.
            See method ``to_deterministic()`` for an alternative and for
            details about what the "deterministic mode" actually does.
        """
        transform = iaa.MultiplyAndAddToBrightness(mul, add, to_colorspace, from_colorspace, random_order, seed, name,
                                                   random_state, deterministic)
        augmented_image = transform(image=self.image)
        self.image = augmented_image

        return self

    def MultiplyBrightness(self, mul=(0.7, 1.3), to_colorspace=['YCrCb', 'HSV', 'HLS', 'Lab', 'Luv', 'YUV'],
                           from_colorspace='RGB', seed=None, name=None, random_state='deprecated',
                           deterministic='deprecated'):
        """
        Multiply the brightness channels of input images.

        This is a wrapper around :class:`WithBrightnessChannels` and hence
        performs internally the same projection to random colorspaces.

        Added in 0.4.0.

        **Supported dtypes**:

        See :class:`~imgaug.augmenters.color.MultiplyAndAddToBrightness`.

        Parameters
        ----------
        mul : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
            See :class:`~imgaug.augmenters.airthmetic.Multiply`.

        to_colorspace : imgaug.ALL or str or list of str or imgaug.parameters.StochasticParameter, optional
            See :class:`~imgaug.augmenters.color.WithBrightnessChannels`.

        from_colorspace : str, optional
            See :class:`~imgaug.augmenters.color.WithBrightnessChannels`.

        seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        name : None or str, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            Old name for parameter `seed`.
            Its usage will not yet cause a deprecation warning,
            but it is still recommended to use `seed` now.
            Outdated since 0.4.0.

        deterministic : bool, optional
            Deprecated since 0.4.0.
            See method ``to_deterministic()`` for an alternative and for
            details about what the "deterministic mode" actually does.
        """

        transform = iaa.MultiplyBrightness(mul, to_colorspace, from_colorspace, seed, name, random_state, deterministic)
        augmented_image = transform(image=self.image)
        self.image = augmented_image

        return self

    def AddToBrightness(self, add=(-30, 30), to_colorspace=['YCrCb', 'HSV', 'HLS', 'Lab', 'Luv', 'YUV'],
                        from_colorspace='RGB', seed=None, name=None, random_state='deprecated',
                        deterministic='deprecated'):
        """
        Add to the brightness channels of input images.

        This is a wrapper around :class:`WithBrightnessChannels` and hence
        performs internally the same projection to random colorspaces.

        Added in 0.4.0.

        **Supported dtypes**:

        See :class:`~imgaug.augmenters.color.MultiplyAndAddToBrightness`.

        Parameters
        ----------
        add : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
            See :class:`~imgaug.augmenters.airthmetic.Add`.

        to_colorspace : imgaug.ALL or str or list of str or imgaug.parameters.StochasticParameter, optional
            See :class:`~imgaug.augmenters.color.WithBrightnessChannels`.

        from_colorspace : str, optional
            See :class:`~imgaug.augmenters.color.WithBrightnessChannels`.

        seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        name : None or str, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            Old name for parameter `seed`.
            Its usage will not yet cause a deprecation warning,
            but it is still recommended to use `seed` now.
            Outdated since 0.4.0.

        deterministic : bool, optional
            Deprecated since 0.4.0.
            See method ``to_deterministic()`` for an alternative and for
            details about what the "deterministic mode" actually does.
        """

        transform = iaa.AddToBrightness(add, to_colorspace, from_colorspace, seed, name, random_state, deterministic)
        augmented_image = transform(image=self.image)
        self.image = augmented_image

        return self

    def WithHueAndSaturation(self, children=None, from_colorspace='RGB', seed=None, name=None,
                             random_state='deprecated', deterministic='deprecated'):
        """
        Apply child augmenters to hue and saturation channels.

        This augumenter takes an image in a source colorspace, converts
        it to HSV, extracts the H (hue) and S (saturation) channels,
        applies the provided child augmenters to these channels
        and finally converts back to the original colorspace.

        The image array generated by this augmenter and provided to its children
        is in ``int16`` (**sic!** only augmenters that can handle ``int16`` arrays
        can be children!). The hue channel is mapped to the value
        range ``[0, 255]``. Before converting back to the source colorspace, the
        saturation channel's values are clipped to ``[0, 255]``. A modulo operation
        is applied to the hue channel's values, followed by a mapping from
        ``[0, 255]`` to ``[0, 180]`` (and finally the colorspace conversion).

        **Supported dtypes**:

        See :func:`~imgaug.augmenters.color.change_colorspaces_`.

        Parameters
        ----------
        from_colorspace : str, optional
            See :func:`~imgaug.augmenters.color.change_colorspace_`.

        children : imgaug.augmenters.meta.Augmenter or list of imgaug.augmenters.meta.Augmenter or None, optional
            One or more augmenters to apply to converted images.
            They receive ``int16`` images with two channels (hue, saturation)
            and have to modify these.

        seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        name : None or str, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            Old name for parameter `seed`.
            Its usage will not yet cause a deprecation warning,
            but it is still recommended to use `seed` now.
            Outdated since 0.4.0.

        deterministic : bool, optional
            Deprecated since 0.4.0.
            See method ``to_deterministic()`` for an alternative and for
            details about what the "deterministic mode" actually does.
        """

        transform = iaa.WithHueAndSaturation(children, from_colorspace, seed, name, random_state, deterministic)
        augmented_image = transform(image=self.image)
        self.image = augmented_image

        return self

    def MultiplyHueAndSaturation(self, mul=None, mul_hue=None, mul_saturation=None, per_channel=False,
                                 from_colorspace='RGB', seed=None, name=None, random_state='deprecated',
                                 deterministic='deprecated'):
        """
        Multipy hue and saturation by random values.

        The augmenter first transforms images to HSV colorspace, then multiplies
        the pixel values in the H and S channels and afterwards converts back to
        RGB.

        This augmenter is a wrapper around ``WithHueAndSaturation``.

        **Supported dtypes**:

        See :class:`~imgaug.augmenters.color.WithHueAndSaturation`.

        Parameters
        ----------
        mul : None or number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
            Multiplier with which to multiply all hue *and* saturation values of
            all pixels.
            It is expected to be in the range ``-10.0`` to ``+10.0``.
            Note that values of ``0.0`` or lower will remove all saturation.

                * If this is ``None``, `mul_hue` and/or `mul_saturation`
                may be set to values other than ``None``.
                * If a number, then that multiplier will be used for all images.
                * If a tuple ``(a, b)``, then a value from the continuous
                range ``[a, b]`` will be sampled per image.
                * If a list, then a random value will be sampled from that list
                per image.
                * If a StochasticParameter, then a value will be sampled from that
                parameter per image.

        mul_hue : None or number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
            Multiplier with which to multiply all hue values.
            This is expected to be in the range ``-10.0`` to ``+10.0`` and will
            automatically be projected to an angular representation using
            ``(hue/255) * (360/2)`` (OpenCV's hue representation is in the
            range ``[0, 180]`` instead of ``[0, 360]``).
            Only this or `mul` may be set, not both.

                * If this and `mul_saturation` are both ``None``, `mul` may
                be set to a non-``None`` value.
                * If a number, then that multiplier will be used for all images.
                * If a tuple ``(a, b)``, then a value from the continuous
                range ``[a, b]`` will be sampled per image.
                * If a list, then a random value will be sampled from that list
                per image.
                * If a StochasticParameter, then a value will be sampled from that
                parameter per image.

        mul_saturation : None or number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
            Multiplier with which to multiply all saturation values.
            It is expected to be in the range ``0.0`` to ``+10.0``.
            Only this or `mul` may be set, not both.

                * If this and `mul_hue` are both ``None``, `mul` may
                be set to a non-``None`` value.
                * If a number, then that value will be used for all images.
                * If a tuple ``(a, b)``, then a value from the continuous
                range ``[a, b]`` will be sampled per image.
                * If a list, then a random value will be sampled from that list
                per image.
                * If a StochasticParameter, then a value will be sampled from that
                parameter per image.

        per_channel : bool or float, optional
            Whether to sample per image only one value from `mul` and use it for
            both hue and saturation (``False``) or to sample independently one
            value for hue and one for saturation (``True``).
            If this value is a float ``p``, then for ``p`` percent of all images
            `per_channel` will be treated as ``True``, otherwise as ``False``.

            This parameter has no effect if `mul_hue` and/or `mul_saturation`
            are used instead of `mul`.

        from_colorspace : str, optional
            See :func:`~imgaug.augmenters.color.change_colorspace_`.

        seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        name : None or str, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            Old name for parameter `seed`.
            Its usage will not yet cause a deprecation warning,
            but it is still recommended to use `seed` now.
            Outdated since 0.4.0.

        deterministic : bool, optional
            Deprecated since 0.4.0.
            See method ``to_deterministic()`` for an alternative and for
            details about what the "deterministic mode" actually does.
        """

        transform = iaa.MultiplyHueAndSaturation(mul, mul_hue, mul_saturation, per_channel, from_colorspace, seed, name,
                                                 random_state, deterministic)
        augmented_image = transform(image=self.image)
        self.image = augmented_image

        return self

    def MultiplyHue(self, mul=(-3.0, 3.0), from_colorspace='RGB', seed=None, name=None, random_state='deprecated',
                    deterministic='deprecated'):
        """
        Multiply the hue of images by random values.

        The augmenter first transforms images to HSV colorspace, then multiplies
        the pixel values in the H channel and afterwards converts back to
        RGB.

        This augmenter is a shortcut for ``MultiplyHueAndSaturation(mul_hue=...)``.

        **Supported dtypes**:

        See :class:`~imgaug.augmenters.color.MultiplyHueAndSaturation`.

        Parameters
        ----------
        mul : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
            Multiplier with which to multiply all hue values.
            This is expected to be in the range ``-10.0`` to ``+10.0`` and will
            automatically be projected to an angular representation using
            ``(hue/255) * (360/2)`` (OpenCV's hue representation is in the
            range ``[0, 180]`` instead of ``[0, 360]``).
            Only this or `mul` may be set, not both.

                * If a number, then that multiplier will be used for all images.
                * If a tuple ``(a, b)``, then a value from the continuous
                range ``[a, b]`` will be sampled per image.
                * If a list, then a random value will be sampled from that list
                per image.
                * If a StochasticParameter, then a value will be sampled from that
                parameter per image.

        from_colorspace : str, optional
            See :func:`~imgaug.augmenters.color.change_colorspace_`.

        seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        name : None or str, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            Old name for parameter `seed`.
            Its usage will not yet cause a deprecation warning,
            but it is still recommended to use `seed` now.
            Outdated since 0.4.0.

        deterministic : bool, optional
            Deprecated since 0.4.0.
            See method ``to_deterministic()`` for an alternative and for
            details about what the "deterministic mode" actually does.
        """
        transform = iaa.MultiplyHue(mul, from_colorspace, seed, name, random_state, deterministic)
        augmented_image = transform(image=self.image)
        self.image = augmented_image

        return self

    def MultiplySaturation(self, mul=(0.0, 3.0), from_colorspace='RGB', seed=None, name=None, random_state='deprecated',
                           deterministic='deprecated'):
        """
        Multiply the saturation of images by random values.

        The augmenter first transforms images to HSV colorspace, then multiplies
        the pixel values in the H channel and afterwards converts back to
        RGB.

        This augmenter is a shortcut for
        ``MultiplyHueAndSaturation(mul_saturation=...)``.

        **Supported dtypes**:

        See :class:`~imgaug.augmenters.color.MultiplyHueAndSaturation`.

        Parameters
        ----------
        mul : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
            Multiplier with which to multiply all saturation values.
            It is expected to be in the range ``0.0`` to ``+10.0``.

                * If a number, then that value will be used for all images.
                * If a tuple ``(a, b)``, then a value from the continuous
                range ``[a, b]`` will be sampled per image.
                * If a list, then a random value will be sampled from that list
                per image.
                * If a StochasticParameter, then a value will be sampled from that
                parameter per image.

        from_colorspace : str, optional
            See :func:`~imgaug.augmenters.color.change_colorspace_`.

        seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        name : None or str, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            Old name for parameter `seed`.
            Its usage will not yet cause a deprecation warning,
            but it is still recommended to use `seed` now.
            Outdated since 0.4.0.

        deterministic : bool, optional
            Deprecated since 0.4.0.
            See method ``to_deterministic()`` for an alternative and for
            details about what the "deterministic mode" actually does.
        """

        transform = iaa.MultiplySaturation(mul, from_colorspace, seed, name, random_state, deterministic)
        augmented_image = transform(image=self.image)
        self.image = augmented_image

        return self

    def RemoveSaturation(self, mul=1, from_colorspace='RGB', seed=None, name=None, random_state='deprecated',
                         deterministic='deprecated'):
        """
        Decrease the saturation of images by varying degrees.

        This creates images looking similar to :class:`Grayscale`.

        This augmenter is the same as ``MultiplySaturation((0.0, 1.0))``.

        Added in 0.4.0.

        **Supported dtypes**:

        See :class:`~imgaug.augmenters.color.MultiplySaturation`.

        Parameters
        ----------
        mul : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
            *Inverse* multiplier to use for the saturation values.
            High values denote stronger color removal. E.g. ``1.0`` will remove
            all saturation, ``0.0`` will remove nothing.
            Expected value range is ``[0.0, 1.0]``.

                * If a number, then that value will be used for all images.
                * If a tuple ``(a, b)``, then a value from the continuous
                range ``[a, b]`` will be sampled per image.
                * If a list, then a random value will be sampled from that list
                per image.
                * If a StochasticParameter, then a value will be sampled from that
                parameter per image.

        from_colorspace : str, optional
            See :func:`~imgaug.augmenters.color.change_colorspace_`.

        seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        name : None or str, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            Old name for parameter `seed`.
            Its usage will not yet cause a deprecation warning,
            but it is still recommended to use `seed` now.
            Outdated since 0.4.0.

        deterministic : bool, optional
            Deprecated since 0.4.0.
            See method ``to_deterministic()`` for an alternative and for
            details about what the "deterministic mode" actually does.
        """

        transform = iaa.RemoveSaturation(mul, from_colorspace, seed, name, random_state, deterministic)
        augmented_image = transform(image=self.image)
        self.image = augmented_image

        return self

    def AddToHueAndSaturation(self, value=None, value_hue=None, value_saturation=None, per_channel=False,
                              from_colorspace='RGB', seed=None, name=None, random_state='deprecated',
                              deterministic='deprecated'):
        """
        Increases or decreases hue and saturation by random values.

        The augmenter first transforms images to HSV colorspace, then adds random
        values to the H and S channels and afterwards converts back to RGB.

        This augmenter is faster than using ``WithHueAndSaturation`` in combination
        with ``Add``.

        TODO add float support

        **Supported dtypes**:

        See :func:`~imgaug.augmenters.color.change_colorspace_`.

        Parameters
        ----------
        value : None or int or tuple of int or list of int or imgaug.parameters.StochasticParameter, optional
            Value to add to the hue *and* saturation of all pixels.
            It is expected to be in the range ``-255`` to ``+255``.

                * If this is ``None``, `value_hue` and/or `value_saturation`
                may be set to values other than ``None``.
                * If an integer, then that value will be used for all images.
                * If a tuple ``(a, b)``, then a value from the discrete
                range ``[a, b]`` will be sampled per image.
                * If a list, then a random value will be sampled from that list
                per image.
                * If a StochasticParameter, then a value will be sampled from that
                parameter per image.

        value_hue : None or int or tuple of int or list of int or imgaug.parameters.StochasticParameter, optional
            Value to add to the hue of all pixels.
            This is expected to be in the range ``-255`` to ``+255`` and will
            automatically be projected to an angular representation using
            ``(hue/255) * (360/2)`` (OpenCV's hue representation is in the
            range ``[0, 180]`` instead of ``[0, 360]``).
            Only this or `value` may be set, not both.

                * If this and `value_saturation` are both ``None``, `value` may
                be set to a non-``None`` value.
                * If an integer, then that value will be used for all images.
                * If a tuple ``(a, b)``, then a value from the discrete
                range ``[a, b]`` will be sampled per image.
                * If a list, then a random value will be sampled from that list
                per image.
                * If a StochasticParameter, then a value will be sampled from that
                parameter per image.

        value_saturation : None or int or tuple of int or list of int or imgaug.parameters.StochasticParameter, optional
            Value to add to the saturation of all pixels.
            It is expected to be in the range ``-255`` to ``+255``.
            Only this or `value` may be set, not both.

                * If this and `value_hue` are both ``None``, `value` may
                be set to a non-``None`` value.
                * If an integer, then that value will be used for all images.
                * If a tuple ``(a, b)``, then a value from the discrete
                range ``[a, b]`` will be sampled per image.
                * If a list, then a random value will be sampled from that list
                per image.
                * If a StochasticParameter, then a value will be sampled from that
                parameter per image.

        per_channel : bool or float, optional
            Whether to sample per image only one value from `value` and use it for
            both hue and saturation (``False``) or to sample independently one
            value for hue and one for saturation (``True``).
            If this value is a float ``p``, then for ``p`` percent of all images
            `per_channel` will be treated as ``True``, otherwise as ``False``.

            This parameter has no effect is `value_hue` and/or `value_saturation`
            are used instead of `value`.

        from_colorspace : str, optional
            See :func:`~imgaug.augmenters.color.change_colorspace_`.

        seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        name : None or str, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            Old name for parameter `seed`.
            Its usage will not yet cause a deprecation warning,
            but it is still recommended to use `seed` now.
            Outdated since 0.4.0.

        deterministic : bool, optional
            Deprecated since 0.4.0.
            See method ``to_deterministic()`` for an alternative and for
            details about what the "deterministic mode" actually does.
        """

        transform = iaa.AddToHueAndSaturation(value, value_hue, value_saturation, per_channel, from_colorspace, seed,
                                              name, random_state, deterministic)
        augmented_image = transform(image=self.image)
        self.image = augmented_image

        return self

    def AddToHue(self, value=(-255, 255), from_colorspace='RGB', seed=None, name=None, random_state='deprecated',
                 deterministic='deprecated'):
        """
        Add random values to the hue of images.

        The augmenter first transforms images to HSV colorspace, then adds random
        values to the H channel and afterwards converts back to RGB.

        If you want to change both the hue and the saturation, it is recommended
        to use ``AddToHueAndSaturation`` as otherwise the image will be
        converted twice to HSV and back to RGB.

        This augmenter is a shortcut for ``AddToHueAndSaturation(value_hue=...)``.

        **Supported dtypes**:

        See :class:`~imgaug.augmenters.color.AddToHueAndSaturation`.

        Parameters
        ----------
        value : None or int or tuple of int or list of int or imgaug.parameters.StochasticParameter, optional
            Value to add to the hue of all pixels.
            This is expected to be in the range ``-255`` to ``+255`` and will
            automatically be projected to an angular representation using
            ``(hue/255) * (360/2)`` (OpenCV's hue representation is in the
            range ``[0, 180]`` instead of ``[0, 360]``).

                * If an integer, then that value will be used for all images.
                * If a tuple ``(a, b)``, then a value from the discrete
                range ``[a, b]`` will be sampled per image.
                * If a list, then a random value will be sampled from that list
                per image.
                * If a StochasticParameter, then a value will be sampled from that
                parameter per image.

        from_colorspace : str, optional
            See :func:`~imgaug.augmenters.color.change_colorspace_`.

        seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        name : None or str, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            Old name for parameter `seed`.
            Its usage will not yet cause a deprecation warning,
            but it is still recommended to use `seed` now.
            Outdated since 0.4.0.

        deterministic : bool, optional
            Deprecated since 0.4.0.
            See method ``to_deterministic()`` for an alternative and for
            details about what the "deterministic mode" actually does.
        """
        transform = iaa.AddToHue(value, from_colorspace, seed, name, random_state, deterministic)
        augmented_image = transform(image=self.image)
        self.image = augmented_image

        return self

    def AddToSaturation(self, value=(-75, 75), from_colorspace='RGB', seed=None, name=None, random_state='deprecated',
                        deterministic='deprecated'):
        """
        Add random values to the saturation of images.

        The augmenter first transforms images to HSV colorspace, then adds random
        values to the S channel and afterwards converts back to RGB.

        If you want to change both the hue and the saturation, it is recommended
        to use ``AddToHueAndSaturation`` as otherwise the image will be
        converted twice to HSV and back to RGB.

        This augmenter is a shortcut for
        ``AddToHueAndSaturation(value_saturation=...)``.

        **Supported dtypes**:

        See :class:`~imgaug.augmenters.color.AddToHueAndSaturation`.

        Parameters
        ----------
        value : None or int or tuple of int or list of int or imgaug.parameters.StochasticParameter, optional
            Value to add to the saturation of all pixels.
            It is expected to be in the range ``-255`` to ``+255``.

                * If an integer, then that value will be used for all images.
                * If a tuple ``(a, b)``, then a value from the discrete
                range ``[a, b]`` will be sampled per image.
                * If a list, then a random value will be sampled from that list
                per image.
                * If a StochasticParameter, then a value will be sampled from that
                parameter per image.

        from_colorspace : str, optional
            See :func:`~imgaug.augmenters.color.change_colorspace_`.

        seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        name : None or str, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            Old name for parameter `seed`.
            Its usage will not yet cause a deprecation warning,
            but it is still recommended to use `seed` now.
            Outdated since 0.4.0.

        deterministic : bool, optional
            Deprecated since 0.4.0.
            See method ``to_deterministic()`` for an alternative and for
            details about what the "deterministic mode" actually does.
        """

        transform = iaa.AddToSaturation(value, from_colorspace, seed, name, random_state, deterministic)
        augmented_image = transform(image=self.image)
        self.image = augmented_image

        return self

    def ChangeColorspace(self, to_colorspace, from_colorspace='RGB', alpha=1.0, seed=None, name=None,
                         random_state='deprecated', deterministic='deprecated'):
        """
        Augmenter to change the colorspace of images.

        .. note::

            This augmenter is not tested. Some colorspaces might work, others
            might not.

        ..note::

            This augmenter tries to project the colorspace value range on
            0-255. It outputs dtype=uint8 images.

        **Supported dtypes**:

        See :func:`~imgaug.augmenters.color.change_colorspace_`.

        Parameters
        ----------
        to_colorspace : str or list of str or imgaug.parameters.StochasticParameter
            The target colorspace.
            Allowed strings are: ``RGB``, ``BGR``, ``GRAY``, ``CIE``, ``YCrCb``,
            ``HSV``, ``HLS``, ``Lab``, ``Luv``.
            These are also accessible via
            ``imgaug.augmenters.color.CSPACE_<NAME>``,
            e.g. ``imgaug.augmenters.CSPACE_YCrCb``.

                * If a string, it must be among the allowed colorspaces.
                * If a list, it is expected to be a list of strings, each one
                being an allowed colorspace. A random element from the list
                will be chosen per image.
                * If a StochasticParameter, it is expected to return string. A new
                sample will be drawn per image.

        from_colorspace : str, optional
            The source colorspace (of the input images).
            See `to_colorspace`. Only a single string is allowed.

        alpha : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
            The alpha value of the new colorspace when overlayed over the
            old one. A value close to 1.0 means that mostly the new
            colorspace is visible. A value close to 0.0 means, that mostly the
            old image is visible.

                * If an int or float, exactly that value will be used.
                * If a tuple ``(a, b)``, a random value from the range
                ``a <= x <= b`` will be sampled per image.
                * If a list, then a random value will be sampled from that list
                per image.
                * If a StochasticParameter, a value will be sampled from the
                parameter per image.

        seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        name : None or str, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            Old name for parameter `seed`.
            Its usage will not yet cause a deprecation warning,
            but it is still recommended to use `seed` now.
            Outdated since 0.4.0.

        deterministic : bool, optional
            Deprecated since 0.4.0.
            See method ``to_deterministic()`` for an alternative and for
            details about what the "deterministic mode" actually does.

        """
        transform = iaa.ChangeColorspace(to_colorspace, from_colorspace, alpha, seed, name, random_state, deterministic)
        augmented_image = transform(image=self.image)
        self.image = augmented_image

        return self

    def ChangeColorTemperature(self, kelvin=(1000, 11000), from_colorspace='RGB', seed=None, name=None,
                               random_state='deprecated', deterministic='deprecated'):
        """
        Change the temperature to a provided Kelvin value.

        Low Kelvin values around ``1000`` to ``4000`` will result in red, yellow
        or orange images. Kelvin values around ``10000`` to ``40000`` will result
        in progressively darker blue tones.

        Color temperatures taken from
        `<http://www.vendian.org/mncharity/dir3/blackbody/UnstableURLs/bbr_color.html>`_

        Basic method to change color temperatures taken from
        `<https://stackoverflow.com/a/11888449>`_

        Added in 0.4.0.

        **Supported dtypes**:

        See :func:`~imgaug.augmenters.color.change_color_temperatures_`.

        Parameters
        ----------
        kelvin : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
            Temperature in Kelvin. The temperatures of images will be modified to
            this value. Must be in the interval ``[1000, 40000]``.

                * If a number, exactly that value will always be used.
                * If a ``tuple`` ``(a, b)``, then a value from the
                interval ``[a, b]`` will be sampled per image.
                * If a ``list``, then a random value will be sampled from that
                ``list`` per image.
                * If a ``StochasticParameter``, then a value will be sampled per
                image from that parameter.
        """
        transform = iaa.ChangeColorTemperature(kelvin, from_colorspace, seed, name, random_state, deterministic)
        augmented_image = transform(image=self.image)
        self.image = augmented_image

        return self

    def KMeansColorQuantization(self, n_colors=(2, 16), from_colorspace='RGB', to_colorspace=['RGB', 'Lab'],
                                max_size=128, interpolation='linear', seed=None, name=None, random_state='deprecated',
                                deterministic='deprecated'):
        """
        Quantize colors using k-Means clustering.

        This "collects" the colors from the input image, groups them into
        ``k`` clusters using k-Means clustering and replaces the colors in the
        input image using the cluster centroids.

        This is slower than ``UniformColorQuantization``, but adapts dynamically
        to the color range in the input image.

        .. note::

            This augmenter expects input images to be either grayscale
            or to have 3 or 4 channels and use colorspace `from_colorspace`. If
            images have 4 channels, it is assumed that the 4th channel is an alpha
            channel and it will not be quantized.

        **Supported dtypes**:

        if (image size <= max_size):

            minimum of (
                :class:`~imgaug.augmenters.color.ChangeColorspace`,
                :func:`~imgaug.augmenters.color.quantize_kmeans`
            )

        if (image size > max_size):

            minimum of (
                :class:`~imgaug.augmenters.color.ChangeColorspace`,
                :func:`~imgaug.augmenters.color.quantize_kmeans`,
                :func:`~imgaug.imgaug.imresize_single_image`
            )

        Parameters
        ----------
        n_colors : int or tuple of int or list of int or imgaug.parameters.StochasticParameter, optional
            Target number of colors in the generated output image.
            This corresponds to the number of clusters in k-Means, i.e. ``k``.
            Sampled values below ``2`` will always be clipped to ``2``.

                * If a number, exactly that value will always be used.
                * If a tuple ``(a, b)``, then a value from the discrete
                interval ``[a..b]`` will be sampled per image.
                * If a list, then a random value will be sampled from that list
                per image.
                * If a ``StochasticParameter``, then a value will be sampled per
                image from that parameter.

        to_colorspace : None or str or list of str or imgaug.parameters.StochasticParameter
            The colorspace in which to perform the quantization.
            See :func:`~imgaug.augmenters.color.change_colorspace_` for valid values.
            This will be ignored for grayscale input images.

                * If ``None`` the colorspace of input images will not be changed.
                * If a string, it must be among the allowed colorspaces.
                * If a list, it is expected to be a list of strings, each one
                being an allowed colorspace. A random element from the list
                will be chosen per image.
                * If a StochasticParameter, it is expected to return string. A new
                sample will be drawn per image.

        from_colorspace : str, optional
            The colorspace of the input images.
            See `to_colorspace`. Only a single string is allowed.

        max_size : int or None, optional
            Maximum image size at which to perform the augmentation.
            If the width or height of an image exceeds this value, it will be
            downscaled before running the augmentation so that the longest side
            matches `max_size`.
            This is done to speed up the augmentation. The final output image has
            the same size as the input image. Use ``None`` to apply no downscaling.

        interpolation : int or str, optional
            Interpolation method to use during downscaling when `max_size` is
            exceeded. Valid methods are the same as in
            :func:`~imgaug.imgaug.imresize_single_image`.

        seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        name : None or str, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            Old name for parameter `seed`.
            Its usage will not yet cause a deprecation warning,
            but it is still recommended to use `seed` now.
            Outdated since 0.4.0.

        deterministic : bool, optional
            Deprecated since 0.4.0.
            See method ``to_deterministic()`` for an alternative and for
            details about what the "deterministic mode" actually does.
        """
        transform = iaa.KMeansColorQuantization(n_colors, from_colorspace, to_colorspace, max_size, interpolation, seed,
                                                name, random_state, deterministic)
        augmented_image = transform(image=self.image)
        self.image = augmented_image

        return self

    def UniformColorQuantization(self, n_colors=(2, 16), from_colorspace='RGB', to_colorspace=None, max_size=None,
                                 interpolation='linear', seed=None, name=None, random_state='deprecated',
                                 deterministic='deprecated'):
        """
        Quantize colors into N bins with regular distance.

        For ``uint8`` images the equation is ``floor(v/q)*q + q/2`` with
        ``q = 256/N``, where ``v`` is a pixel intensity value and ``N`` is
        the target number of colors after quantization.

        This augmenter is faster than ``KMeansColorQuantization``, but the
        set of possible output colors is constant (i.e. independent of the
        input images). It may produce unsatisfying outputs for input images
        that are made up of very similar colors.

        .. note::

            This augmenter expects input images to be either grayscale
            or to have 3 or 4 channels and use colorspace `from_colorspace`. If
            images have 4 channels, it is assumed that the 4th channel is an alpha
            channel and it will not be quantized.

        **Supported dtypes**:

        if (image size <= max_size):

            minimum of (
                :class:`~imgaug.augmenters.color.ChangeColorspace`,
                :func:`~imgaug.augmenters.color.quantize_uniform_`
            )

        if (image size > max_size):

            minimum of (
                :class:`~imgaug.augmenters.color.ChangeColorspace`,
                :func:`~imgaug.augmenters.color.quantize_uniform_`,
                :func:`~imgaug.imgaug.imresize_single_image`
            )

        Parameters
        ----------
        n_colors : int or tuple of int or list of int or imgaug.parameters.StochasticParameter, optional
            Target number of colors to use in the generated output image.

                * If a number, exactly that value will always be used.
                * If a tuple ``(a, b)``, then a value from the discrete
                interval ``[a..b]`` will be sampled per image.
                * If a list, then a random value will be sampled from that list
                per image.
                * If a ``StochasticParameter``, then a value will be sampled per
                image from that parameter.

        to_colorspace : None or str or list of str or imgaug.parameters.StochasticParameter
            The colorspace in which to perform the quantization.
            See :func:`~imgaug.augmenters.color.change_colorspace_` for valid values.
            This will be ignored for grayscale input images.

                * If ``None`` the colorspace of input images will not be changed.
                * If a string, it must be among the allowed colorspaces.
                * If a list, it is expected to be a list of strings, each one
                being an allowed colorspace. A random element from the list
                will be chosen per image.
                * If a StochasticParameter, it is expected to return string. A new
                sample will be drawn per image.

        from_colorspace : str, optional
            The colorspace of the input images.
            See `to_colorspace`. Only a single string is allowed.

        max_size : None or int, optional
            Maximum image size at which to perform the augmentation.
            If the width or height of an image exceeds this value, it will be
            downscaled before running the augmentation so that the longest side
            matches `max_size`.
            This is done to speed up the augmentation. The final output image has
            the same size as the input image. Use ``None`` to apply no downscaling.

        interpolation : int or str, optional
            Interpolation method to use during downscaling when `max_size` is
            exceeded. Valid methods are the same as in
            :func:`~imgaug.imgaug.imresize_single_image`.

        seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        name : None or str, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            Old name for parameter `seed`.
            Its usage will not yet cause a deprecation warning,
            but it is still recommended to use `seed` now.
            Outdated since 0.4.0.

        deterministic : bool, optional
            Deprecated since 0.4.0.
            See method ``to_deterministic()`` for an alternative and for
            details about what the "deterministic mode" actually does.
        """
        transform = iaa.UniformColorQuantization(n_colors, from_colorspace, to_colorspace, max_size, interpolation,
                                                 seed, name, random_state, deterministic)
        augmented_image = transform(image=self.image)
        self.image = augmented_image

        return self

    def UniformColorQuantizationToNBits(self, nb_bits=(1, 8), from_colorspace='RGB', to_colorspace=None, max_size=None,
                                        interpolation='linear', seed=None, name=None, random_state='deprecated',
                                        deterministic='deprecated'):
        """
        Quantize images by setting ``8-B`` bits of each component to zero.

        This augmenter sets the ``8-B`` highest frequency (rightmost) bits of
        each array component to zero. For ``B`` bits this is equivalent to
        changing each component's intensity value ``v`` to
        ``v' = v & (2**(8-B) - 1)``, e.g. for ``B=3`` this results in
        ``v' = c & ~(2**(3-1) - 1) = c & ~3 = c & ~0000 0011 = c & 1111 1100``.

        This augmenter behaves for ``B`` similarly to
        ``UniformColorQuantization(2**B)``, but quantizes each bin with interval
        ``(a, b)`` to ``a`` instead of to ``a + (b-a)/2``.

        This augmenter is comparable to :func:`PIL.ImageOps.posterize`.

        .. note::

            This augmenter expects input images to be either grayscale
            or to have 3 or 4 channels and use colorspace `from_colorspace`. If
            images have 4 channels, it is assumed that the 4th channel is an alpha
            channel and it will not be quantized.

        Added in 0.4.0.

        **Supported dtypes**:

        if (image size <= max_size):

            minimum of (
                :class:`~imgaug.augmenters.color.ChangeColorspace`,
                :func:`~imgaug.augmenters.color.quantize_uniform`
            )

        if (image size > max_size):

            minimum of (
                :class:`~imgaug.augmenters.color.ChangeColorspace`,
                :func:`~imgaug.augmenters.color.quantize_uniform`,
                :func:`~imgaug.imgaug.imresize_single_image`
            )

        Parameters
        ----------
        nb_bits : int or tuple of int or list of int or imgaug.parameters.StochasticParameter, optional
            Number of bits to keep in each image's array component.

                * If a number, exactly that value will always be used.
                * If a tuple ``(a, b)``, then a value from the discrete
                interval ``[a..b]`` will be sampled per image.
                * If a list, then a random value will be sampled from that list
                per image.
                * If a ``StochasticParameter``, then a value will be sampled per
                image from that parameter.

        to_colorspace : None or str or list of str or imgaug.parameters.StochasticParameter
            The colorspace in which to perform the quantization.
            See :func:`~imgaug.augmenters.color.change_colorspace_` for valid values.
            This will be ignored for grayscale input images.

                * If ``None`` the colorspace of input images will not be changed.
                * If a string, it must be among the allowed colorspaces.
                * If a list, it is expected to be a list of strings, each one
                being an allowed colorspace. A random element from the list
                will be chosen per image.
                * If a StochasticParameter, it is expected to return string. A new
                sample will be drawn per image.

        from_colorspace : str, optional
            The colorspace of the input images.
            See `to_colorspace`. Only a single string is allowed.

        max_size : None or int, optional
            Maximum image size at which to perform the augmentation.
            If the width or height of an image exceeds this value, it will be
            downscaled before running the augmentation so that the longest side
            matches `max_size`.
            This is done to speed up the augmentation. The final output image has
            the same size as the input image. Use ``None`` to apply no downscaling.

        interpolation : int or str, optional
            Interpolation method to use during downscaling when `max_size` is
            exceeded. Valid methods are the same as in
            :func:`~imgaug.imgaug.imresize_single_image`.

        seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        name : None or str, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            Old name for parameter `seed`.
            Its usage will not yet cause a deprecation warning,
            but it is still recommended to use `seed` now.
            Outdated since 0.4.0.

        deterministic : bool, optional
            Deprecated since 0.4.0.
            See method ``to_deterministic()`` for an alternative and for
            details about what the "deterministic mode" actually does.
        """
        transform = iaa.UniformColorQuantizationToNBits(nb_bits, from_colorspace, to_colorspace, max_size,
                                                        interpolation, seed, name, random_state, deterministic)
        augmented_image = transform(image=self.image)
        self.image = augmented_image

        return self

    def GammaContrast(self, gamma=(0.7, 1.7), per_channel=False, seed=None, name=None, random_state='deprecated',
                      deterministic='deprecated'):
        """
        Adjust image contrast by scaling pixel values to ``255*((v/255)**gamma)``.

        Values in the range ``gamma=(0.5, 2.0)`` seem to be sensible.

        **Supported dtypes**:

        See :func:`~imgaug.augmenters.contrast.adjust_contrast_gamma`.

        Parameters
        ----------
        gamma : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
            Exponent for the contrast adjustment. Higher values darken the image.

                * If a number, then that value will be used for all images.
                * If a tuple ``(a, b)``, then a value from the range ``[a, b]``
                will be used per image.
                * If a list, then a random value will be sampled from that list
                per image.
                * If a ``StochasticParameter``, then a value will be sampled per
                image from that parameter.

        per_channel : bool or float, optional
            Whether to use the same value for all channels (``False``) or to
            sample a new value for each channel (``True``). If this value is a
            float ``p``, then for ``p`` percent of all images `per_channel` will
            be treated as ``True``, otherwise as ``False``.

        seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        name : None or str, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            Old name for parameter `seed`.
            Its usage will not yet cause a deprecation warning,
            but it is still recommended to use `seed` now.
            Outdated since 0.4.0.

        deterministic : bool, optional
            Deprecated since 0.4.0.
            See method ``to_deterministic()`` for an alternative and for
            details about what the "deterministic mode" actually does.
        """
        transform = iaa.GammaContrast(gamma, per_channel, seed, name, random_state, deterministic)
        augmented_image = transform(image=self.image)
        self.image = augmented_image

        return self

    def SigmoidContrast(self, gain=(5, 6), cutoff=(0.3, 0.6), per_channel=False, seed=None, name=None,
                        random_state='deprecated', deterministic='deprecated'):
        """
        Adjust image contrast to ``255*1/(1+exp(gain*(cutoff-I_ij/255)))``.

        Values in the range ``gain=(5, 20)`` and ``cutoff=(0.25, 0.75)`` seem to
        be sensible.

        A combination of ``gain=5.5`` and ``cutof=0.45`` is fairly close to
        the identity function.

        **Supported dtypes**:

        See :func:`~imgaug.augmenters.contrast.adjust_contrast_sigmoid`.

        Parameters
        ----------
        gain : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
            Multiplier for the sigmoid function's output.
            Higher values lead to quicker changes from dark to light pixels.

                * If a number, then that value will be used for all images.
                * If a tuple ``(a, b)``, then a value from the interval
                ``[a, b]`` will be sampled uniformly per image.
                * If a list, then a random value will be sampled from that list
                per image.
                * If a ``StochasticParameter``, then a value will be sampled per
                image from that parameter.

        cutoff : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
            Cutoff that shifts the sigmoid function in horizontal direction.
            Higher values mean that the switch from dark to light pixels happens
            later, i.e. the pixels will remain darker.

                * If a number, then that value will be used for all images.
                * If a tuple ``(a, b)``, then a value from the range ``[a, b]``
                will be used per image.
                * If a list, then a random value will be sampled from that list
                per image.
                * If a ``StochasticParameter``, then a value will be sampled per
                image from that parameter.

        per_channel : bool or float, optional
            Whether to use the same value for all channels (``False``) or to
            sample a new value for each channel (``True``). If this value is a
            float ``p``, then for ``p`` percent of all images `per_channel` will
            be treated as ``True``, otherwise as ``False``.

        seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        name : None or str, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            Old name for parameter `seed`.
            Its usage will not yet cause a deprecation warning,
            but it is still recommended to use `seed` now.
            Outdated since 0.4.0.

        deterministic : bool, optional
            Deprecated since 0.4.0.
            See method ``to_deterministic()`` for an alternative and for
            details about what the "deterministic mode" actually does.
        """
        transform = iaa.SigmoidContrast(gain, cutoff, per_channel, seed, name, random_state, deterministic)
        augmented_image = transform(image=self.image)
        self.image = augmented_image

        return self

    def LogContrast(self, gain=(0.4, 1.6), per_channel=False, seed=None, name=None, random_state='deprecated',
                    deterministic='deprecated'):
        """
        Adjust image contrast by scaling pixels to ``255*gain*log_2(1+v/255)``.

        This augmenter is fairly similar to
        ``imgaug.augmenters.arithmetic.Multiply``.

        **Supported dtypes**:

        See :func:`~imgaug.augmenters.contrast.adjust_contrast_log`.

        Parameters
        ----------
        gain : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
            Multiplier for the logarithm result. Values around ``1.0`` lead to a
            contrast-adjusted images. Values above ``1.0`` quickly lead to
            partially broken images due to exceeding the datatype's value range.

                * If a number, then that value will be used for all images.
                * If a tuple ``(a, b)``, then a value from the interval ``[a, b]``
                will uniformly sampled be used per image.
                * If a list, then a random value will be sampled from that list
                per image.
                * If a ``StochasticParameter``, then a value will be sampled per
                image from that parameter.

        per_channel : bool or float, optional
            Whether to use the same value for all channels (``False``) or to
            sample a new value for each channel (``True``). If this value is a
            float ``p``, then for ``p`` percent of all images `per_channel` will
            be treated as ``True``, otherwise as ``False``.

        seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        name : None or str, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            Old name for parameter `seed`.
            Its usage will not yet cause a deprecation warning,
            but it is still recommended to use `seed` now.
            Outdated since 0.4.0.

        deterministic : bool, optional
            Deprecated since 0.4.0.
            See method ``to_deterministic()`` for an alternative and for
            details about what the "deterministic mode" actually does.
        """
        transform = iaa.LogContrast(gain, per_channel, seed, name, random_state, deterministic)
        augmented_image = transform(image=self.image)
        self.image = augmented_image

        return self

    def LinearContrast(self, alpha=(0.6, 1.4), per_channel=False, seed=None, name=None, random_state='deprecated',
                       deterministic='deprecated'):
        """
        Adjust contrast by scaling each pixel to ``127 + alpha*(v-127)``.

        **Supported dtypes**:

        See :func:`~imgaug.augmenters.contrast.adjust_contrast_linear`.

        Parameters
        ----------
        alpha : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
            Multiplier to linearly pronounce (``>1.0``), dampen (``0.0`` to
            ``1.0``) or invert (``<0.0``) the difference between each pixel value
            and the dtype's center value, e.g. ``127`` for ``uint8``.

                * If a number, then that value will be used for all images.
                * If a tuple ``(a, b)``, then a value from the interval ``[a, b]``
                will be used per image.
                * If a list, then a random value will be sampled from that list
                per image.
                * If a ``StochasticParameter``, then a value will be sampled per
                image from that parameter.

        per_channel : bool or float, optional
            Whether to use the same value for all channels (``False``) or to
            sample a new value for each channel (``True``). If this value is a
            float ``p``, then for ``p`` percent of all images `per_channel` will
            be treated as ``True``, otherwise as ``False``.

        seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        name : None or str, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            Old name for parameter `seed`.
            Its usage will not yet cause a deprecation warning,
            but it is still recommended to use `seed` now.
            Outdated since 0.4.0.

        deterministic : bool, optional
            Deprecated since 0.4.0.
            See method ``to_deterministic()`` for an alternative and for
            details about what the "deterministic mode" actually does.
        """
        transform = iaa.LinearContrast(alpha, per_channel, seed, name, random_state, deterministic)
        augmented_image = transform(image=self.image)
        self.image = augmented_image

        return self

    def AllChannelsCLAHE(self, clip_limit=(0.1, 8), tile_grid_size_px=(3, 12), tile_grid_size_px_min=3,
                         per_channel=False, seed=None, name=None, random_state='deprecated',
                         deterministic='deprecated'):
        """
        Apply CLAHE to all channels of images in their original colorspaces.

        CLAHE (Contrast Limited Adaptive Histogram Equalization) performs
        histogram equilization within image patches, i.e. over local
        neighbourhoods.

        In contrast to ``imgaug.augmenters.contrast.CLAHE``, this augmenter
        operates directly on all channels of the input images. It does not
        perform any colorspace transformations and does not focus on specific
        channels (e.g. ``L`` in ``Lab`` colorspace).

        **Supported dtypes**:

            * ``uint8``: yes; fully tested
            * ``uint16``: yes; tested
            * ``uint32``: no (1)
            * ``uint64``: no (2)
            * ``int8``: no (2)
            * ``int16``: no (2)
            * ``int32``: no (2)
            * ``int64``: no (2)
            * ``float16``: no (2)
            * ``float32``: no (2)
            * ``float64``: no (2)
            * ``float128``: no (1)
            * ``bool``: no (1)

            - (1) rejected by cv2
            - (2) results in error in cv2: ``cv2.error:
                OpenCV(3.4.2) (...)/clahe.cpp:351: error: (-215:Assertion
                failed) src.type() == (((0) & ((1 << 3) - 1)) + (((1)-1) << 3))
                || _src.type() == (((2) & ((1 << 3) - 1)) + (((1)-1) << 3)) in
                function 'apply'``

        Parameters
        ----------
        clip_limit : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
            See ``imgaug.augmenters.contrast.CLAHE``.

        tile_grid_size_px : int or tuple of int or list of int or imgaug.parameters.StochasticParameter or tuple of tuple of int or tuple of list of int or tuple of imgaug.parameters.StochasticParameter, optional
            See ``imgaug.augmenters.contrast.CLAHE``.

        tile_grid_size_px_min : int, optional
            See ``imgaug.augmenters.contrast.CLAHE``.

        per_channel : bool or float, optional
            Whether to use the same value for all channels (``False``) or to
            sample a new value for each channel (``True``). If this value is a
            float ``p``, then for ``p`` percent of all images `per_channel` will
            be treated as ``True``, otherwise as ``False``.

        seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        name : None or str, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            Old name for parameter `seed`.
            Its usage will not yet cause a deprecation warning,
            but it is still recommended to use `seed` now.
            Outdated since 0.4.0.

        deterministic : bool, optional
            Deprecated since 0.4.0.
            See method ``to_deterministic()`` for an alternative and for
            details about what the "deterministic mode" actually does.
        """
        transform = iaa.AllChannelsCLAHE(clip_limit, tile_grid_size_px, tile_grid_size_px_min, per_channel, seed, name,
                                         random_state, deterministic)
        augmented_image = transform(image=self.image)
        self.image = augmented_image

        return self

    def AllChannelsHistogramEqualization(self, seed=None, name=None, random_state='deprecated',
                                         deterministic='deprecated'):
        """
        Apply Histogram Eq. to all channels of images in their original colorspaces.

        In contrast to ``imgaug.augmenters.contrast.HistogramEqualization``, this
        augmenter operates directly on all channels of the input images. It does
        not perform any colorspace transformations and does not focus on specific
        channels (e.g. ``L`` in ``Lab`` colorspace).

        **Supported dtypes**:

            * ``uint8``: yes; fully tested
            * ``uint16``: no (1)
            * ``uint32``: no (2)
            * ``uint64``: no (1)
            * ``int8``: no (1)
            * ``int16``: no (1)
            * ``int32``: no (1)
            * ``int64``: no (1)
            * ``float16``: no (2)
            * ``float32``: no (1)
            * ``float64``: no (1)
            * ``float128``: no (2)
            * ``bool``: no (1)

            - (1) causes cv2 error: ``cv2.error:
                OpenCV(3.4.5) (...)/histogram.cpp:3345: error: (-215:Assertion
                failed) src.type() == CV_8UC1 in function 'equalizeHist'``
            - (2) rejected by cv2

        Parameters
        ----------
        seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        name : None or str, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            Old name for parameter `seed`.
            Its usage will not yet cause a deprecation warning,
            but it is still recommended to use `seed` now.
            Outdated since 0.4.0.

        deterministic : bool, optional
            Deprecated since 0.4.0.
            See method ``to_deterministic()`` for an alternative and for
            details about what the "deterministic mode" actually does.
        """
        transform = iaa.AllChannelsHistogramEqualization(seed, name, random_state, deterministic)
        augmented_image = transform(image=self.image)
        self.image = augmented_image

        return self

    def HistogramEqualization(self, from_colorspace='RGB', to_colorspace='Lab', seed=None, name=None,
                              random_state='deprecated', deterministic='deprecated'):
        """
        Apply Histogram Eq. to L/V/L channels of images in HLS/HSV/Lab colorspaces.

        This augmenter is similar to ``imgaug.augmenters.contrast.CLAHE``.

        The augmenter transforms input images to a target colorspace (e.g.
        ``Lab``), extracts an intensity-related channel from the converted images
        (e.g. ``L`` for ``Lab``), applies Histogram Equalization to the channel
        and then converts the resulting image back to the original colorspace.

        Grayscale images (images without channel axis or with only one channel
        axis) are automatically handled, `from_colorspace` does not have to be
        adjusted for them. For images with four channels (e.g. RGBA), the fourth
        channel is ignored in the colorspace conversion (e.g. from an ``RGBA``
        image, only the ``RGB`` part is converted, normalized, converted back and
        concatenated with the input ``A`` channel). Images with unusual channel
        numbers (2, 5 or more than 5) are normalized channel-by-channel (same
        behaviour as ``AllChannelsHistogramEqualization``, though a warning will
        be raised).

        If you want to apply HistogramEqualization to each channel of the original
        input image's colorspace (without any colorspace conversion), use
        ``imgaug.augmenters.contrast.AllChannelsHistogramEqualization`` instead.

        **Supported dtypes**:

            * ``uint8``: yes; fully tested
            * ``uint16``: no (1)
            * ``uint32``: no (1)
            * ``uint64``: no (1)
            * ``int8``: no (1)
            * ``int16``: no (1)
            * ``int32``: no (1)
            * ``int64``: no (1)
            * ``float16``: no (1)
            * ``float32``: no (1)
            * ``float64``: no (1)
            * ``float128``: no (1)
            * ``bool``: no (1)

            - (1) This augmenter uses :class:`AllChannelsHistogramEqualization`,
                which only supports ``uint8``.

        Parameters
        ----------
        from_colorspace : {"RGB", "BGR", "HSV", "HLS", "Lab"}, optional
            Colorspace of the input images.
            If any input image has only one or zero channels, this setting will be
            ignored and it will be assumed that the input is grayscale.
            If a fourth channel is present in an input image, it will be removed
            before the colorspace conversion and later re-added.
            See also :func:`~imgaug.augmenters.color.change_colorspace_` for
            details.

        to_colorspace : {"Lab", "HLS", "HSV"}, optional
            Colorspace in which to perform Histogram Equalization. For ``Lab``,
            the equalization will only be applied to the first channel (``L``),
            for ``HLS`` to the second (``L``) and for ``HSV`` to the third (``V``).
            To apply histogram equalization to all channels of an input image
            (without colorspace conversion), see
            ``imgaug.augmenters.contrast.AllChannelsHistogramEqualization``.

        seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        name : None or str, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            Old name for parameter `seed`.
            Its usage will not yet cause a deprecation warning,
            but it is still recommended to use `seed` now.
            Outdated since 0.4.0.

        deterministic : bool, optional
            Deprecated since 0.4.0.
            See method ``to_deterministic()`` for an alternative and for
            details about what the "deterministic mode" actually does.
        """

        transform = iaa.HistogramEqualization(from_colorspace, to_colorspace, seed, name, random_state, deterministic)
        augmented_image = transform(image=self.image)
        self.image = augmented_image

        return self

    def Convolve(self, matrix=None, seed=None, name=None, random_state='deprecated', deterministic='deprecated'):
        """
        Apply a convolution to input images.

        **Supported dtypes**:

            * ``uint8``: yes; fully tested
            * ``uint16``: yes; tested
            * ``uint32``: no (1)
            * ``uint64``: no (2)
            * ``int8``: yes; tested (3)
            * ``int16``: yes; tested
            * ``int32``: no (2)
            * ``int64``: no (2)
            * ``float16``: yes; tested (4)
            * ``float32``: yes; tested
            * ``float64``: yes; tested
            * ``float128``: no (1)
            * ``bool``: yes; tested (4)

            - (1) rejected by ``cv2.filter2D()``.
            - (2) causes error: cv2.error: OpenCV(3.4.2) (...)/filter.cpp:4487:
                error: (-213:The function/feature is not implemented)
                Unsupported combination of source format (=1), and destination
                format (=1) in function 'getLinearFilter'.
            - (3) mapped internally to ``int16``.
            - (4) mapped internally to ``float32``.

        Parameters
        ----------
        matrix : None or (H, W) ndarray or imgaug.parameters.StochasticParameter or callable, optional
            The weight matrix of the convolution kernel to apply.

                * If ``None``, the input images will not be changed.
                * If a 2D numpy array, that array will always be used for all
                images and channels as the kernel.
                * If a callable, that method will be called for each image
                via ``parameter(image, C, random_state)``. The function must
                either return a list of ``C`` matrices (i.e. one per channel)
                or a 2D numpy array (will be used for all channels) or a
                3D ``HxWxC`` numpy array. If a list is returned, each entry may
                be ``None``, which will result in no changes to the respective
                channel.

        seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        name : None or str, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            Old name for parameter `seed`.
            Its usage will not yet cause a deprecation warning,
            but it is still recommended to use `seed` now.
            Outdated since 0.4.0.

        deterministic : bool, optional
            Deprecated since 0.4.0.
            See method ``to_deterministic()`` for an alternative and for
            details about what the "deterministic mode" actually does.
        """
        transform = iaa.Convolve(matrix, seed, name, random_state, deterministic)
        augmented_image = transform(image=self.image)
        self.image = augmented_image

        return self

    def EdgeDetect(self, alpha=(0.0, 0.75), seed=None, name=None, random_state='deprecated',
                   deterministic='deprecated'):
        """
        Generate a black & white edge image and alpha-blend it with the input image.

        **Supported dtypes**:

        See :class:`~imgaug.augmenters.convolutional.Convolve`.

        Parameters
        ----------
        alpha : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
            Blending factor of the edge image. At ``0.0``, only the original
            image is visible, at ``1.0`` only the edge image is visible.

                * If a number, exactly that value will always be used.
                * If a tuple ``(a, b)``, a random value will be sampled from the
                interval ``[a, b]`` per image.
                * If a list, a random value will be sampled from that list
                per image.
                * If a ``StochasticParameter``, a value will be sampled from that
                parameter per image.

        seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        name : None or str, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            Old name for parameter `seed`.
            Its usage will not yet cause a deprecation warning,
            but it is still recommended to use `seed` now.
            Outdated since 0.4.0.

        deterministic : bool, optional
            Deprecated since 0.4.0.
            See method ``to_deterministic()`` for an alternative and for
            details about what the "deterministic mode" actually does.
        """

        transform = iaa.EdgeDetect(alpha, seed, name, random_state, deterministic)
        augmented_image = transform(image=self.image)
        self.image = augmented_image

        return self

    def DirectedEdgeDetect(self, alpha=(0.0, 0.75), direction=(0.0, 1.0), seed=None, name=None,
                           random_state='deprecated', deterministic='deprecated'):
        """
        Detect edges from specified angles and alpha-blend with the input image.

        This augmenter first detects edges along a certain angle.
        Usually, edges are detected in x- or y-direction, while here the edge
        detection kernel is rotated to match a specified angle.
        The result of applying the kernel is a black (non-edges) and white (edges)
        image. That image is alpha-blended with the input image.

        **Supported dtypes**:

        See :class:`~imgaug.augmenters.convolutional.Convolve`.

        Parameters
        ----------
        alpha : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
            Blending factor of the edge image. At ``0.0``, only the original
            image is visible, at ``1.0`` only the edge image is visible.

                * If a number, exactly that value will always be used.
                * If a tuple ``(a, b)``, a random value will be sampled from the
                interval ``[a, b]`` per image.
                * If a list, a random value will be sampled from that list
                per image.
                * If a ``StochasticParameter``, a value will be sampled from that
                parameter per image.

        direction : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
            Angle (in degrees) of edges to pronounce, where ``0`` represents
            ``0`` degrees and ``1.0`` represents 360 degrees (both clockwise,
            starting at the top). Default value is ``(0.0, 1.0)``, i.e. pick a
            random angle per image.

                * If a number, exactly that value will always be used.
                * If a tuple ``(a, b)``, a random value will be sampled from the
                interval ``[a, b]`` will be sampled per image.
                * If a list, then a random value will be sampled from that list
                per image.
                * If a ``StochasticParameter``, a value will be sampled from the
                parameter per image.

        seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        name : None or str, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            Old name for parameter `seed`.
            Its usage will not yet cause a deprecation warning,
            but it is still recommended to use `seed` now.
            Outdated since 0.4.0.

        deterministic : bool, optional
            Deprecated since 0.4.0.
            See method ``to_deterministic()`` for an alternative and for
            details about what the "deterministic mode" actually does.
        """

        transform = iaa.DirectedEdgeDetect(alpha, direction, seed, name, random_state, deterministic)
        augmented_image = transform(image=self.image)
        self.image = augmented_image

        return self

    def Canny(self, alpha=(0.0, 1.0), hysteresis_thresholds=((60, 140), (160, 240)), sobel_kernel_size=(3, 7),
              colorizer=None, seed=None, name=None, random_state='deprecated', deterministic='deprecated'):
        """
        Apply a canny edge detector to input images.

        **Supported dtypes**:

            * ``uint8``: yes; fully tested
            * ``uint16``: no; not tested
            * ``uint32``: no; not tested
            * ``uint64``: no; not tested
            * ``int8``: no; not tested
            * ``int16``: no; not tested
            * ``int32``: no; not tested
            * ``int64``: no; not tested
            * ``float16``: no; not tested
            * ``float32``: no; not tested
            * ``float64``: no; not tested
            * ``float128``: no; not tested
            * ``bool``: no; not tested

        Parameters
        ----------
        alpha : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
            Blending factor to use in alpha blending.
            A value close to 1.0 means that only the edge image is visible.
            A value close to 0.0 means that only the original image is visible.
            A value close to 0.5 means that the images are merged according to
            `0.5*image + 0.5*edge_image`.
            If a sample from this parameter is 0, no action will be performed for
            the corresponding image.

                * If an int or float, exactly that value will be used.
                * If a tuple ``(a, b)``, a random value from the range
                ``a <= x <= b`` will be sampled per image.
                * If a list, then a random value will be sampled from that list
                per image.
                * If a StochasticParameter, a value will be sampled from the
                parameter per image.

        hysteresis_thresholds : number or tuple of number or list of number or imgaug.parameters.StochasticParameter or tuple of tuple of number or tuple of list of number or tuple of imgaug.parameters.StochasticParameter, optional
            Min and max values to use in hysteresis thresholding.
            (This parameter seems to have not very much effect on the results.)
            Either a single parameter or a tuple of two parameters.
            If a single parameter is provided, the sampling happens once for all
            images with `(N,2)` samples being requested from the parameter,
            where each first value denotes the hysteresis minimum and each second
            the maximum.
            If a tuple of two parameters is provided, one sampling of `(N,)` values
            is independently performed per parameter (first parameter: hysteresis
            minimum, second: hysteresis maximum).

                * If this is a single number, both min and max value will always be
                exactly that value.
                * If this is a tuple of numbers ``(a, b)``, two random values from
                the range ``a <= x <= b`` will be sampled per image.
                * If this is a list, two random values will be sampled from that
                list per image.
                * If this is a StochasticParameter, two random values will be
                sampled from that parameter per image.
                * If this is a tuple ``(min, max)`` with ``min`` and ``max``
                both *not* being numbers, they will be treated according to the
                rules above (i.e. may be a number, tuple, list or
                StochasticParameter). A single value will be sampled per image
                and parameter.

        sobel_kernel_size : int or tuple of int or list of int or imgaug.parameters.StochasticParameter, optional
            Kernel size of the sobel operator initially applied to each image.
            This corresponds to ``apertureSize`` in ``cv2.Canny()``.
            If a sample from this parameter is ``<=1``, no action will be performed
            for the corresponding image.
            The maximum for this parameter is ``7`` (inclusive). Higher values are
            not accepted by OpenCV.
            If an even value ``v`` is sampled, it is automatically changed to
            ``v-1``.

                * If this is a single integer, the kernel size always matches that
                value.
                * If this is a tuple of integers ``(a, b)``, a random discrete
                value will be sampled from the range ``a <= x <= b`` per image.
                * If this is a list, a random value will be sampled from that
                list per image.
                * If this is a StochasticParameter, a random value will be sampled
                from that parameter per image.

        colorizer : None or imgaug.augmenters.edges.IBinaryImageColorizer, optional
            A strategy to convert binary edge images to color images.
            If this is ``None``, an instance of ``RandomColorBinaryImageColorizer``
            is created, which means that each edge image is converted into an
            ``uint8`` image, where edge and non-edge pixels each have a different
            color that was uniformly randomly sampled from the space of all
            ``uint8`` colors.

        seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        name : None or str, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            Old name for parameter `seed`.
            Its usage will not yet cause a deprecation warning,
            but it is still recommended to use `seed` now.
            Outdated since 0.4.0.

        deterministic : bool, optional
            Deprecated since 0.4.0.
            See method ``to_deterministic()`` for an alternative and for
            details about what the "deterministic mode" actually does.
        """
        transform = iaa.Canny(alpha, hysteresis_thresholds, sobel_kernel_size, colorizer, seed, name, random_state,
                              deterministic)
        augmented_image = transform(image=self.image)
        self.image = augmented_image

        return self

    def Fliplr(self, p=1, seed=None, name=None, random_state='deprecated', deterministic='deprecated'):
        """Flip/mirror input images horizontally.

        .. note::

            The default value for the probability is ``0.0``.
            So, to flip *all* input images use ``Fliplr(1.0)`` and *not* just
            ``Fliplr()``.

        **Supported dtypes**:

        See :func:`~imgaug.augmenters.flip.fliplr`.

        Parameters
        ----------
        p : number or imgaug.parameters.StochasticParameter, optional
            Probability of each image to get flipped.

        seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        name : None or str, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            Old name for parameter `seed`.
            Its usage will not yet cause a deprecation warning,
            but it is still recommended to use `seed` now.
            Outdated since 0.4.0.

        deterministic : bool, optional
            Deprecated since 0.4.0.
            See method ``to_deterministic()`` for an alternative and for
            details about what the "deterministic mode" actually does.
        """
        transform = iaa.Fliplr(p, seed, name, random_state, deterministic)
        augmented_image, augmented_bbox = transform(image=self.image, bounding_boxes=self.bbox)
        self.image = augmented_image
        self.bbox = augmented_bbox

        return self

    def Flipud(self, p=1, seed=None, name=None, random_state='deprecated', deterministic='deprecated'):
        """Flip/mirror input images vertically.

        .. note::

            The default value for the probability is ``0.0``.
            So, to flip *all* input images use ``Flipud(1.0)`` and *not* just
            ``Flipud()``.

        **Supported dtypes**:

        See :func:`~imgaug.augmenters.flip.flipud`.

        Parameters
        ----------
        p : number or imgaug.parameters.StochasticParameter, optional
            Probability of each image to get flipped.

        seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        name : None or str, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            Old name for parameter `seed`.
            Its usage will not yet cause a deprecation warning,
            but it is still recommended to use `seed` now.
            Outdated since 0.4.0.

        deterministic : bool, optional
            Deprecated since 0.4.0.
            See method ``to_deterministic()`` for an alternative and for
            details about what the "deterministic mode" actually does.
        """
        transform = iaa.Flipud(p, seed, name, random_state, deterministic)
        augmented_image, augmented_bbox = transform(image=self.image, bounding_boxes=self.bbox)
        self.image = augmented_image
        self.bbox = augmented_bbox

        return self

    def Jigsaw(self, nb_rows=(3, 10), nb_cols=(3, 10), max_steps=1, allow_pad=True, seed=None, name=None,
               random_state='deprecated', deterministic='deprecated'):
        """
        Move cells within images similar to jigsaw patterns.

        .. note::

            This augmenter will by default pad images until their height is a
            multiple of `nb_rows`. Analogous for `nb_cols`.

        .. note::

            This augmenter will resize heatmaps and segmentation maps to the
            image size, then apply similar padding as for the corresponding images
            and resize back to the original map size. That also means that images
            may change in shape (due to padding), but heatmaps/segmaps will not
            change. For heatmaps/segmaps, this deviates from pad augmenters that
            will change images and heatmaps/segmaps in corresponding ways and then
            keep the heatmaps/segmaps at the new size.

        .. warning::

            This augmenter currently only supports augmentation of images,
            heatmaps, segmentation maps and keypoints. Other augmentables,
            i.e. bounding boxes, polygons and line strings, will result in errors.

        Added in 0.4.0.

        **Supported dtypes**:

        See :func:`~imgaug.augmenters.geometric.apply_jigsaw`.

        Parameters
        ----------
        nb_rows : int or list of int or tuple of int or imgaug.parameters.StochasticParameter, optional
            How many rows the jigsaw pattern should have.

                * If a single ``int``, then that value will be used for all images.
                * If a tuple ``(a, b)``, then a random value will be uniformly
                sampled per image from the discrete interval ``[a..b]``.
                * If a list, then for each image a random value will be sampled
                from that list.
                * If ``StochasticParameter``, then that parameter is queried per
                image to sample the value to use.

        nb_cols : int or list of int or tuple of int or imgaug.parameters.StochasticParameter, optional
            How many cols the jigsaw pattern should have.

                * If a single ``int``, then that value will be used for all images.
                * If a tuple ``(a, b)``, then a random value will be uniformly
                sampled per image from the discrete interval ``[a..b]``.
                * If a list, then for each image a random value will be sampled
                from that list.
                * If ``StochasticParameter``, then that parameter is queried per
                image to sample the value to use.

        max_steps : int or list of int or tuple of int or imgaug.parameters.StochasticParameter, optional
            How many steps each jigsaw cell may be moved.

                * If a single ``int``, then that value will be used for all images.
                * If a tuple ``(a, b)``, then a random value will be uniformly
                sampled per image from the discrete interval ``[a..b]``.
                * If a list, then for each image a random value will be sampled
                from that list.
                * If ``StochasticParameter``, then that parameter is queried per
                image to sample the value to use.

        allow_pad : bool, optional
            Whether to allow automatically padding images until they are evenly
            divisible by ``nb_rows`` and ``nb_cols``.

        seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        name : None or str, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            Old name for parameter `seed`.
            Its usage will not yet cause a deprecation warning,
            but it is still recommended to use `seed` now.
            Outdated since 0.4.0.

        deterministic : bool, optional
            Deprecated since 0.4.0.
            See method ``to_deterministic()`` for an alternative and for
            details about what the "deterministic mode" actually does.
        """
        transform = iaa.jigsaw(nb_rows, nb_cols, max_steps, allow_pad, seed, name, random_state, deterministic)
        augmented_image = transform(image=self.image)
        self.image = augmented_image

        return self

    def ShotNoise(self, severity=(1, 5), seed=None, name=None, random_state='deprecated', deterministic='deprecated'):
        """
        Wrapper around ``imagecorruptions.shot_noise``.

        .. note::

            This augmenter only affects images. Other data is not changed.

        Added in 0.4.0.

        **Supported dtypes**:

        See :func:`~imgaug.augmenters.imgcorruptlike.apply_shot_noise`.

        Parameters
        ----------
        severity : int, optional
            Strength of the corruption, with valid values being
            ``1 <= severity <= 5``.

        seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        name : None or str, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            Old name for parameter `seed`.
            Its usage will not yet cause a deprecation warning,
            but it is still recommended to use `seed` now.
            Outdated since 0.4.0.

        deterministic : bool, optional
            Deprecated since 0.4.0.
            See method ``to_deterministic()`` for an alternative and for
            details about what the "deterministic mode" actually does.
        """
        transform = iaa.imgcorruptlike.ShotNoise(severity, seed, name, random_state, deterministic)
        augmented_image = transform(image=self.image)
        self.image = augmented_image

        return self

    def SpeckleNoise(self, severity=(1, 5), seed=None, name=None, random_state='deprecated',
                     deterministic='deprecated'):
        """
        Wrapper around ``imagecorruptions.corruptions.speckle_noise``.

        .. note::

            This augmenter only affects images. Other data is not changed.

        Added in 0.4.0.

        **Supported dtypes**:

        See :func:`~imgaug.augmenters.imgcorruptlike.apply_speckle_noise`.

        Parameters
        ----------
        severity : int, optional
            Strength of the corruption, with valid values being
            ``1 <= severity <= 5``.

        seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        name : None or str, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            Old name for parameter `seed`.
            Its usage will not yet cause a deprecation warning,
            but it is still recommended to use `seed` now.
            Outdated since 0.4.0.

        deterministic : bool, optional
            Deprecated since 0.4.0.
            See method ``to_deterministic()`` for an alternative and for
            details about what the "deterministic mode" actually does.
        """
        transform = iaa.imgcorruptlike.SpeckleNoise(severity, seed, name, random_state, deterministic)
        augmented_image = transform(image=self.image)
        self.image = augmented_image

        return self

    def DefocusBlur(self, severity=(1, 5), seed=None, name=None, random_state='deprecated', deterministic='deprecated'):
        """
        Wrapper around ``imagecorruptions.corruptions.defocus_blur``.

        .. note::

            This augmenter only affects images. Other data is not changed.

        Added in 0.4.0.

        **Supported dtypes**:

        See :func:`~imgaug.augmenters.imgcorruptlike.apply_defocus_blur`.

        Parameters
        ----------
        severity : int, optional
            Strength of the corruption, with valid values being
            ``1 <= severity <= 5``.

        seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        name : None or str, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            Old name for parameter `seed`.
            Its usage will not yet cause a deprecation warning,
            but it is still recommended to use `seed` now.
            Outdated since 0.4.0.

        deterministic : bool, optional
            Deprecated since 0.4.0.
            See method ``to_deterministic()`` for an alternative and for
            details about what the "deterministic mode" actually does.
        """
        transform = iaa.imgcorruptlike.DefocusBlur(severity, seed, name, random_state, deterministic)
        augmented_image = transform(image=self.image)
        self.image = augmented_image

        return self

    def ZoomBlur(self, severity=(1, 5), seed=None, name=None, random_state='deprecated', deterministic='deprecated'):
        """
        Wrapper around ``imagecorruptions.corruptions.zoom_blur``.

        .. note::

            This augmenter only affects images. Other data is not changed.

        Added in 0.4.0.

        **Supported dtypes**:

        See :func:`~imgaug.augmenters.imgcorruptlike.apply_zoom_blur`.

        Parameters
        ----------
        severity : int, optional
            Strength of the corruption, with valid values being
            ``1 <= severity <= 5``.

        seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        name : None or str, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            Old name for parameter `seed`.
            Its usage will not yet cause a deprecation warning,
            but it is still recommended to use `seed` now.
            Outdated since 0.4.0.

        deterministic : bool, optional
            Deprecated since 0.4.0.
            See method ``to_deterministic()`` for an alternative and for
            details about what the "deterministic mode" actually does.
        """
        transform = iaa.imgcorruptlike.ZoomBlur(severity, seed, name, random_state, deterministic)
        augmented_image = transform(image=self.image)
        self.image = augmented_image

        return self

    def Frost(self, severity=(1, 5), seed=None, name=None, random_state='deprecated', deterministic='deprecated'):
        """
        Wrapper around ``imagecorruptions.corruptions.frost``.

        .. note::

            This augmenter only affects images. Other data is not changed.

        Added in 0.4.0.

        **Supported dtypes**:

        See :func:`~imgaug.augmenters.imgcorruptlike.apply_frost`.

        Parameters
        ----------
        severity : int, optional
            Strength of the corruption, with valid values being
            ``1 <= severity <= 5``.

        seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        name : None or str, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            Old name for parameter `seed`.
            Its usage will not yet cause a deprecation warning,
            but it is still recommended to use `seed` now.
            Outdated since 0.4.0.

        deterministic : bool, optional
            Deprecated since 0.4.0.
            See method ``to_deterministic()`` for an alternative and for
            details about what the "deterministic mode" actually does.
        """
        transform = iaa.imgcorruptlike.Frost(severity, seed, name, random_state, deterministic)
        augmented_image = transform(image=self.image)
        self.image = augmented_image

        return self

    def Spatter(self, severity=(1, 5), seed=None, name=None, random_state='deprecated', deterministic='deprecated'):
        """
        Wrapper around ``imagecorruptions.corruptions.spatter``.

        .. note::

            This augmenter only affects images. Other data is not changed.

        Added in 0.4.0.

        **Supported dtypes**:

        See :func:`~imgaug.augmenters.imgcorruptlike.apply_spatter`.

        Parameters
        ----------
        severity : int, optional
            Strength of the corruption, with valid values being
            ``1 <= severity <= 5``.

        seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        name : None or str, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            Old name for parameter `seed`.
            Its usage will not yet cause a deprecation warning,
            but it is still recommended to use `seed` now.
            Outdated since 0.4.0.

        deterministic : bool, optional
            Deprecated since 0.4.0.
            See method ``to_deterministic()`` for an alternative and for
            details about what the "deterministic mode" actually does.
        """
        transform = iaa.imgcorruptlike.Spatter(severity, seed, name, random_state, deterministic)
        augmented_image = transform(image=self.image)
        self.image = augmented_image

        return self

    def Saturate(self, severity=(1, 5), seed=None, name=None, random_state='deprecated', deterministic='deprecated'):
        """
        Wrapper around ``imagecorruptions.corruptions.saturate``.

        .. note::

            This augmenter only affects images. Other data is not changed.

        Added in 0.4.0.

        **Supported dtypes**:

        See :func:`~imgaug.augmenters.imgcorruptlike.apply_saturate`.

        Parameters
        ----------
        severity : int, optional
            Strength of the corruption, with valid values being
            ``1 <= severity <= 5``.

        seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        name : None or str, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            Old name for parameter `seed`.
            Its usage will not yet cause a deprecation warning,
            but it is still recommended to use `seed` now.
            Outdated since 0.4.0.

        deterministic : bool, optional
            Deprecated since 0.4.0.
            See method ``to_deterministic()`` for an alternative and for
            details about what the "deterministic mode" actually does.
        """
        transform = iaa.imgcorruptlike.Saturate(severity, seed, name, random_state, deterministic)
        augmented_image = transform(image=self.image)
        self.image = augmented_image

        return self

    def Pixelate(self, severity=(1, 5), seed=None, name=None, random_state='deprecated', deterministic='deprecated'):
        """
        Wrapper around ``imagecorruptions.corruptions.pixelate``.

        .. note::

            This augmenter only affects images. Other data is not changed.

        Added in 0.4.0.

        **Supported dtypes**:

        See :func:`~imgaug.augmenters.imgcorruptlike.apply_pixelate`.

        Parameters
        ----------
        severity : int, optional
            Strength of the corruption, with valid values being
            ``1 <= severity <= 5``.

        seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        name : None or str, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            Old name for parameter `seed`.
            Its usage will not yet cause a deprecation warning,
            but it is still recommended to use `seed` now.
            Outdated since 0.4.0.

        deterministic : bool, optional
            Deprecated since 0.4.0.
            See method ``to_deterministic()`` for an alternative and for
            details about what the "deterministic mode" actually does.
        """
        transform = iaa.imgcorruptlike.Pixelate(severity, seed, name, random_state, deterministic)
        augmented_image = transform(image=self.image)
        self.image = augmented_image

        return self

    def Contrast(self, severity=(1, 5), seed=None, name=None, random_state='deprecated', deterministic='deprecated'):
        """
           Wrapper around ``imagecorruptions.corruptions.contrast``.

           .. note::

               This augmenter only affects images. Other data is not changed.

           Added in 0.4.0.

           **Supported dtypes**:

           See :func:`~imgaug.augmenters.imgcorruptlike.apply_contrast`.

           Parameters
           ----------
           severity : int, optional
               Strength of the corruption, with valid values being
               ``1 <= severity <= 5``.

           seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
               See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

           name : None or str, optional
               See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

           random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
               Old name for parameter `seed`.
               Its usage will not yet cause a deprecation warning,
               but it is still recommended to use `seed` now.
               Outdated since 0.4.0.

           deterministic : bool, optional
               Deprecated since 0.4.0.
               See method ``to_deterministic()`` for an alternative and for
               details about what the "deterministic mode" actually does.
               """
        transform = iaa.imgcorruptlike.Contrast(severity, seed, name, random_state, deterministic)
        augmented_image = transform(image=self.image)
        self.image = augmented_image

        return self

    def EnhanceColor(self, factor=(0.0, 3.0), seed=None, name=None, random_state='deprecated',
                     deterministic='deprecated'):
        """
        Convert images to grayscale.

        This augmenter has identical outputs to ``PIL.ImageEnhance.Color``.

        Added in 0.4.0.

        **Supported dtypes**:

        See :func:`~imgaug.augmenters.pillike.enhance_color`.

        Parameters
        ----------
        factor : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
            Colorfulness of the output image. Values close to ``0.0`` lead
            to grayscale images, values above ``1.0`` increase the strength of
            colors. Sane values are roughly in ``[0.0, 3.0]``.

                * If ``number``: The value will be used for all images.
                * If ``tuple`` ``(a, b)``: A value will be uniformly sampled per
                image from the interval ``[a, b)``.
                * If ``list``: A random value will be picked from the list per
                image.
                * If ``StochasticParameter``: Per batch of size ``N``, the
                parameter will be queried once to return ``(N,)`` samples.

        seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        name : None or str, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            Old name for parameter `seed`.
            Its usage will not yet cause a deprecation warning,
            but it is still recommended to use `seed` now.
            Outdated since 0.4.0.

        deterministic : bool, optional
            Deprecated since 0.4.0.
            See method ``to_deterministic()`` for an alternative and for
            details about what the "deterministic mode" actually does.
        """

        transform = iaa.pillike.EnhanceColor(factor, seed, name, random_state, deterministic)
        augmented_image = transform(image=self.image)
        self.image = augmented_image

        return self

    def EnhanceContrast(self, factor=(0.5, 1.5), seed=None, name=None, random_state='deprecated',
                        deterministic='deprecated'):
        """
        Change the contrast of images.

        This augmenter has identical outputs to ``PIL.ImageEnhance.Contrast``.

        Added in 0.4.0.

        **Supported dtypes**:

        See :func:`~imgaug.augmenters.pillike.enhance_contrast`.

        Parameters
        ----------
        factor : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
            Strength of contrast in the image. Values below ``1.0`` decrease the
            contrast, leading to a gray image around ``0.0``. Values
            above ``1.0`` increase the contrast. Sane values are roughly in
            ``[0.5, 1.5]``.

                * If ``number``: The value will be used for all images.
                * If ``tuple`` ``(a, b)``: A value will be uniformly sampled per
                image from the interval ``[a, b)``.
                * If ``list``: A random value will be picked from the list per
                image.
                * If ``StochasticParameter``: Per batch of size ``N``, the
                parameter will be queried once to return ``(N,)`` samples.

        seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        name : None or str, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            Old name for parameter `seed`.
            Its usage will not yet cause a deprecation warning,
            but it is still recommended to use `seed` now.
            Outdated since 0.4.0.

        deterministic : bool, optional
            Deprecated since 0.4.0.
            See method ``to_deterministic()`` for an alternative and for
            details about what the "deterministic mode" actually does.
        """
        transform = iaa.pillike.EnhanceContrast(factor, seed, name, random_state, deterministic)
        augmented_image = transform(image=self.image)
        self.image = augmented_image

        return self

    def EnhanceBrightness(self, factor=(0.5, 1.5), seed=None, name=None, random_state='deprecated',
                          deterministic='deprecated'):
        """
        Change the brightness of images.

        This augmenter has identical outputs to
        ``PIL.ImageEnhance.Brightness``.

        Added in 0.4.0.

        **Supported dtypes**:

        See :func:`~imgaug.augmenters.pillike.enhance_brightness`.

        Parameters
        ----------
        factor : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
            Brightness of the image. Values below ``1.0`` decrease the brightness,
            leading to a black image around ``0.0``. Values above ``1.0`` increase
            the brightness. Sane values are roughly in ``[0.5, 1.5]``.

                * If ``number``: The value will be used for all images.
                * If ``tuple`` ``(a, b)``: A value will be uniformly sampled per
                image from the interval ``[a, b)``.
                * If ``list``: A random value will be picked from the list per
                image.
                * If ``StochasticParameter``: Per batch of size ``N``, the
                parameter will be queried once to return ``(N,)`` samples.

        seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        name : None or str, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            Old name for parameter `seed`.
            Its usage will not yet cause a deprecation warning,
            but it is still recommended to use `seed` now.
            Outdated since 0.4.0.

        deterministic : bool, optional
            Deprecated since 0.4.0.
            See method ``to_deterministic()`` for an alternative and for
            details about what the "deterministic mode" actually does.
        """
        transform = iaa.pillike.EnhanceBrightness(factor, seed, name, random_state, deterministic)
        augmented_image = transform(image=self.image)
        self.image = augmented_image

        return self

    def EnhanceSharpness(self, factor=(0.0, 2.0), seed=None, name=None, random_state='deprecated',
                         deterministic='deprecated'):
        """
        Change the sharpness of images.

        This augmenter has identical outputs to
        ``PIL.ImageEnhance.Sharpness``.

        Added in 0.4.0.

        **Supported dtypes**:

        See :func:`~imgaug.augmenters.pillike.enhance_sharpness`.

        Parameters
        ----------
        factor : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
            Sharpness of the image. Values below ``1.0`` decrease the sharpness,
            values above ``1.0`` increase it. Sane values are roughly in
            ``[0.0, 2.0]``.

                * If ``number``: The value will be used for all images.
                * If ``tuple`` ``(a, b)``: A value will be uniformly sampled per
                image from the interval ``[a, b)``.
                * If ``list``: A random value will be picked from the list per
                image.
                * If ``StochasticParameter``: Per batch of size ``N``, the
                parameter will be queried once to return ``(N,)`` samples.

        seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        name : None or str, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            Old name for parameter `seed`.
            Its usage will not yet cause a deprecation warning,
            but it is still recommended to use `seed` now.
            Outdated since 0.4.0.

        deterministic : bool, optional
            Deprecated since 0.4.0.
            See method ``to_deterministic()`` for an alternative and for
            details about what the "deterministic mode" actually does.
        """
        transform = iaa.pillike.EnhanceSharpness(factor, seed, name, random_state, deterministic)
        augmented_image = transform(image=self.image)
        self.image = augmented_image

        return self

    def FilterBlur(self, seed=None, name=None, random_state='deprecated', deterministic='deprecated'):
        """
        Apply a blur filter kernel to images.

        This augmenter has identical outputs to
        calling ``PIL.Image.filter`` with kernel ``PIL.ImageFilter.BLUR``.

        Added in 0.4.0.

        **Supported dtypes**:

        See :func:`~imgaug.augmenters.pillike.filter_blur`.

        Parameters
        ----------
        seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        name : None or str, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            Old name for parameter `seed`.
            Its usage will not yet cause a deprecation warning,
            but it is still recommended to use `seed` now.
            Outdated since 0.4.0.

        deterministic : bool, optional
            Deprecated since 0.4.0.
            See method ``to_deterministic()`` for an alternative and for
            details about what the "deterministic mode" actually does.
        """
        transform = iaa.pillike.FilterBlur(seed, name, random_state, deterministic)
        augmented_image = transform(image=self.image)
        self.image = augmented_image

        return self

    def FilterSmooth(self, seed=None, name=None, random_state='deprecated', deterministic='deprecated'):
        """
        Apply a smoothening filter kernel to images.

        This augmenter has identical outputs to
        calling ``PIL.Image.filter`` with kernel ``PIL.ImageFilter.SMOOTH``.

        Added in 0.4.0.

        **Supported dtypes**:

        See :func:`~imgaug.augmenters.pillike.filter_smooth`.

        Parameters
        ----------
        seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        name : None or str, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            Old name for parameter `seed`.
            Its usage will not yet cause a deprecation warning,
            but it is still recommended to use `seed` now.
            Outdated since 0.4.0.

        deterministic : bool, optional
            Deprecated since 0.4.0.
            See method ``to_deterministic()`` for an alternative and for
            details about what the "deterministic mode" actually does.
        """
        transform = iaa.pillike.FilterSmooth(seed, name, random_state, deterministic)
        augmented_image = transform(image=self.image)
        self.image = augmented_image

        return self

    def FilterSmoothMore(self, seed=None, name=None, random_state='deprecated', deterministic='deprecated'):
        """
        Apply a strong smoothening filter kernel to images.

        This augmenter has identical outputs to
        calling ``PIL.Image.filter`` with kernel ``PIL.ImageFilter.BLUR``.

        Added in 0.4.0.

        **Supported dtypes**:

        See :func:`~imgaug.augmenters.pillike.filter_smooth_more`.

        Parameters
        ----------
        seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        name : None or str, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            Old name for parameter `seed`.
            Its usage will not yet cause a deprecation warning,
            but it is still recommended to use `seed` now.
            Outdated since 0.4.0.

        deterministic : bool, optional
            Deprecated since 0.4.0.
            See method ``to_deterministic()`` for an alternative and for
            details about what the "deterministic mode" actually does.
        """
        transform = iaa.pillike.FilterSmoothMore(seed, name, random_state, deterministic)
        augmented_image = transform(image=self.image)
        self.image = augmented_image

        return self

    def FilterEdgeEnhance(self, seed=None, name=None, random_state='deprecated', deterministic='deprecated'):
        """
        Apply an edge enhance filter kernel to images.

        This augmenter has identical outputs to
        calling ``PIL.Image.filter`` with kernel
        ``PIL.ImageFilter.EDGE_ENHANCE``.

        Added in 0.4.0.

        **Supported dtypes**:

        See :func:`~imgaug.augmenters.pillike.filter_edge_enhance`.

        Parameters
        ----------
        seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        name : None or str, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            Old name for parameter `seed`.
            Its usage will not yet cause a deprecation warning,
            but it is still recommended to use `seed` now.
            Outdated since 0.4.0.

        deterministic : bool, optional
            Deprecated since 0.4.0.
            See method ``to_deterministic()`` for an alternative and for
            details about what the "deterministic mode" actually does.
        """
        transform = iaa.pillike.FilterEdgeEnhance(seed, name, random_state, deterministic)
        augmented_image = transform(image=self.image)
        self.image = augmented_image

        return self

    def FilterEdgeEnhanceMore(self, seed=None, name=None, random_state='deprecated', deterministic='deprecated'):
        """
        Apply a strong edge enhancement filter kernel to images.

        This augmenter has identical outputs to
        calling ``PIL.Image.filter`` with kernel
        ``PIL.ImageFilter.EDGE_ENHANCE_MORE``.

        Added in 0.4.0.

        **Supported dtypes**:

        See :func:`~imgaug.augmenters.pillike.filter_edge_enhance_more`.

        Parameters
        ----------
        seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        name : None or str, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            Old name for parameter `seed`.
            Its usage will not yet cause a deprecation warning,
            but it is still recommended to use `seed` now.
            Outdated since 0.4.0.

        deterministic : bool, optional
            Deprecated since 0.4.0.
            See method ``to_deterministic()`` for an alternative and for
            details about what the "deterministic mode" actually does.
        """
        transform = iaa.pillike.FilterEdgeEnhanceMore(seed, name, random_state, deterministic)
        augmented_image = transform(image=self.image)
        self.image = augmented_image

        return self

    def FilterFindEdges(self, seed=None, name=None, random_state='deprecated', deterministic='deprecated'):
        """
        Apply a edge detection kernel to images.

        This augmenter has identical outputs to
        calling ``PIL.Image.filter`` with kernel
        ``PIL.ImageFilter.FIND_EDGES``.

        Added in 0.4.0.

        **Supported dtypes**:

        See :func:`~imgaug.augmenters.pillike.filter_find_edges`.

        Parameters
        ----------
        seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        name : None or str, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            Old name for parameter `seed`.
            Its usage will not yet cause a deprecation warning,
            but it is still recommended to use `seed` now.
            Outdated since 0.4.0.

        deterministic : bool, optional
            Deprecated since 0.4.0.
            See method ``to_deterministic()`` for an alternative and for
            details about what the "deterministic mode" actually does.
        """
        transform = iaa.pillike.FilterFindEdges(seed, name, random_state, deterministic)
        augmented_image = transform(image=self.image)
        self.image = augmented_image

        return self

    def FilterContour(self, seed=None, name=None, random_state='deprecated', deterministic='deprecated'):
        """
        Apply a contour detection filter kernel to images.

        This augmenter has identical outputs to
        calling ``PIL.Image.filter`` with kernel ``PIL.ImageFilter.CONTOUR``.

        Added in 0.4.0.

        **Supported dtypes**:

        See :func:`~imgaug.augmenters.pillike.filter_contour`.

        Parameters
        ----------
        seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        name : None or str, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            Old name for parameter `seed`.
            Its usage will not yet cause a deprecation warning,
            but it is still recommended to use `seed` now.
            Outdated since 0.4.0.

        deterministic : bool, optional
            Deprecated since 0.4.0.
            See method ``to_deterministic()`` for an alternative and for
            details about what the "deterministic mode" actually does.
        """
        transform = iaa.pillike.FilterContour(seed, name, random_state, deterministic)
        augmented_image = transform(image=self.image)
        self.image = augmented_image

        return self

    def FilterEmboss(self, seed=None, name=None, random_state='deprecated', deterministic='deprecated'):
        """
        Apply an emboss filter kernel to images.

        This augmenter has identical outputs to
        calling ``PIL.Image.filter`` with kernel ``PIL.ImageFilter.EMBOSS``.

        Added in 0.4.0.

        **Supported dtypes**:

        See :func:`~imgaug.augmenters.pillike.filter_emboss`.

        Parameters
        ----------
        seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        name : None or str, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            Old name for parameter `seed`.
            Its usage will not yet cause a deprecation warning,
            but it is still recommended to use `seed` now.
            Outdated since 0.4.0.

        deterministic : bool, optional
            Deprecated since 0.4.0.
            See method ``to_deterministic()`` for an alternative and for
            details about what the "deterministic mode" actually does.
        """
        transform = iaa.pillike.FilterEmboss(seed, name, random_state, deterministic)
        augmented_image = transform(image=self.image)
        self.image = augmented_image

        return self

    def FilterSharpen(self, seed=None, name=None, random_state='deprecated', deterministic='deprecated'):
        """
        Apply a sharpening filter kernel to images.

        This augmenter has identical outputs to
        calling ``PIL.Image.filter`` with kernel ``PIL.ImageFilter.SHARPEN``.

        Added in 0.4.0.

        **Supported dtypes**:

        See :func:`~imgaug.augmenters.pillike.filter_sharpen`.

        Parameters
        ----------
        seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        name : None or str, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            Old name for parameter `seed`.
            Its usage will not yet cause a deprecation warning,
            but it is still recommended to use `seed` now.
            Outdated since 0.4.0.

        deterministic : bool, optional
            Deprecated since 0.4.0.
            See method ``to_deterministic()`` for an alternative and for
            details about what the "deterministic mode" actually does.
        """
        transform = iaa.pillike.FilterSharpen(seed, name, random_state, deterministic)
        augmented_image = transform(image=self.image)
        self.image = augmented_image

        return self

    def FilterDetail(self, seed=None, name=None, random_state='deprecated', deterministic='deprecated'):
        """
        Apply a detail enhancement filter kernel to images.

        This augmenter has identical outputs to
        calling ``PIL.Image.filter`` with kernel ``PIL.ImageFilter.DETAIL``.

        Added in 0.4.0.

        **Supported dtypes**:

        See :func:`~imgaug.augmenters.pillike.filter_detail`.

        Parameters
        ----------
        seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        name : None or str, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            Old name for parameter `seed`.
            Its usage will not yet cause a deprecation warning,
            but it is still recommended to use `seed` now.
            Outdated since 0.4.0.

        deterministic : bool, optional
            Deprecated since 0.4.0.
            See method ``to_deterministic()`` for an alternative and for
            details about what the "deterministic mode" actually does.
        """
        transform = iaa.pillike.FilterDetail(seed, name, random_state, deterministic)
        augmented_image = transform(image=self.image)
        self.image = augmented_image

        return self

    def AveragePooling(self, kernel_size=(1, 5), keep_size=True, seed=None, name=None, random_state='deprecated',
                       deterministic='deprecated'):
        """
        Apply average pooling to images.

        This augmenter pools images with kernel sizes ``H x W`` by averaging the
        pixel values within these windows. For e.g. ``2 x 2`` this halves the image
        size. Optionally, the augmenter will automatically re-upscale the image
        to the input size (by default this is activated).

        Note that this augmenter is very similar to ``AverageBlur``.
        ``AverageBlur`` applies averaging within windows of given kernel size
        *without* striding, while ``AveragePooling`` applies striding corresponding
        to the kernel size, with optional upscaling afterwards. The upscaling
        is configured to create "pixelated"/"blocky" images by default.

        .. note::

            During heatmap or segmentation map augmentation, the respective
            arrays are not changed, only the shapes of the underlying images
            are updated. This is because imgaug can handle maps/maks that are
            larger/smaller than their corresponding image.

        **Supported dtypes**:

        See :func:`~imgaug.imgaug.avg_pool`.

        Attributes
        ----------
        kernel_size : int or tuple of int or list of int or imgaug.parameters.StochasticParameter or tuple of tuple of int or tuple of list of int or tuple of imgaug.parameters.StochasticParameter, optional
            The kernel size of the pooling operation.

            * If an int, then that value will be used for all images for both
            kernel height and width.
            * If a tuple ``(a, b)``, then a value from the discrete range
            ``[a..b]`` will be sampled per image.
            * If a list, then a random value will be sampled from that list per
            image and used for both kernel height and width.
            * If a StochasticParameter, then a value will be sampled per image
            from that parameter per image and used for both kernel height and
            width.
            * If a tuple of tuple of int given as ``((a, b), (c, d))``, then two
            values will be sampled independently from the discrete ranges
            ``[a..b]`` and ``[c..d]`` per image and used as the kernel height
            and width.
            * If a tuple of lists of int, then two values will be sampled
            independently per image, one from the first list and one from the
            second, and used as the kernel height and width.
            * If a tuple of StochasticParameter, then two values will be sampled
            indepdently per image, one from the first parameter and one from the
            second, and used as the kernel height and width.

        keep_size : bool, optional
            After pooling, the result image will usually have a different
            height/width compared to the original input image. If this
            parameter is set to True, then the pooled image will be resized
            to the input image's size, i.e. the augmenter's output shape is always
            identical to the input shape.

        seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        name : None or str, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            Old name for parameter `seed`.
            Its usage will not yet cause a deprecation warning,
            but it is still recommended to use `seed` now.
            Outdated since 0.4.0.

        deterministic : bool, optional
            Deprecated since 0.4.0.
            See method ``to_deterministic()`` for an alternative and for
            details about what the "deterministic mode" actually does.
        """
        transform = iaa.AveragePooling(kernel_size, keep_size, seed, name, random_state, deterministic)
        augmented_image = transform(image=self.image)
        self.image = augmented_image

        return self

    def MaxPooling(self, kernel_size=(1, 5), keep_size=True, seed=None, name=None, random_state='deprecated',
                   deterministic='deprecated'):
        """
        Apply max pooling to images.

        This augmenter pools images with kernel sizes ``H x W`` by taking the
        maximum pixel value over windows. For e.g. ``2 x 2`` this halves the image
        size. Optionally, the augmenter will automatically re-upscale the image
        to the input size (by default this is activated).

        The maximum within each pixel window is always taken channelwise..

        .. note::

            During heatmap or segmentation map augmentation, the respective
            arrays are not changed, only the shapes of the underlying images
            are updated. This is because imgaug can handle maps/maks that are
            larger/smaller than their corresponding image.

        **Supported dtypes**:

        See :func:`~imgaug.imgaug.max_pool`.

        Attributes
        ----------
        kernel_size : int or tuple of int or list of int or imgaug.parameters.StochasticParameter or tuple of tuple of int or tuple of list of int or tuple of imgaug.parameters.StochasticParameter, optional
            The kernel size of the pooling operation.

            * If an int, then that value will be used for all images for both
            kernel height and width.
            * If a tuple ``(a, b)``, then a value from the discrete range
            ``[a..b]`` will be sampled per image.
            * If a list, then a random value will be sampled from that list per
            image and used for both kernel height and width.
            * If a StochasticParameter, then a value will be sampled per image
            from that parameter per image and used for both kernel height and
            width.
            * If a tuple of tuple of int given as ``((a, b), (c, d))``, then two
            values will be sampled independently from the discrete ranges
            ``[a..b]`` and ``[c..d]`` per image and used as the kernel height
            and width.
            * If a tuple of lists of int, then two values will be sampled
            independently per image, one from the first list and one from the
            second, and used as the kernel height and width.
            * If a tuple of StochasticParameter, then two values will be sampled
            indepdently per image, one from the first parameter and one from the
            second, and used as the kernel height and width.

        keep_size : bool, optional
            After pooling, the result image will usually have a different
            height/width compared to the original input image. If this
            parameter is set to True, then the pooled image will be resized
            to the input image's size, i.e. the augmenter's output shape is always
            identical to the input shape.

        seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        name : None or str, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            Old name for parameter `seed`.
            Its usage will not yet cause a deprecation warning,
            but it is still recommended to use `seed` now.
            Outdated since 0.4.0.

        deterministic : bool, optional
            Deprecated since 0.4.0.
            See method ``to_deterministic()`` for an alternative and for
            details about what the "deterministic mode" actually does.
        """
        transform = iaa.MaxPooling(kernel_size, keep_size, seed, name, random_state, deterministic)
        augmented_image = transform(image=self.image)
        self.image = augmented_image

        return self

    def MinPooling(self, kernel_size=(1, 5), keep_size=True, seed=None, name=None, random_state='deprecated',
                   deterministic='deprecated'):
        """
        Apply minimum pooling to images.

        This augmenter pools images with kernel sizes ``H x W`` by taking the
        minimum pixel value over windows. For e.g. ``2 x 2`` this halves the image
        size. Optionally, the augmenter will automatically re-upscale the image
        to the input size (by default this is activated).

        The minimum within each pixel window is always taken channelwise.

        .. note::

            During heatmap or segmentation map augmentation, the respective
            arrays are not changed, only the shapes of the underlying images
            are updated. This is because imgaug can handle maps/maks that are
            larger/smaller than their corresponding image.

        **Supported dtypes**:

        See :func:`~imgaug.imgaug.min_pool`.

        Attributes
        ----------
        kernel_size : int or tuple of int or list of int or imgaug.parameters.StochasticParameter or tuple of tuple of int or tuple of list of int or tuple of imgaug.parameters.StochasticParameter, optional
            The kernel size of the pooling operation.

            * If an int, then that value will be used for all images for both
            kernel height and width.
            * If a tuple ``(a, b)``, then a value from the discrete range
            ``[a..b]`` will be sampled per image.
            * If a list, then a random value will be sampled from that list per
            image and used for both kernel height and width.
            * If a StochasticParameter, then a value will be sampled per image
            from that parameter per image and used for both kernel height and
            width.
            * If a tuple of tuple of int given as ``((a, b), (c, d))``, then two
            values will be sampled independently from the discrete ranges
            ``[a..b]`` and ``[c..d]`` per image and used as the kernel height
            and width.
            * If a tuple of lists of int, then two values will be sampled
            independently per image, one from the first list and one from the
            second, and used as the kernel height and width.
            * If a tuple of StochasticParameter, then two values will be sampled
            indepdently per image, one from the first parameter and one from the
            second, and used as the kernel height and width.

        keep_size : bool, optional
            After pooling, the result image will usually have a different
            height/width compared to the original input image. If this
            parameter is set to True, then the pooled image will be resized
            to the input image's size, i.e. the augmenter's output shape is always
            identical to the input shape.

        seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        name : None or str, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            Old name for parameter `seed`.
            Its usage will not yet cause a deprecation warning,
            but it is still recommended to use `seed` now.
            Outdated since 0.4.0.

        deterministic : bool, optional
            Deprecated since 0.4.0.
            See method ``to_deterministic()`` for an alternative and for
            details about what the "deterministic mode" actually does.
        """
        transform = iaa.MinPooling(kernel_size, keep_size, seed, name, random_state, deterministic)
        augmented_image = transform(image=self.image)
        self.image = augmented_image

        return self

    def MedianPooling(self, kernel_size=(1, 5), keep_size=True, seed=None, name=None, random_state='deprecated',
                      deterministic='deprecated'):
        """
        Apply median pooling to images.

        This augmenter pools images with kernel sizes ``H x W`` by taking the
        median pixel value over windows. For e.g. ``2 x 2`` this halves the image
        size. Optionally, the augmenter will automatically re-upscale the image
        to the input size (by default this is activated).

        The median within each pixel window is always taken channelwise.

        .. note::

            During heatmap or segmentation map augmentation, the respective
            arrays are not changed, only the shapes of the underlying images
            are updated. This is because imgaug can handle maps/maks that are
            larger/smaller than their corresponding image.

        **Supported dtypes**:

        See :func:`~imgaug.imgaug.median_pool`.

        Attributes
        ----------
        kernel_size : int or tuple of int or list of int or imgaug.parameters.StochasticParameter or tuple of tuple of int or tuple of list of int or tuple of imgaug.parameters.StochasticParameter, optional
            The kernel size of the pooling operation.

            * If an int, then that value will be used for all images for both
            kernel height and width.
            * If a tuple ``(a, b)``, then a value from the discrete range
            ``[a..b]`` will be sampled per image.
            * If a list, then a random value will be sampled from that list per
            image and used for both kernel height and width.
            * If a StochasticParameter, then a value will be sampled per image
            from that parameter per image and used for both kernel height and
            width.
            * If a tuple of tuple of int given as ``((a, b), (c, d))``, then two
            values will be sampled independently from the discrete ranges
            ``[a..b]`` and ``[c..d]`` per image and used as the kernel height
            and width.
            * If a tuple of lists of int, then two values will be sampled
            independently per image, one from the first list and one from the
            second, and used as the kernel height and width.
            * If a tuple of StochasticParameter, then two values will be sampled
            indepdently per image, one from the first parameter and one from the
            second, and used as the kernel height and width.

        keep_size : bool, optional
            After pooling, the result image will usually have a different
            height/width compared to the original input image. If this
            parameter is set to True, then the pooled image will be resized
            to the input image's size, i.e. the augmenter's output shape is always
            identical to the input shape.

        seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        name : None or str, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            Old name for parameter `seed`.
            Its usage will not yet cause a deprecation warning,
            but it is still recommended to use `seed` now.
            Outdated since 0.4.0.

        deterministic : bool, optional
            Deprecated since 0.4.0.
            See method ``to_deterministic()`` for an alternative and for
            details about what the "deterministic mode" actually does.
        """
        transform = iaa.MedianPooling(kernel_size, keep_size, seed, name, random_state, deterministic)
        augmented_image = transform(image=self.image)
        self.image = augmented_image

        return self

    def PadToFixedSize(self, width=100, height=100, pad_mode='constant', pad_cval=0, position='uniform', seed=None,
                       name=None, random_state='deprecated', deterministic='deprecated'):
        """
        Pad images to a predefined minimum width and/or height.

        If images are already at the minimum width/height or are larger, they will
        not be padded. Note that this also means that images will not be cropped if
        they exceed the required width/height.

        The augmenter randomly decides per image how to distribute the required
        padding amounts over the image axis. E.g. if 2px have to be padded on the
        left or right to reach the required width, the augmenter will sometimes
        add 2px to the left and 0px to the right, sometimes add 2px to the right
        and 0px to the left and sometimes add 1px to both sides. Set `position`
        to ``center`` to prevent that.

        **Supported dtypes**:

        See :func:`~imgaug.augmenters.size.pad`.

        Parameters
        ----------
        width : int or None
            Pad images up to this minimum width.
            If ``None``, image widths will not be altered.

        height : int or None
            Pad images up to this minimum height.
            If ``None``, image heights will not be altered.

        pad_mode : imgaug.ALL or str or list of str or imgaug.parameters.StochasticParameter, optional
            See :func:`~imgaug.augmenters.size.CropAndPad.__init__`.

        pad_cval : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
            See :func:`~imgaug.augmenters.size.CropAndPad.__init__`.

        position : {'uniform', 'normal', 'center', 'left-top', 'left-center', 'left-bottom', 'center-top', 'center-center', 'center-bottom', 'right-top', 'right-center', 'right-bottom'} or tuple of float or StochasticParameter or tuple of StochasticParameter, optional
            Sets the center point of the padding, which determines how the
            required padding amounts are distributed to each side. For a ``tuple``
            ``(a, b)``, both ``a`` and ``b`` are expected to be in range
            ``[0.0, 1.0]`` and describe the fraction of padding applied to the
            left/right (low/high values for ``a``) and the fraction of padding
            applied to the top/bottom (low/high values for ``b``). A padding
            position at ``(0.5, 0.5)`` would be the center of the image and
            distribute the padding equally to all sides. A padding position at
            ``(0.0, 1.0)`` would be the left-bottom and would apply 100% of the
            required padding to the bottom and left sides of the image so that
            the bottom left corner becomes more and more the new image
            center (depending on how much is padded).

                * If string ``uniform`` then the share of padding is randomly and
                uniformly distributed over each side.
                Equivalent to ``(Uniform(0.0, 1.0), Uniform(0.0, 1.0))``.
                * If string ``normal`` then the share of padding is distributed
                based on a normal distribution, leading to a focus on the
                center of the images.
                Equivalent to
                ``(Clip(Normal(0.5, 0.45/2), 0, 1),
                Clip(Normal(0.5, 0.45/2), 0, 1))``.
                * If string ``center`` then center point of the padding is
                identical to the image center.
                Equivalent to ``(0.5, 0.5)``.
                * If a string matching regex
                ``^(left|center|right)-(top|center|bottom)$``, e.g. ``left-top``
                or ``center-bottom`` then sets the center point of the padding
                to the X-Y position matching that description.
                * If a tuple of float, then expected to have exactly two entries
                between ``0.0`` and ``1.0``, which will always be used as the
                combination the position matching (x, y) form.
                * If a ``StochasticParameter``, then that parameter will be queried
                once per call to ``augment_*()`` to get ``Nx2`` center positions
                in ``(x, y)`` form (with ``N`` the number of images).
                * If a ``tuple`` of ``StochasticParameter``, then expected to have
                exactly two entries that will both be queried per call to
                ``augment_*()``, each for ``(N,)`` values, to get the center
                positions. First parameter is used for ``x`` coordinates,
                second for ``y`` coordinates.

        seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        name : None or str, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            Old name for parameter `seed`.
            Its usage will not yet cause a deprecation warning,
            but it is still recommended to use `seed` now.
            Outdated since 0.4.0.

        deterministic : bool, optional
            Deprecated since 0.4.0.
            See method ``to_deterministic()`` for an alternative and for
            details about what the "deterministic mode" actually does.
        """
        transform = iaa.PadToFixedSize(width, height, pad_mode, pad_cval, position, seed, name, random_state,
                                       deterministic)
        augmented_image = transform(image=self.image)
        self.image = augmented_image

        return self

    def PadToMultiplesOf(self, width_multiple=10, height_multiple=10, pad_mode='constant', pad_cval=0,
                         position='uniform', seed=None, name=None, random_state='deprecated',
                         deterministic='deprecated'):
        """
        Pad images until their height/width is a multiple of a value.

        Added in 0.4.0.

        **Supported dtypes**:

        See :class:`~imgaug.augmenters.size.PadToFixedSize`.

        Parameters
        ----------
        width_multiple : int or None
            Multiple for the width. Images will be padded until their
            width is a multiple of this value.
            If ``None``, image widths will not be altered.

        height_multiple : int or None
            Multiple for the height. Images will be padded until their
            height is a multiple of this value.
            If ``None``, image heights will not be altered.

        pad_mode : imgaug.ALL or str or list of str or imgaug.parameters.StochasticParameter, optional
            See :func:`~imgaug.augmenters.size.PadToFixedSize.__init__`.

        pad_cval : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
            See :func:`~imgaug.augmenters.size.PadToFixedSize.__init__`.

        position : {'uniform', 'normal', 'center', 'left-top', 'left-center', 'left-bottom', 'center-top', 'center-center', 'center-bottom', 'right-top', 'right-center', 'right-bottom'} or tuple of float or StochasticParameter or tuple of StochasticParameter, optional
            See :func:`PadToFixedSize.__init__`.

        seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        name : None or str, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            Old name for parameter `seed`.
            Its usage will not yet cause a deprecation warning,
            but it is still recommended to use `seed` now.
            Outdated since 0.4.0.

        deterministic : bool, optional
            Deprecated since 0.4.0.
            See method ``to_deterministic()`` for an alternative and for
            details about what the "deterministic mode" actually does.
        """
        transform = iaa.PadToMultiplesOf(width_multiple, height_multiple, pad_mode, pad_cval, position, seed, name,
                                         random_state, deterministic)
        augmented_image = transform(image=self.image)
        self.image = augmented_image

        return self

    def CropToMultiplesOf(self, width_multiple=10, height_multiple=10, position='uniform', seed=None, name=None,
                          random_state='deprecated', deterministic='deprecated'):
        """
        Crop images down until their height/width is a multiple of a value.

        .. note::

            For a given axis size ``A`` and multiple ``M``, if ``A`` is in the
            interval ``[0 .. M]``, the axis will not be changed.
            As a result, this augmenter can still produce axis sizes that are
            not multiples of the given values.

        Added in 0.4.0.

        **Supported dtypes**:

        See :class:`~imgaug.augmenters.size.CropToFixedSize`.

        Parameters
        ----------
        width_multiple : int or None
            Multiple for the width. Images will be cropped down until their
            width is a multiple of this value.
            If ``None``, image widths will not be altered.

        height_multiple : int or None
            Multiple for the height. Images will be cropped down until their
            height is a multiple of this value.
            If ``None``, image heights will not be altered.

        position : {'uniform', 'normal', 'center', 'left-top', 'left-center', 'left-bottom', 'center-top', 'center-center', 'center-bottom', 'right-top', 'right-center', 'right-bottom'} or tuple of float or StochasticParameter or tuple of StochasticParameter, optional
            See :func:`CropToFixedSize.__init__`.

        seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        name : None or str, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            Old name for parameter `seed`.
            Its usage will not yet cause a deprecation warning,
            but it is still recommended to use `seed` now.
            Outdated since 0.4.0.

        deterministic : bool, optional
            Deprecated since 0.4.0.
            See method ``to_deterministic()`` for an alternative and for
            details about what the "deterministic mode" actually does.
        """
        transform = iaa.CropToMultiplesOf(width_multiple, height_multiple, position, seed, name, random_state,
                                          deterministic)
        augmented_image = transform(image=self.image)
        self.image = augmented_image

        return self

    def CropToPowersOf(self, width_base=2, height_base=3, position='uniform', seed=None, name=None,
                       random_state='deprecated', deterministic='deprecated'):
        """
        Crop images until their height/width is a power of a base.

        This augmenter removes pixels from an axis with size ``S`` leading to the
        new size ``S'`` until ``S' = B^E`` is fulfilled, where ``B`` is a
        provided base (e.g. ``2``) and ``E`` is an exponent from the discrete
        interval ``[1 .. inf)``.

        .. note::

            This augmenter does nothing for axes with size less than ``B^1 = B``.
            If you have images with ``S < B^1``, it is recommended
            to combine this augmenter with a padding augmenter that pads each
            axis up to ``B``.

        Added in 0.4.0.

        **Supported dtypes**:

        See :class:`~imgaug.augmenters.size.CropToFixedSize`.

        Parameters
        ----------
        width_base : int or None
            Base for the width. Images will be cropped down until their
            width fulfills ``width' = width_base ^ E`` with ``E`` being any
            natural number.
            If ``None``, image widths will not be altered.

        height_base : int or None
            Base for the height. Images will be cropped down until their
            height fulfills ``height' = height_base ^ E`` with ``E`` being any
            natural number.
            If ``None``, image heights will not be altered.

        position : {'uniform', 'normal', 'center', 'left-top', 'left-center', 'left-bottom', 'center-top', 'center-center', 'center-bottom', 'right-top', 'right-center', 'right-bottom'} or tuple of float or StochasticParameter or tuple of StochasticParameter, optional
            See :func:`CropToFixedSize.__init__`.

        seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        name : None or str, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            Old name for parameter `seed`.
            Its usage will not yet cause a deprecation warning,
            but it is still recommended to use `seed` now.
            Outdated since 0.4.0.

        deterministic : bool, optional
            Deprecated since 0.4.0.
            See method ``to_deterministic()`` for an alternative and for
            details about what the "deterministic mode" actually does.
        """
        transform = iaa.CropToPowersOf(width_base, height_base, position, seed, name, random_state, deterministic)
        augmented_image = transform(image=self.image)
        self.image = augmented_image

        return self

    def PadToPowersOf(self, width_base, height_base, pad_mode='constant', pad_cval=0, position='uniform', seed=None,
                      name=None, random_state='deprecated', deterministic='deprecated'):
        """
        Pad images until their height/width is a power of a base.

        This augmenter adds pixels to an axis with size ``S`` leading to the
        new size ``S'`` until ``S' = B^E`` is fulfilled, where ``B`` is a
        provided base (e.g. ``2``) and ``E`` is an exponent from the discrete
        interval ``[1 .. inf)``.

        Added in 0.4.0.

        **Supported dtypes**:

        See :class:`~imgaug.augmenters.size.PadToFixedSize`.

        Parameters
        ----------
        width_base : int or None
            Base for the width. Images will be padded down until their
            width fulfills ``width' = width_base ^ E`` with ``E`` being any
            natural number.
            If ``None``, image widths will not be altered.

        height_base : int or None
            Base for the height. Images will be padded until their
            height fulfills ``height' = height_base ^ E`` with ``E`` being any
            natural number.
            If ``None``, image heights will not be altered.

        pad_mode : imgaug.ALL or str or list of str or imgaug.parameters.StochasticParameter, optional
            See :func:`~imgaug.augmenters.size.PadToFixedSize.__init__`.

        pad_cval : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
            See :func:`~imgaug.augmenters.size.PadToFixedSize.__init__`.

        position : {'uniform', 'normal', 'center', 'left-top', 'left-center', 'left-bottom', 'center-top', 'center-center', 'center-bottom', 'right-top', 'right-center', 'right-bottom'} or tuple of float or StochasticParameter or tuple of StochasticParameter, optional
            See :func:`PadToFixedSize.__init__`.

        seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        name : None or str, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            Old name for parameter `seed`.
            Its usage will not yet cause a deprecation warning,
            but it is still recommended to use `seed` now.
            Outdated since 0.4.0.

        deterministic : bool, optional
            Deprecated since 0.4.0.
            See method ``to_deterministic()`` for an alternative and for
            details about what the "deterministic mode" actually does.
        """
        transform = iaa.PadToPowersOf(width_base, height_base, pad_mode, pad_cval, position, seed, name, random_state,
                                      deterministic)
        augmented_image = transform(image=self.image)
        self.image = augmented_image

        return self

    def CropToAspectRatio(self, aspect_ratio=2.0, position='uniform', seed=None, name=None, random_state='deprecated',
                          deterministic='deprecated'):
        """
        Crop images until their width/height matches an aspect ratio.

        This augmenter removes either rows or columns until the image reaches
        the desired aspect ratio given in ``width / height``. The cropping
        operation is stopped once the desired aspect ratio is reached or the image
        side to crop reaches a size of ``1``. If any side of the image starts
        with a size of ``0``, the image will not be changed.

        Added in 0.4.0.

        **Supported dtypes**:

        See :class:`~imgaug.augmenters.size.CropToFixedSize`.

        Parameters
        ----------
        aspect_ratio : number
            The desired aspect ratio, given as ``width/height``. E.g. a ratio
            of ``2.0`` denotes an image that is twice as wide as it is high.

        position : {'uniform', 'normal', 'center', 'left-top', 'left-center', 'left-bottom', 'center-top', 'center-center', 'center-bottom', 'right-top', 'right-center', 'right-bottom'} or tuple of float or StochasticParameter or tuple of StochasticParameter, optional
            See :func:`CropToFixedSize.__init__`.

        seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        name : None or str, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            Old name for parameter `seed`.
            Its usage will not yet cause a deprecation warning,
            but it is still recommended to use `seed` now.
            Outdated since 0.4.0.

        deterministic : bool, optional
            Deprecated since 0.4.0.
            See method ``to_deterministic()`` for an alternative and for
            details about what the "deterministic mode" actually does.
        """
        transform = iaa.CropToAspectRatio(aspect_ratio, position, seed, name, random_state, deterministic)
        augmented_image = transform(image=self.image)
        self.image = augmented_image

        return self

    def PadToAspectRatio(self, aspect_ratio=2.0, pad_mode='constant', pad_cval=0, position='uniform', seed=None,
                         name=None, random_state='deprecated', deterministic='deprecated'):
        """
        Pad images until their width/height matches an aspect ratio.

        This augmenter adds either rows or columns until the image reaches
        the desired aspect ratio given in ``width / height``.

        Added in 0.4.0.

        **Supported dtypes**:

        See :class:`~imgaug.augmenters.size.PadToFixedSize`.

        Parameters
        ----------
        aspect_ratio : number
            The desired aspect ratio, given as ``width/height``. E.g. a ratio
            of ``2.0`` denotes an image that is twice as wide as it is high.

        position : {'uniform', 'normal', 'center', 'left-top', 'left-center', 'left-bottom', 'center-top', 'center-center', 'center-bottom', 'right-top', 'right-center', 'right-bottom'} or tuple of float or StochasticParameter or tuple of StochasticParameter, optional
            See :func:`PadToFixedSize.__init__`.

        pad_mode : imgaug.ALL or str or list of str or imgaug.parameters.StochasticParameter, optional
            See :func:`~imgaug.augmenters.size.PadToFixedSize.__init__`.

        pad_cval : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
            See :func:`~imgaug.augmenters.size.PadToFixedSize.__init__`.

        seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        name : None or str, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            Old name for parameter `seed`.
            Its usage will not yet cause a deprecation warning,
            but it is still recommended to use `seed` now.
            Outdated since 0.4.0.

        deterministic : bool, optional
            Deprecated since 0.4.0.
            See method ``to_deterministic()`` for an alternative and for
            details about what the "deterministic mode" actually does.
        """
        transform = iaa.PadToAspectRatio(aspect_ratio, pad_mode, pad_cval, position, seed, name, random_state,
                                         deterministic)
        augmented_image = transform(image=self.image)
        self.image = augmented_image

        return self

    def CropToSquare(self, position='uniform', seed=None, name=None, random_state='deprecated',
                     deterministic='deprecated'):
        """
        Crop images until their width and height are identical.

        This is identical to :class:`~imgaug.augmenters.size.CropToAspectRatio`
        with ``aspect_ratio=1.0``.

        Images with axis sizes of ``0`` will not be altered.

        Added in 0.4.0.

        **Supported dtypes**:

        See :class:`~imgaug.augmenters.size.CropToFixedSize`.

        Parameters
        ----------
        position : {'uniform', 'normal', 'center', 'left-top', 'left-center', 'left-bottom', 'center-top', 'center-center', 'center-bottom', 'right-top', 'right-center', 'right-bottom'} or tuple of float or StochasticParameter or tuple of StochasticParameter, optional
            See :func:`CropToFixedSize.__init__`.

        seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        name : None or str, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            Old name for parameter `seed`.
            Its usage will not yet cause a deprecation warning,
            but it is still recommended to use `seed` now.
            Outdated since 0.4.0.

        deterministic : bool, optional
            Deprecated since 0.4.0.
            See method ``to_deterministic()`` for an alternative and for
            details about what the "deterministic mode" actually does.
        """
        transform = iaa.CropToSquare(position, seed, name, random_state, deterministic)
        augmented_image = transform(image=self.image)
        self.image = augmented_image

        return self

    def PadToSquare(self, pad_mode='constant', pad_cval=0, position='uniform', seed=None, name=None,
                    random_state='deprecated', deterministic='deprecated'):
        """
        Pad images until their height and width are identical.

        This augmenter is identical to
        :class:`~imgaug.augmenters.size.PadToAspectRatio` with ``aspect_ratio=1.0``.

        Added in 0.4.0.

        **Supported dtypes**:

        See :class:`~imgaug.augmenters.size.PadToFixedSize`.

        Parameters
        ----------
        position : {'uniform', 'normal', 'center', 'left-top', 'left-center', 'left-bottom', 'center-top', 'center-center', 'center-bottom', 'right-top', 'right-center', 'right-bottom'} or tuple of float or StochasticParameter or tuple of StochasticParameter, optional
            See :func:`PadToFixedSize.__init__`.

        pad_mode : imgaug.ALL or str or list of str or imgaug.parameters.StochasticParameter, optional
            See :func:`~imgaug.augmenters.size.PadToFixedSize.__init__`.

        pad_cval : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
            See :func:`~imgaug.augmenters.size.PadToFixedSize.__init__`.

        seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        name : None or str, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            Old name for parameter `seed`.
            Its usage will not yet cause a deprecation warning,
            but it is still recommended to use `seed` now.
            Outdated since 0.4.0.

        deterministic : bool, optional
            Deprecated since 0.4.0.
            See method ``to_deterministic()`` for an alternative and for
            details about what the "deterministic mode" actually does.
        """
        transform = iaa.PadToSquare(pad_mode, pad_cval, position, seed, name, random_state, deterministic)
        augmented_image = transform(image=self.image)
        self.image = augmented_image

        return self

    def CenterPadToFixedSize(self, width=30, height=20, pad_mode='constant', pad_cval=0, seed=None, name=None,
                             random_state='deprecated', deterministic='deprecated'):
        """
        Pad images equally on all sides up to given minimum heights/widths.

        This is an alias for :class:`~imgaug.augmenters.size.PadToFixedSize`
        with ``position="center"``. It spreads the pad amounts equally over
        all image sides, while :class:`~imgaug.augmenters.size.PadToFixedSize`
        by defaults spreads them randomly.

        Added in 0.4.0.

        **Supported dtypes**:

        See :class:`~imgaug.augmenters.size.PadToFixedSize`.

        Parameters
        ----------
        width : int or None
            See :func:`PadToFixedSize.__init__`.

        height : int or None
            See :func:`PadToFixedSize.__init__`.

        pad_mode : imgaug.ALL or str or list of str or imgaug.parameters.StochasticParameter, optional
            See :func:`PadToFixedSize.__init__`.

        pad_cval : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
            See :func:`PadToFixedSize.__init__`.

        seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        name : None or str, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            Old name for parameter `seed`.
            Its usage will not yet cause a deprecation warning,
            but it is still recommended to use `seed` now.
            Outdated since 0.4.0.

        deterministic : bool, optional
            Deprecated since 0.4.0.
            See method ``to_deterministic()`` for an alternative and for
            details about what the "deterministic mode" actually does.
        """
        transform = iaa.CenterPadToFixedSize(width, height, pad_mode, pad_cval, seed, name, random_state, deterministic)
        augmented_image = transform(image=self.image)
        self.image = augmented_image

        return self

    def CenterCropToFixedSize(self, width=30, height=20, seed=None, name=None, random_state='deprecated',
                              deterministic='deprecated'):
        """
        Take a crop from the center of each image.

        This is an alias for :class:`~imgaug.augmenters.size.CropToFixedSize` with
        ``position="center"``.

        .. note::

            If images already have a width and/or height below the provided
            width and/or height then this augmenter will do nothing for the
            respective axis. Hence, resulting images can be smaller than the
            provided axis sizes.

        Added in 0.4.0.

        **Supported dtypes**:

        See :class:`~imgaug.augmenters.size.CropToFixedSize`.

        Parameters
        ----------
        width : int or None
            See :func:`CropToFixedSize.__init__`.

        height : int or None
            See :func:`CropToFixedSize.__init__`.

        seed : None or int or imgaug.random.RNG or numpy.random.Generator or numCenterCropToMultiplesOf(width_multiple, height_multiple, seed=None, name=None, random_state='deprecated', deterministic='deprecated'):py.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        name : None or str, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            Old name for parameter `seed`.
            Its usage will not yet cause a deprecation warning,
            but it is still recommended to use `seed` now.
            Outdated since 0.4.0.

        deterministic : bool, optional
            Deprecated since 0.4.0.
            See method ``to_deterministic()`` for an alternative and for
            details about what the "deterministic mode" actually does.
        """
        transform = iaa.CenterCropToFixedSize(width, height, seed, name, random_state, deterministic)
        augmented_image = transform(image=self.image)
        self.image = augmented_image

        return self

    def CenterCropToMultiplesOf(self, width_multiple=10, height_multiple=10, seed=None, name=None,
                                random_state='deprecated', deterministic='deprecated'):
        """
        Crop images equally on all sides until H/W are multiples of given values.

        This is the same as :class:`~imgaug.augmenters.size.CropToMultiplesOf`,
        but uses ``position="center"`` by default, which spreads the crop amounts
        equally over all image sides, while
        :class:`~imgaug.augmenters.size.CropToMultiplesOf` by default spreads
        them randomly.

        Added in 0.4.0.

        **Supported dtypes**:

        See :class:`~imgaug.augmenters.size.CropToFixedSize`.

        Parameters
        ----------
        width_multiple : int or None
            See :func:`CropToMultiplesOf.__init__`.

        height_multiple : int or None
            See :func:`CropToMultiplesOf.__init__`.

        seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        name : None or str, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            Old name for parameter `seed`.
            Its usage will not yet cause a deprecation warning,
            but it is still recommended to use `seed` now.
            Outdated since 0.4.0.

        deterministic : bool, optional
            Deprecated since 0.4.0.
            See method ``to_deterministic()`` for an alternative and for
            details about what the "deterministic mode" actually does.
        """

        transform = iaa.CenterCropToMultiplesOf(width_multiple, height_multiple, seed, name, random_state,
                                                deterministic)
        augmented_image = transform(image=self.image)
        self.image = augmented_image

        return self

    def CenterPadToMultiplesOf(self, width_multiple=10, height_multiple=10, pad_mode='constant', pad_cval=0, seed=None,
                               name=None, random_state='deprecated', deterministic='deprecated'):
        """
        Pad images equally on all sides until H/W are multiples of given values.

        This is the same as :class:`~imgaug.augmenters.size.PadToMultiplesOf`, but
        uses ``position="center"`` by default, which spreads the pad amounts
        equally over all image sides, while
        :class:`~imgaug.augmenters.size.PadToMultiplesOf` by default spreads them
        randomly.

        Added in 0.4.0.

        **Supported dtypes**:

        See :class:`~imgaug.augmenters.size.PadToFixedSize`.

        Parameters
        ----------
        width_multiple : int or None
            See :func:`PadToMultiplesOf.__init__`.

        height_multiple : int or None
            See :func:`PadToMultiplesOf.__init__`.

        pad_mode : imgaug.ALL or str or list of str or imgaug.parameters.StochasticParameter, optional
            See :func:`~imgaug.augmenters.size.PadToMultiplesOf.__init__`.

        pad_cval : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
            See :func:`~imgaug.augmenters.size.PadToMultiplesOf.__init__`.

        seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        name : None or str, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            Old name for parameter `seed`.
            Its usage will not yet cause a deprecation warning,
            but it is still recommended to use `seed` now.
            Outdated since 0.4.0.

        deterministic : bool, optional
            Deprecated since 0.4.0.
            See method ``to_deterministic()`` for an alternative and for
            details about what the "deterministic mode" actually does.
        """
        transform = iaa.CenterPadToMultiplesOf(width_multiple, height_multiple, pad_mode, pad_cval, seed, name,
                                               random_state, deterministic)
        augmented_image = transform(image=self.image)
        self.image = augmented_image

        return self

    def CenterCropToPowersOf(self, width_base=2, height_base=3, seed=None, name=None, random_state='deprecated',
                             deterministic='deprecated'):
        """
        Crop images equally on all sides until H/W is a power of a base.

        This is the same as :class:`~imgaug.augmenters.size.CropToPowersOf`, but
        uses ``position="center"`` by default, which spreads the crop amounts
        equally over all image sides, while
        :class:`~imgaug.augmenters.size.CropToPowersOf` by default spreads them
        randomly.

        Added in 0.4.0.

        **Supported dtypes**:

        See :class:`~imgaug.augmenters.size.CropToFixedSize`.

        Parameters
        ----------
        width_base : int or None
            See :func:`CropToPowersOf.__init__`.

        height_base : int or None
            See :func:`CropToPowersOf.__init__`.

        seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        name : None or str, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            Old name for parameter `seed`.
            Its usage will not yet cause a deprecation warning,
            but it is still recommended to use `seed` now.
            Outdated since 0.4.0.

        deterministic : bool, optional
            Deprecated since 0.4.0.
            See method ``to_deterministic()`` for an alternative and for
            details about what the "deterministic mode" actually does.
        """
        transform = iaa.CenterCropToPowersOf(width_base, height_base, seed, name, random_state, deterministic)
        augmented_image = transform(image=self.image)
        self.image = augmented_image

        return self

    def CenterPadToPowersOf(self, width_base=2, height_base=3, pad_mode='constant', pad_cval=0, seed=None, name=None,
                            random_state='deprecated', deterministic='deprecated'):
        """
        Pad images equally on all sides until H/W is a power of a base.

        This is the same as :class:`~imgaug.augmenters.size.PadToPowersOf`, but uses
        ``position="center"`` by default, which spreads the pad amounts equally
        over all image sides, while :class:`~imgaug.augmenters.size.PadToPowersOf`
        by default spreads them randomly.

        Added in 0.4.0.

        **Supported dtypes**:

        See :class:`~imgaug.augmenters.size.PadToFixedSize`.

        Parameters
        ----------
        width_base : int or None
            See :func:`PadToPowersOf.__init__`.

        height_base : int or None
            See :func:`PadToPowersOf.__init__`.

        pad_mode : imgaug.ALL or str or list of str or imgaug.parameters.StochasticParameter, optional
            See :func:`~imgaug.augmenters.size.PadToPowersOf.__init__`.

        pad_cval : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
            See :func:`~imgaug.augmenters.size.PadToPowersOf.__init__`.

        seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        name : None or str, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            Old name for parameter `seed`.
            Its usage will not yet cause a deprecation warning,
            but it is still recommended to use `seed` now.
            Outdated since 0.4.0.

        deterministic : bool, optional
            Deprecated since 0.4.0.
            See method ``to_deterministic()`` for an alternative and for
            details about what the "deterministic mode" actually does.
        """
        transform = iaa.CenterPadToPowersOf(width_base, height_base, seed, name, random_state, deterministic)
        augmented_image = transform(image=self.image)
        self.image = augmented_image

        return self

    def CenterCropToAspectRatio(self, aspect_ratio=2.0, seed=None, name=None, random_state='deprecated',
                                deterministic='deprecated'):
        """
        Crop images equally on all sides until they reach an aspect ratio.

        This is the same as :class:`~imgaug.augmenters.size.CropToAspectRatio`, but
        uses ``position="center"`` by default, which spreads the crop amounts
        equally over all image sides, while
        :class:`~imgaug.augmenters.size.CropToAspectRatio` by default spreads
        them randomly.

        Added in 0.4.0.

        **Supported dtypes**:

        See :class:`~imgaug.augmenters.size.CropToFixedSize`.

        Parameters
        ----------
        aspect_ratio : number
            See :func:`CropToAspectRatio.__init__`.

        seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        name : None or str, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            Old name for parameter `seed`.
            Its usage will not yet cause a deprecation warning,
            but it is still recommended to use `seed` now.
            Outdated since 0.4.0.

        deterministic : bool, optional
            Deprecated since 0.4.0.
            See method ``to_deterministic()`` for an alternative and for
            details about what the "deterministic mode" actually does.
        """
        transform = iaa.CenterCropToAspectRatio(aspect_ratio, seed, name, random_state, deterministic)
        augmented_image = transform(image=self.image)
        self.image = augmented_image

        return self

    def CenterPadToAspectRatio(self, aspect_ratio=2.0, pad_mode='constant', pad_cval=0, seed=None, name=None,
                               random_state='deprecated', deterministic='deprecated'):
        """
        Pad images equally on all sides until H/W matches an aspect ratio.

        This is the same as :class:`~imgaug.augmenters.size.PadToAspectRatio`, but
        uses ``position="center"`` by default, which spreads the pad amounts
        equally over all image sides, while
        :class:`~imgaug.augmenters.size.PadToAspectRatio` by default spreads them
        randomly.

        Added in 0.4.0.

        **Supported dtypes**:

        See :class:`~imgaug.augmenters.size.PadToFixedSize`.

        Parameters
        ----------
        aspect_ratio : number
            See :func:`PadToAspectRatio.__init__`.

        name : None or str, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        pad_mode : imgaug.ALL or str or list of str or imgaug.parameters.StochasticParameter, optional
            See :func:`~imgaug.augmenters.size.PadToAspectRatio.__init__`.

        pad_cval : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
            See :func:`~imgaug.augmenters.size.PadToAspectRatio.__init__`.

        deterministic : bool, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.
        """
        transform = iaa.CenterPadToAspectRatio(aspect_ratio, pad_mode, pad_cval, seed, name, random_state,
                                               deterministic)
        augmented_image = transform(image=self.image)
        self.image = augmented_image

        return self

    def CenterCropToSquare(self, seed=None, name=None, random_state='deprecated', deterministic='deprecated'):
        """
        Crop images equally on all sides until their height/width are identical.

        In contrast to :class:`~imgaug.augmenters.size.CropToSquare`, this
        augmenter always tries to spread the columns/rows to remove equally over
        both sides of the respective axis to be cropped.
        :class:`~imgaug.augmenters.size.CropToAspectRatio` by default spreads the
        croppings randomly.

        This augmenter is identical to :class:`~imgaug.augmenters.size.CropToSquare`
        with ``position="center"``, and thereby the same as
        :class:`~imgaug.augmenters.size.CropToAspectRatio` with
        ``aspect_ratio=1.0, position="center"``.

        Images with axis sizes of ``0`` will not be altered.

        Added in 0.4.0.

        **Supported dtypes**:

        See :class:`~imgaug.augmenters.size.CropToFixedSize`.

        Parameters
        ----------
        seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        name : None or str, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            Old name for parameter `seed`.
            Its usage will not yet cause a deprecation warning,
            but it is still recommended to use `seed` now.
            Outdated since 0.4.0.

        deterministic : bool, optional
            Deprecated since 0.4.0.
            See method ``to_deterministic()`` for an alternative and for
            details about what the "deterministic mode" actually does.
        """
        transform = iaa.CenterCropToSquare(seed, name, random_state, deterministic)
        augmented_image = transform(image=self.image)
        self.image = augmented_image

        return self

    def CenterPadToSquare(self, pad_mode='constant', pad_cval=0, seed=None, name=None, random_state='deprecated',
                          deterministic='deprecated'):
        """
        Pad images equally on all sides until their height & width are identical.

        This is the same as :class:`~imgaug.augmenters.size.PadToSquare`, but uses
        ``position="center"`` by default, which spreads the pad amounts equally
        over all image sides, while :class:`~imgaug.augmenters.size.PadToSquare`
        by default spreads them randomly. This augmenter is thus also identical to
        :class:`~imgaug.augmenters.size.PadToAspectRatio` with
        ``aspect_ratio=1.0, position="center"``.

        Added in 0.4.0.

        **Supported dtypes**:

        See :class:`~imgaug.augmenters.size.PadToFixedSize`.

        Parameters
        ----------
        name : None or str, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        pad_mode : imgaug.ALL or str or list of str or imgaug.parameters.StochasticParameter, optional
            See :func:`~imgaug.augmenters.size.PadToAspectRatio.__init__`.

        pad_cval : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
            See :func:`~imgaug.augmenters.size.PadToAspectRatio.__init__`.

        deterministic : bool, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.
        """
        transform = iaa.CenterPadToSquare(pad_mode, pad_cval, seed, name, random_state, deterministic)
        augmented_image = transform(image=self.image)
        self.image = augmented_image

        return self

    def FastSnowyLandscape(self, lightness_threshold=(100, 255), lightness_multiplier=(1.0, 4.0), from_colorspace='RGB',
                           seed=None, name=None, random_state='deprecated', deterministic='deprecated'):
        """
        Convert non-snowy landscapes to snowy ones.

        This augmenter expects to get an image that roughly shows a landscape.

        This augmenter is based on the method proposed in
        https://medium.freecodecamp.org/image-augmentation-make-it-rain-make-it-snow-how-to-modify-a-photo-with-machine-learning-163c0cb3843f?gi=bca4a13e634c

        **Supported dtypes**:

            * ``uint8``: yes; fully tested
            * ``uint16``: no (1)
            * ``uint32``: no (1)
            * ``uint64``: no (1)
            * ``int8``: no (1)
            * ``int16``: no (1)
            * ``int32``: no (1)
            * ``int64``: no (1)
            * ``float16``: no (1)
            * ``float32``: no (1)
            * ``float64``: no (1)
            * ``float128``: no (1)
            * ``bool``: no (1)

            - (1) This augmenter is based on a colorspace conversion to HLS.
                Hence, only RGB ``uint8`` inputs are sensible.

        Parameters
        ----------
        lightness_threshold : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
            All pixels with lightness in HLS colorspace that is below this value
            will have their lightness increased by `lightness_multiplier`.

                * If a ``number``, then that value will always be used.
                * If a ``tuple`` ``(a, b)``, then a value will be uniformly sampled
                per image from the discrete interval ``[a..b]``.
                * If a ``list``, then a random value will be sampled from that
                ``list`` per image.
                * If a ``StochasticParameter``, then a value will be sampled
                per image from that parameter.

        lightness_multiplier : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
            Multiplier for pixel's lightness value in HLS colorspace.
            Affects all pixels selected via `lightness_threshold`.

                * If a ``number``, then that value will always be used.
                * If a ``tuple`` ``(a, b)``, then a value will be uniformly sampled
                per image from the discrete interval ``[a..b]``.
                * If a ``list``, then a random value will be sampled from that
                ``list`` per image.
                * If a ``StochasticParameter``, then a value will be sampled
                per image from that parameter.

        from_colorspace : str, optional
            The source colorspace of the input images.
            See :func:`~imgaug.augmenters.color.ChangeColorspace.__init__`.

        seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        name : None or str, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            Old name for parameter `seed`.
            Its usage will not yet cause a deprecation warning,
            but it is still recommended to use `seed` now.
            Outdated since 0.4.0.

        deterministic : bool, optional
            Deprecated since 0.4.0.
            See method ``to_deterministic()`` for an alternative and for
            details about what the "deterministic mode" actually does.
        """
        transform = iaa.FastSnowyLandscape(lightness_threshold, lightness_multiplier, from_colorspace, seed, name,
                                           random_state, deterministic)
        augmented_image = transform(image=self.image)
        self.image = augmented_image

        return self

    def Clouds(self, seed=None, name=None, random_state='deprecated', deterministic='deprecated'):
        """
        Add clouds to images.

        This is a wrapper around :class:`~imgaug.augmenters.weather.CloudLayer`.
        It executes 1 to 2 layers per image, leading to varying densities and
        frequency patterns of clouds.

        This augmenter seems to be fairly robust w.r.t. the image size. Tested
        with ``96x128``, ``192x256`` and ``960x1280``.

        **Supported dtypes**:

            * ``uint8``: yes; tested
            * ``uint16``: no (1)
            * ``uint32``: no (1)
            * ``uint64``: no (1)
            * ``int8``: no (1)
            * ``int16``: no (1)
            * ``int32``: no (1)
            * ``int64``: no (1)
            * ``float16``: no (1)
            * ``float32``: no (1)
            * ``float64``: no (1)
            * ``float128``: no (1)
            * ``bool``: no (1)

            - (1) Parameters of this augmenter are optimized for the value range
                of ``uint8``. While other dtypes may be accepted, they will lead
                to images augmented in ways inappropriate for the respective
                dtype.

        Parameters
        ----------
        seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        name : None or str, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            Old name for parameter `seed`.
            Its usage will not yet cause a deprecation warning,
            but it is still recommended to use `seed` now.
            Outdated since 0.4.0.

        deterministic : bool, optional
            Deprecated since 0.4.0.
            See method ``to_deterministic()`` for an alternative and for
            details about what the "deterministic mode" actually does.
        """
        transform = iaa.Clouds(seed, name, random_state, deterministic)
        augmented_image = transform(image=self.image)
        self.image = augmented_image

        return self

    def FDA(self, reference_images=None, beta_limit=0.1, read_fn=None, p=1.0):
        """
            Fourier Domain Adaptation from https://github.com/YanchaoYang/FDA
            Simple "style transfer".
            Parameters
            ----------
                reference_images (List[str] or List(np.ndarray)): List of file paths for reference images
                    or list of reference images.
                beta_limit (float or tuple of float): coefficient beta from paper. Recommended less 0.3.
                read_fn (Callable): Used-defined function to read image. Function should get image path and return numpy
                    array of image pixels.

            Targets
            -------
                image

            Image types
            -----------
                uint8, float32

            Reference
            ---------
                https://github.com/YanchaoYang/FDA
                https://openaccess.thecvf.com/content_CVPR_2020/papers/Yang_FDA_Fourier_Domain_Adaptation_for_Semantic_Segmentation_CVPR_2020_paper.pdf
        """
        if reference_images is None:
            reference_images = np.random.randint(0, 256, [100, 100, 3], dtype=np.uint8)
            read_fn = lambda x: x
            transform = A.FDA(reference_images, beta_limit, read_fn, p)
            augmented_image = transform(image=self.image)['image']
            self.image = augmented_image
        else:
            reference_images = glob.glob(os.path.join(reference_images, "*.jpg"))

            def read_fn(image):
                return cv2.imread(image)

            transform = A.FDA(reference_images, beta_limit, read_fn, p)
            augmented_image = transform(image=self.image)['image']
            self.image = augmented_image


    # add probability to function
    def RandomShape(self, num=1, color='blue', shape = 'circle'):

        # not complete
        """generate random shapes with different range of specific color
            """

        rects =[]
        # color_dict_HSV = {'black': [[180, 255, 30], [0, 0, 0]],
        #                   'white': [[180, 18, 255], [0, 0, 231]],
        #                   'red1': [[180, 255, 255], [159, 50, 70]],
        #                   'red2': [[9, 255, 255], [0, 50, 70]],
        #                   'green': [[89, 255, 255], [36, 50, 70]],
        #                   'blue': [[128, 255, 255], [90, 50, 70]],
        #                   'yellow': [[35, 255, 255], [25, 50, 70]],
        #                   'purple': [[158, 255, 255], [129, 50, 70]],
        #                   'orange': [[24, 255, 255], [10, 50, 70]],
        #                   'gray': [[180, 18, 230], [0, 0, 40]]}

        annotation = self.bbox
        height, width, c = self.image.shape

        for i in range(len(annotation)):
            if annotation[i] == 'empty':
                rect = [0, 0, 0, 0]
                rects.append(rect)
            else:
                x, y, w, h = float(annotation[i][1]), float(annotation[i][2]), float(annotation[i][3]), float(annotation[i][4])

                xmin = int((x * width) - (w * width) / 2.0)
                xmax = int((x * width) + (w * width) / 2.0)
                ymin = int((y * height) - (h * height) / 2.0)
                ymax = int((y * height) + (h * height) / 2.0)

                rect = [xmin, ymin, xmax, ymax]
                rects.append(rect)

        # pick random color from base_color
        def rand_color(base_color, cnt):
            i = 0
            color_list = []
            while i < cnt:
                # red color range
                if base_color == 'red':
                    random_color = (r, g, b)
                    color_list.append(random_color)
                    i = i + 1
                # green base_color range
                elif base_color == 'green':
                    random_color = (r, g, b)
                    color_list.append(random_color)
                    i = i + 1
                # blue base_color range
                elif base_color == 'blue':
                    r = random.randint(150, 255)
                    g = random.randint(0, 100)
                    b = random.randint(0, 100)

                    random_color = (r, g, b)
                    color_list.append(random_color)
                    i = i+1
            return color_list

        # check the overlap between generated shape location and object bounding boxes
        def has_overlap(rexts, R2):
            for i in range(len(rects)):
                R1 = rects[i]
                if (R1[0] >= R2[2]) or (R1[2] <= R2[0]) or (R1[3] <= R2[1]) or (R1[1] >= R2[3]):
                    return False
                else:
                    return True

        def make_circle(image, color):
            overlap = True
            while overlap:
                x = random.randint(0, width/2)
                y = random.randint(0, height/2)
                center_coordinates = (x, y)
                radius = random.randint(0, height/4)
                R2 = [x-radius, y-radius, x+radius, y+radius]
                overlap = has_overlap(rects, R2)

            return cv2.circle(image, center_coordinates, radius, color, -1)

        def make_rectangle(image, color):
            overlap = True
            while overlap:
                x1 = random.randint(0, width/4)
                y1 = random.randint(0, height/4)
                x2 = random.randint(0, width/4)
                y2 = random.randint(0, height/4)
                if x1 > x2:
                    R2 = [x2, y1, x1, y2]
                    start_point = (x2, y1)
                    end_point = (x1, y2)
                    overlap = has_overlap(rects, R2)
                elif y1 > y2:
                    start_point = (x1, y2)
                    end_point = (x2, y1)
                    R2 = [x1, y2, x2, y1]
                    overlap = has_overlap(rects, R2)
                else:
                    start_point = (x1, y1)
                    end_point = (x2, y2)
                    R2 = [x1, y1, x2, y2]
                    overlap = has_overlap(rects, R2)

            return cv2.rectangle(image, start_point, end_point, color, -1)

        def make_triangle(image, color):
            # overlap = True
            # while overlap:
            p1_1 = random.randint(0, width)
            p1_2 = random.randint(0, height)
            p2_1 = random.randint(0, width)
            p2_2 = random.randint(0, height)
            p3_1 = random.randint(0, width)
            p3_2 = random.randint(0, height)

            pts = [(p1_1, p1_2), (p2_1, p2_2), (p3_1, p3_2)]

            return cv2.fillPoly(image, np.array([pts]), color)

        # add random shape with random range color in image
        need_to_run = random.random() < self.p
        if need_to_run:
            color = rand_color(color, num)
            for i in range(len(color)):
                # generate shape
                if shape == 'circle':
                    augment_image = make_circle(self.image, color[i])
                    self.image = augment_image
                elif shape == 'rectangle':
                    augment_image = make_rectangle(self.image, color[i])
                    self.image = augment_image
                elif shape == 'triangle':
                    augment_image = make_triangle(self.image, color[i])
                    self.image = augment_image
        else:
            pass

        return self

    def color_adaptation(self, source,  target, name='EmdTransport'):
        """
        4 color adaptation transforms (EmdTransport, SinkhornTransport, Image_mapping_linear, Image_mapping_gaussian)

        Parameters
        -----------
            source : source image path
            target: target image path
            name: str
                EmdTransport
                SinkhornTransport
                Image_mapping_linear
                Image_mapping_gaussian

        Targets
        ----------
            image

        """
        rng = np.random.RandomState(42)

        def im2mat(img):
            """Converts and image to matrix (one pixel per line)"""
            return img.reshape((img.shape[0] * img.shape[1], img.shape[2]))

        def mat2im(X, shape):
            """Converts back a matrix to an image"""
            return X.reshape(shape)

        def minmax(img):
            return np.clip(img, 0, 1)

        # Loading images
        I1 = plt.imread(source).astype(np.float64) / 256
        I2 = plt.imread(target).astype(np.float64) / 256

        X1 = im2mat(I1)
        X2 = im2mat(I2)

        # training samples
        nb = 500
        idx1 = rng.randint(X1.shape[0], size=(nb,))
        idx2 = rng.randint(X2.shape[0], size=(nb,))

        Xs = X1[idx1, :]
        Xt = X2[idx2, :]

        if name == 'EmdTransport':
            ot_emd = ot.da.EMDTransport()
            ot_emd.fit(Xs=Xs, Xt=Xt)
            transp_Xs_emd = ot_emd.transform(Xs=X1)
            self.image = minmax(mat2im(transp_Xs_emd, I1.shape))*255

        elif name == 'SinkhornTransport':
            ot_sinkhorn = ot.da.SinkhornTransport(reg_e=1e-1)
            ot_sinkhorn.fit(Xs=Xs, Xt=Xt)
            transp_Xs_sinkhorn = ot_sinkhorn.transform(Xs=X1)
            self.image = minmax(mat2im(transp_Xs_sinkhorn, I1.shape))*255

        elif name == 'Image_mapping_linear':
            ot_mapping_linear = ot.da.MappingTransport(
                mu=1e0, eta=1e-8, bias=True, max_iter=20, verbose=True)
            ot_mapping_linear.fit(Xs=Xs, Xt=Xt)
            X1tl = ot_mapping_linear.transform(Xs=X1)
            self.image = minmax(mat2im(X1tl, I1.shape))*255

        elif name == 'Image_mapping_gaussian':
            ot_mapping_gaussian = ot.da.MappingTransport(
                mu=1e0, eta=1e-2, sigma=1, bias=False, max_iter=10, verbose=True)
            ot_mapping_gaussian.fit(Xs=Xs, Xt=Xt)
            X1tn = ot_mapping_gaussian.transform(Xs=X1)  # use the estimated mapping
            self.image = minmax(mat2im(X1tn, I1.shape))*255

        return self

    def DustyLook(self):
        DataAugmentation.ChangeColorTemperature(self, kelvin=(20000, 30000))
        DataAugmentation.Pepper(self, p=0.09)
        DataAugmentation.GaussianBlur(self, blur_limit=(5, 11))
        DataAugmentation.Contrast(self, severity=1)

        return self

    def FDM(self, source):

        print('HERRRRRRR')
        """
        Parameters
        -----------
            source : source image path
        ----------
            image

        """

        cv2.imwrite('temp/temp.jpg',self.image)
        myImg = 'temp/temp.jpg'


        result_path = 'temp/out.jpg'
        ref_list = []
        for i in glob.glob(source + '/*.PNG'):
            ref_list.append(i)
        # print(len(ref_list))



        # counter = 0
        # for i in glob.glob(source + '/*'):
        #     if i.endswith('jpg'):
        #         name = i.split('/')[-1]
        #         #selecting random image from test
        images = random.choice(ref_list)
        print(images)
        #print(ref_list)
        #random_img = random.choice(images)

        print('rand bg image', images)



        os.system(f'python /media/2TB_1/Sabaghian/DATAAUG/Data_Augmentation/FDM/image-statistics-matching/main.py fdm -s rgb -c 0,1,2 {myImg} {images} {result_path}')
                #counter = counter+1

        img = cv2.imread(result_path)
        self.image = img

        return self