from unittest import TestCase
import numpy as np

from image_processing.image.ColorConversion import ColorConversion


class BaseTest(TestCase):
    @staticmethod
    def create_pixel(x, y, z):
        return np.array([x, y, z]).reshape((1, 1, 3))

    @staticmethod
    def create_white():
        return BaseTest.create_pixel(1, 1, 1)

    @staticmethod
    def create_black():
        return BaseTest.create_pixel(0, 0, 0)

    @staticmethod
    def create_red():
        return BaseTest.create_pixel(1, 0, 0)

    @staticmethod
    def create_green():
        return BaseTest.create_pixel(0, 1, 0)

    @staticmethod
    def create_blue():
        return BaseTest.create_pixel(0, 0, 1)

    def base_channel_test(self, function, description, expected_values):
        im = np.array([[[1, 1, 1], [1, 0, 0]], [[0, 1, 0], [0, 0, 1]]])
        result = function(im)
        self.assertAlmostEqual(expected_values[0], result[0][0], msg=description + " does not match for white", places=3)
        self.assertAlmostEqual(expected_values[1], result[0][1], msg=description + " does not match for red", places=3)
        self.assertAlmostEqual(expected_values[2], result[1][0], msg=description + " does not match for green", places=3)
        self.assertAlmostEqual(expected_values[3], result[1][1], msg=description + " does not match for blue", places=3)


class TestColorConversionHue(BaseTest):
    def test_hue(self):
        self.base_channel_test(lambda im: ColorConversion.calculate_hue(im), "Hue", [0, 0, 1/3, 2/3])


class TestColorConversionSaturationHSV(BaseTest):
    def test_saturation_hsv(self):
        self.base_channel_test(lambda im: ColorConversion.calculate_saturation_hsv(im), "Saturation", [0, 1, 1, 1])


class TestColorConversionValue(BaseTest):
    def test_value(self):
        self.base_channel_test(lambda im: ColorConversion.calculate_value(im), "Value", [1, 1, 1, 1])


class TestColorConversionSaturationHSI(BaseTest):
    def test_saturation_hsv(self):
        self.base_channel_test(lambda im: ColorConversion.calculate_saturation_hsi(im), "Saturation", [0, 1, 1, 1])


class TestColorConversionIntensity(BaseTest):
    def test_intensity(self):
        self.base_channel_test(lambda im: ColorConversion.calculate_intensity(im), "Intensity", [1, 1/3, 1/3, 1/3])


class TestColorConversionSaturationHSL(BaseTest):
    def test_saturation_hsl(self):
        self.base_channel_test(lambda im: ColorConversion.calculate_saturation_hsl(im), "Saturation", [0, 1, 1, 1])


class TestColorConversionLightness(BaseTest):
    def test_lightness(self):
        self.base_channel_test(lambda im: ColorConversion.calculate_lightness(im), "Lightness", [1, 1/2, 1/2, 1/2])

