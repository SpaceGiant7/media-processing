import numpy as np


class ColorConversion:
    @staticmethod
    def rbg_to_hsv(im):
        return np.stack(
                (
                    ColorConversion.calculate_hue(im),
                    ColorConversion.calculate_saturation_hsv(im),
                    ColorConversion.calculate_value(im),
                ),
                axis=2,
            )

    @staticmethod
    def rbg_to_hsi(im):
        return np.stack(
            (
                (ColorConversion.calculate_hue(im)),
                (ColorConversion.calculate_saturation_hsi(im)),
                (ColorConversion.calculate_intensity(im)),
            ),
            axis=2,
        )

    @staticmethod
    def rbg_to_hsl(im):
        return np.stack(
            (
                (ColorConversion.calculate_hue(im)),
                (ColorConversion.calculate_saturation_hsl(im)),
                (ColorConversion.calculate_lightness(im)),
            ),
            axis=2,
        )

    @staticmethod
    def calculate_hue(im):
        maximums = np.max(im, axis=2)
        minimums = np.min(im, axis=2)
        deltas = maximums - minimums

        # Condition for equal min and max of color channels
        hue = (deltas != 0).astype(float)

        # Condition for Red maximum
        boolean_index = np.logical_and(maximums == im[:, :, 0], hue != 0)
        hue[boolean_index] = np.mod(
            (im[boolean_index, 1] - im[boolean_index, 2]) / deltas[boolean_index], 6
        )

        # Condition for Green maximum
        boolean_index = np.logical_and(maximums == im[:, :, 1], hue != 0)
        hue[boolean_index] = (im[boolean_index, 2] - im[boolean_index, 0]) / deltas[
            boolean_index
        ] + 2

        # Condition for Blue maximum
        boolean_index = np.logical_and(maximums == im[:, :, 2], hue != 0)
        hue[boolean_index] = (im[boolean_index, 0] - im[boolean_index, 1]) / deltas[
            boolean_index
        ] + 4

        # Normalize Hue
        hue /= 6
        return hue

    @staticmethod
    def calculate_saturation_hsv(im):
        V = ColorConversion.calculate_value(im)
        C = np.max(im, axis=2) - np.min(im, axis=2)

        S = np.zeros((im.shape[0], im.shape[1]))
        S[V != 0] = C[V != 0] / V[V != 0]
        return S

    @staticmethod
    def calculate_value(im):
        return np.max(im, 2)

    @staticmethod
    def calculate_intensity(im):
        return np.mean(im, axis=2)

    @staticmethod
    def calculate_saturation_hsi(im):
        intensity = ColorConversion.calculate_intensity(im)
        minimums = np.min(im, axis=2)

        saturation = np.zeros((im.shape[0], im.shape[1]))
        saturation[intensity != 0] = 1 - minimums[intensity != 0] / intensity[intensity != 0]
        return saturation

    @staticmethod
    def calculate_lightness(im):
        return (np.max(im, axis=2) + np.min(im, axis=2)) / 2

    @staticmethod
    def calculate_saturation_hsl(im):
        lightness = ColorConversion.calculate_lightness(im)
        delta = np.max(im, axis=2) - np.min(im, axis=2)

        saturation = np.zeros((im.shape[0], im.shape[1]))
        index = np.logical_and(lightness < 0.999, lightness > 0.001)
        saturation[index] = delta[index] / (1 - np.absolute(2 * lightness[index] - 1))
        return saturation
