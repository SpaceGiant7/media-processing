import cv2
import matplotlib.pyplot as plt
from numpy import np
from abc import ABC, abstractmethod


class BaseImage(ABC):
    def __init__(self, im=None):
        if im is not None:
            self.im = im
            self.size = im.shape

    @abstractmethod
    def apply_filter(self, mask) -> "BaseImage":
        pass

    @abstractmethod
    def get_hist(self):
        pass

    @abstractmethod
    def show(self) -> None:
        pass

    @abstractmethod
    def as_bw(self):
        pass

    @abstractmethod
    def as_rgb(self):
        pass

    @abstractmethod
    def as_hsi(self):
        pass

    @abstractmethod
    def as_hsl(self):
        pass

    @abstractmethod
    def as_hsv(self):
        pass

class BWImage(BaseImage):
    def __init__(self, im=None):
        if im is not None:
            self.im = im
            self.size = im.shape

    def apply_filter(self, mask: np.array):
        self.im = cv2.filter2D(self.im, -1, mask)

    def get_hist(self) -> np.array:
        intIm = np.round(255 * self.im)
        return np.array([np.count_nonzero(intIm == x) for x in range(255)])

    def show(self) -> None:
        plt.figure()
        plt.imshow(np.stack((self.im, self.im, self.im), axis=2))
        plt.show()

    def as_bw(self):
        return self

    def as_rgb(self):
        return None

    def as_hsi(self):
        return None

    def as_hsl(self):
        return None

    def as_hsv(self):
        return None


class RBGImage(BaseImage):
    def __init__(self, im=None):
        if im is not None:
            self.im = im
            self.size = im.shape

    def apply_filter(self, mask: np.array):
        pass

    def get_hist(self) -> np.array:
        return self.as_bw().get_hist()

    def show(self) -> None:
        plt.figure()
        plt.imshow(self.im)
        plt.show()

    def as_bw(self):
        return BWImage(0.21 * self.im[:, :, 0] + 0.72 * self.im[:, :, 1] + 0.07 * self.im[:, :, 2])

    def as_rgb(self):
        return self

    def as_hsi(self):
        return HSIImage(
            np.stack((self.__calculate_hue(),
                      self.__calculate_saturation_hsi(),
                      self.__calculate_intensity()),
                     axis=2))

    def as_hsl(self):
        ### Calcualte Hue Channel ###
        H = self.__calculate_hue()

        ### Calculate Lightness Channel ##
        L = self.__calculate_lightness()

        ## Calculate Saturation Channel ##
        S = self.__calculate_saturation_hsl()
        return HSLImage(np.stack((H, S, L), axis=2))

    def as_hsv(self):
        ### Calcualte Hue Channel ###
        H = self.__calculate_hue()

        ### Calculate Value Channel ##
        V = self.__calculate_value()

        ## Calculate Saturation Channel ##
        S = self.__calculate_saturation_hsv()

        return HSVImage(np.stack((H, S, V), axis=2))

    def __calculate_hue(self):
        M = np.max(self.im, axis=2)
        m = np.min(self.im, axis=2)
        C = M - m
        ### Calculate Hue Channel ###

        # Condition for equal min and max of color channels
        H = (C != 0).astype(float)

        # Condition for Red maximum
        booleanIndex = np.logical_and(M == self.im[:, :, 0], H != 0)
        H[booleanIndex] = np.mod((self.im[booleanIndex, 1] - self.im[booleanIndex, 2]) / C[booleanIndex], 6)

        # Condition for Green maximum
        booleanIndex = np.logical_and(M == self.im[:, :, 1], H != 0)
        H[booleanIndex] = (self.im[booleanIndex, 2] - self.im[booleanIndex, 0]) / C[booleanIndex] + 2

        # Condition for Blue maximum
        booleanIndex = np.logical_and(M == self.im[:, :, 2], H != 0)
        H[booleanIndex] = (self.im[booleanIndex, 0] - self.im[booleanIndex, 1]) / C[booleanIndex] + 4

        # Normalize Hue
        H /= 6
        return H

    def __calculate_intensity(self):
        return np.mean(self.im, axis=2)

    def __calculate_value(self):
        return np.max(self.im, 2)

    def __calculate_lightness(self):
        return 0.299 * self.im[:, :, 0] + 0.587 * self.im[:, :, 1] + 0.114 * self.im[:, :, 2]

    def __calculate_saturation_hsv(self):
        V = self.__calculate_value()
        C = np.max(self.im, axis=2) - np.min(self.im, axis=2)

        S = np.zeros((self.size[0], self.size[1]))
        S[V != 0] = C[V != 0] / V[V != 0]
        return S

    def __calculate_saturation_hsl(self):
        L = self.__calculate_lightness()
        C = np.max(self.im, axis=2) - np.min(self.im, axis=2)

        S = np.zeros((self.size[0], self.size[1]))
        index = np.logical_and(L < 0.999, L > 0.001)
        S[index] = C[index] / (1 - np.absolute(2 * L[index] - 1))
        return S

    def __calculate_saturation_hsi(self):
        I = self.__calculate_intensity()
        m = np.min(self.im, axis=2)

        S = np.zeros((self.size[0], self.size[1]))
        S[I != 0] = 1 - m[I != 0] / I[I != 0]
        return S


class HSIImage(BaseImage):
    def __init__(self, im=None):
        if im is not None:
            self.im = im
            self.size = im.shape

    def apply_filter(self, mask: np.array):
        pass

    def get_hist(self) -> np.array:
        return self.as_bw().get_hist()

    def show(self) -> None:
        plt.figure()
        plt.imshow(self.im)
        plt.show()

    def as_bw(self):
        return None

    def as_rgb(self):
        return None

    def as_hsi(self):
        return self

    def as_hsl(self):
        return None

    def as_hsv(self):
        return None

class HSLImage(BaseImage):
    def __init__(self, im=None):
        if im is not None:
            self.im = im
            self.size = im.shape

    def apply_filter(self, mask: np.array):
        pass

    def get_hist(self) -> np.array:
        return self.as_bw().get_hist()

    def show(self) -> None:
        plt.figure()
        plt.imshow(self.im)
        plt.show()

    def as_bw(self):
        return None

    def as_rgb(self):
        return None

    def as_hsi(self):
        return self

    def as_hsl(self):
        return None

    def as_hsv(self):
        return None

class HSVImage(BaseImage):
    def __init__(self, im=None):
        if im is not None:
            self.im = im
            self.size = im.shape

    def apply_filter(self, mask: np.array):
        pass

    def get_hist(self) -> np.array:
        return self.as_bw().get_hist()

    def show(self) -> None:
        plt.figure()
        plt.imshow(self.im)
        plt.show()

    def as_bw(self):
        return None

    def as_rgb(self):
        return None

    def as_hsi(self):
        return self

    def as_hsl(self):
        return None

    def as_hsv(self):
        return None


