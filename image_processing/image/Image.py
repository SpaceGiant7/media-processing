import cv2
import matplotlib.pyplot as plt
from numpy import np
from abc import ABC, abstractmethod

from image_processing.image.ColorConversion import ColorConversion


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
        BaseImage.__init__(self, im)

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
        BaseImage.__init__(self, im)

    def apply_filter(self, mask: np.array):
        pass

    def get_hist(self) -> np.array:
        return self.as_bw().get_hist()

    def show(self) -> None:
        plt.figure()
        plt.imshow(self.im)
        plt.show()

    def as_bw(self):
        return BWImage(
            0.21 * self.im[:, :, 0] + 0.72 * self.im[:, :, 1] + 0.07 * self.im[:, :, 2]
        )

    def as_rgb(self):
        return self

    def as_hsi(self):
        return HSIImage(ColorConversion.rbg_to_hsi(self.im))

    def as_hsl(self):
        return HSLImage(ColorConversion.rbg_to_hsl(self.im))

    def as_hsv(self):
        return HSVImage(ColorConversion.rbg_to_hsv(self.im))


class HSIImage(BaseImage):
    def __init__(self, im=None):
        BaseImage.__init__(self, im)

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
        BaseImage.__init__(self, im)

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
        BaseImage.__init__(self, im)

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
