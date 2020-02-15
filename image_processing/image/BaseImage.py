import cv2
import matplotlib.pyplot as plt
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

    def _as_rgb(self, im):
        return

    @abstractmethod
    def as_hsi(self):
        pass

    @abstractmethod
    def as_hsl(self):
        pass

    @abstractmethod
    def as_hsv(self):
        pass
