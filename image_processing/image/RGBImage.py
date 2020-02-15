from image_processing.image.BaseImage import BaseImage


class RBGImage(BaseImage):
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
        size = self.size[1:2]
        im = 0.21 * self.im[:, :, 0] + 0.72 * self.im[:, :, 1] + 0.07 * self.im[:, :, 2]
        return im

    def as_rgb(self):
        return None

    def as_hsi(self):
        return None

    def as_hsl(self):
        return None

    def as_hsv(self):
        return None