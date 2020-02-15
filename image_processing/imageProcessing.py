import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import misc


class Image:
    def __init__(self, im=None):
        if im is not None:
            self.im = im
            self.size = im.shape 
    
    def applyFilter(self, mask) -> "Image":
        if len(self.size) == 1:
            self.im = cv2.filter2D(self.im, -1, mask)
            
        return self

    def calculateHue(self):
        M = np.max(self.im, axis=2)
        m = np.min(self.im, axis=2)
        C = M - m
        ### Calculate Hue Channel ###
        
        # Condition for equal min and max of color channels
        H = (C!=0).astype(float)
        
        # Condition for Red maximum
        booleanIndex = np.logical_and(M == self.im[:,:,0], H!=0)
        H[booleanIndex] = np.mod((self.im[booleanIndex,1] - self.im[booleanIndex,2])/C[booleanIndex], 6)
        
        # Condition for Green maximum
        booleanIndex = np.logical_and(M == self.im[:,:,1], H!=0)
        H[booleanIndex] = (self.im[booleanIndex,2] - self.im[booleanIndex,0])/C[booleanIndex] + 2

        # Condition for Blue maximum
        booleanIndex = np.logical_and(M == self.im[:,:,2], H!=0)
        H[booleanIndex] = (self.im[booleanIndex,0] - self.im[booleanIndex,1])/C[booleanIndex] + 4

        # Normalize Hue
        H /= 6
        return H 

    def histogramEqualization(self):
        if len(self.size) not in (2,3):
            return self.im
        if len(self.size) == 3:
            self.rgb2hsl()
            self.im[:,:,2] = Image(self.im[:,:,2]).histogramEqualization()
            self.hsl2rgb()
            return self.im
        masks = []
        masks.append(np.array([[0,1,0],[1,1,1],[0,1,0]])/5)
        masks.append(np.ones((3,3))/9)
        masks.append(np.array([[0,0,1,0,0],[0,1,1,1,0],[1,1,1,1,1],[0,1,1,1,0],[0,0,1,0,0]])/13)
        masks.append(np.array([[0,1,1,1,0],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[0,1,1,1,0]])/21)
        masks.append(np.ones((5,5))/25)
        images = []
        images.append(self.im.flatten())
        for m in masks:
            images.append(Image(self.im).applyFilter(m).im.flatten())
        indices = np.argsort(np.lexsort(images))
        # TODO: convert images to linear array, sort using lexsort, re-assemble image
        imSize = self.size[0]*self.size[1]
        returnIm = np.zeros(self.size)
        returnIm = indices.reshape(self.size)/imSize
        #for i in range(self.size[0]):
        #    for j in range(self.size[1]):
        #        returnIm[i,j] = np.floor(255*indices[j + i*self.size[1]]/imSize)/255
        return returnIm
        
         

    def hsl2rgb(self):
        C = (1 - np.absolute(2*self.im[:,:,2] - 1))*self.im[:,:,1]
        H = self.im[:,:,0] * 6
        X = C * (1 - np.absolute(np.mod(H,2) - 1))
        rgb = np.zeros(self.size)
        rgb[np.logical_and(H >= 0, H <= 1),0] = C[np.logical_and(H >=0, H <= 1)]         
        rgb[np.logical_and(H >= 0, H <= 1),1] = X[np.logical_and(H >=0, H <= 1)]         
        rgb[np.logical_and(H >= 1, H <= 2),0] = X[np.logical_and(H >=1, H <= 2)]         
        rgb[np.logical_and(H >= 1, H <= 2),1] = C[np.logical_and(H >=1, H <= 2)]         
        rgb[np.logical_and(H >= 2, H <= 3),1] = C[np.logical_and(H >=2, H <= 3)]         
        rgb[np.logical_and(H >= 2, H <= 3),2] = X[np.logical_and(H >=2, H <= 3)]         
        rgb[np.logical_and(H >= 3, H <= 4),1] = X[np.logical_and(H >=3, H <= 4)]         
        rgb[np.logical_and(H >= 3, H <= 4),2] = C[np.logical_and(H >=3, H <= 4)]         
        rgb[np.logical_and(H >= 4, H <= 5),0] = X[np.logical_and(H >=4, H <= 5)]         
        rgb[np.logical_and(H >= 4, H <= 5),2] = C[np.logical_and(H >=4, H <= 5)]         
        rgb[np.logical_and(H >= 5, H <= 6),0] = C[np.logical_and(H >=5, H <= 6)]         
        rgb[np.logical_and(H >= 5, H <= 6),2] = X[np.logical_and(H >=5, H <= 6)]
        
        m = self.im[:,:,2] - (0.3 * rgb[:,:,0] + 0.59 * rgb[:,:,1] + 0.11 * rgb[:,:,2])
        self.im = np.stack((rgb[:,:,0] + m, rgb[:,:,1] + m, rgb[:,:,2] + m), axis=2) 

    def im2Hist(self):
        intIm = np.round(255*self.im)
        return np.array([np.count_nonzero(intIm == x) for x in range(255)])

    def imRead(self, name):
        self.im = misc.imread(name)/255
        self.size = self.im.shape
        return self

    def imShow(self):
        plt.figure()
        if len(self.size) == 3:
            plt.imshow(self.im)
        else:
            plt.imshow(np.stack((self.im, self.im, self.im), axis=2))
        plt.show()

    def rgb2bw(self):
        if len(self.size) == 3:
            self.size = self.size[1:2]
            self.im = 0.21*im[:,:,0] + 0.72*im[:,:,1] + 0.07*im[:,:,2]
        return self
    
    def rgb2hsi(self):
        
        ### Calcualte Hue Channel ###
        H = self.calculateHue()

        ### Calculate Value Channel ##
        
        I = np.mean(self.im, axis=2)

        ## Calculate Saturation Channel ##
        m = np.min(self.im, axis=2)
       
        S = np.zeros((self.size[0],self.size[1]))
        S[I != 0] = 1 - m[I != 0]/I[I != 0]
        
        self.im = np.stack((H,S,I), axis = 2) 

    def rgb2hsl(self):
        
        ### Calcualte Hue Channel ###
        H = self.calculateHue()

        ### Calculate Value Channel ##
        
        L = 0.299 * self.im[:,:,0] + 0.587 * self.im[:,:,1] + 0.114 * self.im[:,:,2]
        ## Calculate Saturation Channel ##
        M = np.max(self.im, axis=2)
        m = np.min(self.im, axis=2)
        C = M - m

        #L = 0.5*(M + m)      
 
        S = np.zeros((self.size[0],self.size[1]))
        index = np.logical_and(L < 0.999, L > 0.001)
        S[index] = C[index]/(1-np.absolute(2*L[index] - 1))
        self.im = np.stack((H,S,L), axis = 2) 

    def rgb2hsv(self):
        
        ### Calcualte Hue Channel ###
        H = self.calculateHue()

        ### Calculate Value Channel ##
        
        V = np.max(self.im, 2)

        ## Calculate Saturation Channel ##
        M = np.max(self.im, axis=2)
        m = np.min(self.im, axis=2)
        C = M - m
       
        S = np.zeros((self.size[0],self.size[1]))
        S[V != 0] = C[V != 0]/V[V != 0]
        
        self.im = np.stack((H,S,V), axis = 2) 
         

            

def applyFilterbw(im, mask):
    return 
    

def im2hist(im):
    intIm = np.round(255*im) 
    return np.array([np.count_nonzero(intIm == x) for x in range(255)])

def imRead(name):
    return misc.imread(name)/255

def imShow(im):
    plt.figure()
    if len(im.shape) == 3:
        plt.imshow(im)
    else:
        plt.imshow(np.stack((im,im,im), axis=2))
    plt.show()

def makeCheckerboard(resolution, size):
    return np.array([[np.floor(y/size)%2 != np.floor(x/size)%2 for y in range(resolution[0])]for x in range(resolution[1])])

def makeGrid(resolution, lineWidth, duty):
    return np.array([[np.floor((x-(resolution[0]-lineWidth)/2)/lineWidth)%int(1/duty) and np.floor((y - (resolution[1]-lineWidth)/2)/lineWidth)%int(1/duty) for x in range(resolution[0])]for y in range(resolution[1])])

def rgb2bw(im):
    return 0.21*im[:,:,0] + 0.72*im[:,:,1] + 0.07*im[:,:,2]

if __name__ == '__main__':
    im = Image().imRead('doge.jpg')
    im.imShow()
    im.histogramEqualization()
    im.imShow()
