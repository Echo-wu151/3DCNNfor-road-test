import numpy as np
import cv2
import os
from tqdm import tqdm


class LoadImg:

    def __init__(self, width, height, depth):
        self.width = width
        self.height = height
        self.depth = depth

    def loadimg(self, filename, color=0):
        frame = cv2.imread(filename)
        frame = cv2.resize(frame, (32, 32))

        if not color:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return np.array(frame)

    def get_img_classname(self, filename,files_ID):
        y=np.load(filename)
        for label_index in files_ID:
            del_id=int(label_index)-1
            y=np.delete(y,del_id,0)
        y = y[:, 1]

        return y
