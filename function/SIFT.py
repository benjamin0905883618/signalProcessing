import numpy as np
import cv2
import torch

def sift(imgs):
    # tensor to cv2 numpy array
    img = imgs.numpy()
    max_nor = np.max(img)
    img = np.uint8(img * 255 / max_nor) # normalize
    img = img.transpose(0, 2, 3, 1)
    #rint(img[0].shape)
    #cv2.imwrite("img.png", img[0])
    sift_filter = cv2.SIFT_create()
    for i in range(img.shape[0]):
        keypoints, descriptors = sift_filter.detectAndCompute(img[i], None)
        tmp_img = np.zeros_like(img[i]).copy()
        tmp_img = cv2.drawKeypoints(img[i], keypoints, tmp_img, )

        tmp_img = tmp_img.transpose(2, 0, 1)
        tmp_img = np.float64((tmp_img - np.min(tmp_img))/ (np.max(tmp_img) - np.min(tmp_img)))
        imgs[i] = torch.from_numpy(tmp_img)
    return imgs