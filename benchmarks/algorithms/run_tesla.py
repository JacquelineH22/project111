import os,csv,re, time
import pickle
import random
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from scipy import stats
from scipy.sparse import issparse
import scanpy as sc
import matplotlib.colors as clr
import matplotlib.pyplot as plt
import cv2
import TESLA as tesla
from IPython.display import Image

print(tesla.__version__)
#Read in gene expression and spatial location
counts=sc.read("/data1/hounaiqiao/wzr/Simulated_Xenium/luca/w100/simulated_square_spot_data.h5ad")
#Read in histology image
img=cv2.imread("/data1/hounaiqiao/wzr/Simulated_Xenium/luca/HE_aligned/wholeslide/align_he.png")

resize_factor=1000/np.min(img.shape[0:2])
resize_width=int(img.shape[1]*resize_factor)
resize_height=int(img.shape[0]*resize_factor)
counts.var.index=[i.upper() for i in counts.var.index]
counts.var_names_make_unique()
counts.raw=counts
sc.pp.log1p(counts) # impute on log scale
if issparse(counts.X):
    counts.X=counts.X.A.copy()

# #Three different algorithms to detect contour, select the best one.Here we use cv2.
# #Important note: If you get incorrect contour for all of the 3 three methods, please double check your array_x, array_y, pixel_x, pixel_y are matched correctly.

# # #-----------------1. Detect contour using cv2-----------------
cnt=tesla.cv2_detect_contour(img, CANNY_THRESH_1 = 30, CANNY_THRESH_2 = 90, 
                             apertureSize=5,L2gradient = True)

# binary=np.zeros((img.shape[0:2]), dtype=np.uint8)
# cv2.drawContours(binary, [cnt], -1, (1), thickness=-1)
# #Enlarged filter
binary=np.zeros((img.shape[0:2]), dtype=np.uint8)
cv2.drawContours(binary, [cnt], -1, (1), thickness=-1)
#Enlarged filter
cnt_enlarged = tesla.scale_contour(cnt, 1.05)
binary_enlarged = np.zeros(img.shape[0:2])
cv2.drawContours(binary_enlarged, [cnt_enlarged], -1, (1), thickness=-1)
img_new = img.copy()
cv2.drawContours(img_new, [cnt], -1, (255), thickness=50)
img_new=cv2.resize(img_new, ((resize_width, resize_height)))
cv2.imwrite('/data1/hounaiqiao/wzr/benchmarks/deconvolution_mapping/result/TESLA/luca/luca_cnt.jpg', img_new)

res=50
rename_dict = {
    'x': 'array_x',
    'y': 'array_y',
    'x_pixel': 'pixel_x',
    'y_pixel': 'pixel_y'
}
counts.obs = counts.obs.rename(columns=rename_dict)
# Note, if the numer of superpixels is too large and take too long, you can increase the res to 100
enhanced_exp_adata=tesla.imputation(img=img, raw=counts, cnt=cnt,
                                    genes=counts.var.index.tolist(),
                                    shape="None", res=res, s=1, k=2,
                                    num_nbs=10)

# Save results
enhanced_exp_adata.write_h5ad("/data1/hounaiqiao/wzr/benchmarks/deconvolution_mapping/result/TESLA/luca/spot100/enhanced_exp.h5ad")
