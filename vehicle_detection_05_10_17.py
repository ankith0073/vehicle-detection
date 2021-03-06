# author: Ankith Manjunath
# Date : 10.05.17


# coding: utf-8

# In[2]:

# Import the necessary libraries
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split

import pickle

# In[4]:

# HOG features
# Read in our vehicles and non-vehicles
cars = glob.glob('./vehicles/**/*.png')
notcars = glob.glob('./non-vehicles/**/*.png')



# Define a function to return HOG features and visualization
# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False,
                                  visualise=True, feature_vector=False)
        return features, hog_image
    else:
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False,
                       visualise=False, feature_vector=feature_vec)
        return features



# In[6]:

# Histogram features
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

image = mpimg.imread('test_images/test1.jpg')

# In[ ]:

# extract features
# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, cspace='RGB', orient=9,
                     pix_per_cell=8, cell_per_block=2, hog_channel=0,
                     spatial_feature=False, hist_feat=False, HOG_feature=True):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        temp_features = []
        # Read in each one by one
        image = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif cspace == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else:
            feature_image = np.copy(image)

        if spatial_feature:
            temp = 0
            # spatial_features = bin_spatial(feature_image, size=spatial_size)
            # temp_features.append(spatial_features)

        if hist_feat:
            temp = 0
            # hist_features = color_hist(feature_image, nbins=hist_bins, bins_range=hist_range)
            # temp_features.append(hist_features)
        # Call get_hog_features() with vis=False, feature_vec=True

        if HOG_feature:
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:, :, channel],
                                                         orient, pix_per_cell, cell_per_block,
                                                         vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)
            else:
                hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,
                                                pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            temp_features.append(hog_features)
        # Append the new feature vector to the features list
        features.append(np.concatenate(temp_features))
    # Return list of feature vectors
    return features


# In[ ]:

# define sliding window
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                 xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0] * (1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1] * (1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0] * (xy_overlap[0]))
    ny_buffer = np.int(xy_window[1] * (xy_overlap[1]))
    nx_windows = np.int((xspan - nx_buffer) / nx_pix_per_step)
    ny_windows = np.int((yspan - ny_buffer) / ny_pix_per_step)
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs * nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys * ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list


# In[ ]:

# draw boxes function
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # make a copy of the image
    draw_img = np.copy(img)
    # draw each bounding box on your image copy using cv2.rectangle()
    for bbox in bboxes:
        cv2.rectangle(draw_img, bbox[0], bbox[1], color, thick)
    # return the image copy with boxes drawn
    return draw_img  # Change this line to return image copy with boxes


# Add bounding boxes in this format, these are just example coordinates.
# bboxes = [((100, 100), (200, 200)), ((300, 300), (400, 400))]


# In[ ]:

# Define single image features
# Define a function to extract features from a single image window
# This function is very similar to extract_features()
# just for a single image rather than list of images
def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=False, hist_feat=False, hog_feat=True):
    # 1) Define an empty list to receive features
    img_features = []
    # 2) Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else:
        feature_image = np.copy(img)
    # 3) Compute spatial features if flag is set
    if spatial_feat == True:
        temp = 0
        #spatial_features = bin_spatial(feature_image, size=spatial_size)
        # 4) Append features to list
        #img_features.append(spatial_features)
    # 5) Compute histogram features if flag is set
    if hist_feat == True:
        temp = 0
        #hist_features = color_hist(feature_image, nbins=hist_bins)
        # 6) Append features to list
        #img_features.append(hist_features)
    # 7) Compute HOG features if flag is set

    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:, :, channel],
                                                     orient, pix_per_cell, cell_per_block,
                                                     vis=False, feature_vec=True))
        else:
            hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,
                                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        # 8) Append features to list
        img_features.append(hog_features)

    # 9) Return concatenated array of features
    return np.concatenate(img_features)


# In[ ]:

# Define a function you will pass an image
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler, color_space='RGB',
                   spatial_size=(32, 32), hist_bins=32,
                   hist_range=(0, 256), orient=9,
                   pix_per_cell=8, cell_per_block=2,
                   hog_channel=0, spatial_feat=False,
                   hist_feat=False, hog_feat=True):
    # 1) Create an empty list to receive positive detection windows
    on_windows = []
    # 2) Iterate over all windows in the list
    for window in windows:
        # 3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        # 4) Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space,
                                       spatial_size=spatial_size, hist_bins=hist_bins,
                                       orient=orient, pix_per_cell=pix_per_cell,
                                       cell_per_block=cell_per_block,
                                       hog_channel=hog_channel, spatial_feat=spatial_feat,
                                       hist_feat=hist_feat, hog_feat=hog_feat)
        # 5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        # 6) Predict using your classifier
        prediction = clf.predict(test_features)
        # 7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    # 8) Return windows for positive detections
    return on_windows


# In[ ]:

# get the car and non car images
# car_images = glob.glob('./vehicles_smallset/**/*.jpeg')
# noncar_images = glob.glob('./non-vehicles_smallset/**/*.jpeg')
# print(len(car_images),"number of car images")
# print(len(noncar_images),"number of non-car images")


# In[ ]:

# specify the parameters
car_single_ind = np.random.randint(0, len(cars))
non_car_single_ind = np.random.randint(0, len(notcars))

car_single_img = mpimg.imread(cars[car_single_ind])
non_carsingle_img = mpimg.imread(notcars[non_car_single_ind])

colorspace = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = 'ALL'  # Can be 0, 1, 2, or "ALL"
spatial_feat = False
hist_feat = False
hog_feat = True

# extract features
# car_features, car_hog_image = single_img_features(car_single_img, colorspace, None,
#                         None, orient,
#                         pix_per_cell, cell_per_block, hog_channel,
#                         spatial_feat, hist_feat, hog_feat)
#
# non_car_features, non_car_hog_image = single_img_features(non_carsingle_img, colorspace, None,
#                         None, orient,
#                         pix_per_cell, cell_per_block, hog_channel,
#                         spatial_feat, hist_feat, hog_feat)


# In[ ]:

# extract all features
# limit the number of training samples
limit_samples = 5000

img_indxs = np.random.randint(0, len(cars), limit_samples)

training_cars = np.array(cars)[img_indxs]
training_non_cars = np.array(notcars)[img_indxs]

t = time.time()
car_features = extract_features(training_cars,
                                cspace=colorspace,
                                orient=orient,
                                pix_per_cell=pix_per_cell,
                                cell_per_block=cell_per_block,
                                hog_channel=hog_channel,
                                spatial_feature=spatial_feat,
                                hist_feat=hist_feat,
                                HOG_feature=hog_feat)

non_car_features = extract_features(training_non_cars,
                                    cspace=colorspace,
                                    orient=orient,
                                    pix_per_cell=pix_per_cell,
                                    cell_per_block=cell_per_block,
                                    hog_channel=hog_channel,
                                    spatial_feature=spatial_feat,
                                    hist_feat=hist_feat,
                                    HOG_feature=hog_feat)

t2 = time.time()
print(round(t2 - t, 2), 'Seconds to extract HOG features...')

# stack the training set and labels and normalize then and shuffle to be put in classifier
# Create an array stack of feature vectors
X = np.vstack((car_features, non_car_features)).astype(np.float64)
# Fit a per-column scaler
#X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
#scaled_X = X_scaler.transform(X)
scaled_X = X

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(non_car_features))))

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)

print('Using:', orient, 'orientations', pix_per_cell,
      'pixels per cell and', cell_per_block, 'cells per block')
print('Feature vector length:', len(X_train[0]))

# Linear SVM
# Use a linear SVC
svc = LinearSVC()
# Check the training time for the SVC
t = time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2 - t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t = time.time()
n_predict = 10
print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
print('For these', n_predict, 'labels: ', y_test[0:n_predict])
t2 = time.time()
print(round(t2 - t, 5), 'Seconds to predict', n_predict, 'labels with SVC')

#save the trained model
filename = 'finalized_model.p'
pickle.dump(svc, open(filename, 'wb'))
#scalar = 'standard_Scalar.p'
#pickle.dump(X_scaler,open(scalar, 'wb'))

#filename.close()

# 05.10.2017

# test_images = glob.glob('test_images/*.jpg')
#
# images = []
# titles = []
# y_start_stop = [None, None]
# overlap = 0.5
#
# def convert_color(img, conv='RGB2YCrCb'):
#     if conv == 'RGB2YCrCb':
#         return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
#     if conv == 'BGR2YCrCb':
#         return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
#     if conv == 'RGB2LUV':
#         return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
#     if conv == 'RGB2HSV':
#         return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
#
#     # heatmap
# def add_heat(heatmap, bbox_list):
#     # Iterate through list of bboxes
#     for box in bbox_list:
#         # Add += 1 for all pixels inside each bbox
#         # Assuming each "box" takes the form ((x1, y1), (x2, y2))
#         heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
#
#     # Return updated heatmap
#     return heatmap
#
# # find the hog feature of the complete image and subsample the windows
# # Define a single function that can extract features using hog sub-sampling and make predictions
# def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
#     draw_img = np.copy(img)
#     img = img.astype(np.float32) / 255
#     rectangles = []
#     heatmap = np.zeros_like(img)
#
#     img_tosearch = img[ystart:ystop, :, :]
#     ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
#     if scale != 1:
#         imshape = ctrans_tosearch.shape
#         ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))
#
#     ch1 = ctrans_tosearch[:, :, 0]
#     ch2 = ctrans_tosearch[:, :, 1]
#     ch3 = ctrans_tosearch[:, :, 2]
#
#     # Define blocks and steps as above
#     nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
#     nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
#     nfeat_per_block = orient * cell_per_block ** 2
#
#     # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
#     window = 64
#     nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
#     cells_per_step = 2  # Instead of overlap, define how many cells to step
#     nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
#     nysteps = (nyblocks - nblocks_per_window) // cells_per_step
#
#     # Compute individual channel HOG features for the entire image
#     hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
#     hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
#     hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
#
#     for xb in range(nxsteps):
#         for yb in range(nysteps):
#             ypos = yb * cells_per_step
#             xpos = xb * cells_per_step
#             # Extract HOG for this patch
#             hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
#             hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
#             hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
#             hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
#
#             xleft = xpos * pix_per_cell
#             ytop = ypos * pix_per_cell
#
#             # Extract the image patch
#             subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))
#
#             # Get color features
#             # spatial_features = bin_spatial(subimg, size=spatial_size)
#             # hist_features = color_hist(subimg, nbins=hist_bins)
#
#             # Scale features and make a prediction
#             # test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
#             test_features = X_scaler.transform(np.hstack((hog_features)).reshape(1, -1))
#             # test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
#             test_prediction = svc.predict(test_features)
#
#             if test_prediction == 1:
#                 xbox_left = np.int(xleft * scale)
#                 ytop_draw = np.int(ytop * scale)
#                 win_draw = np.int(window * scale)
#                 # cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6)
#                 rectangles.append(
#                     ((xbox_left, ytop_draw + ystart), (xbox_left + win_draw, ytop_draw + win_draw + ystart)))
#                 heatmap[ytop_draw+ystart:ytop_draw+win_draw+ystart,xbox_left:xbox_left+win_draw] += 1
#     return rectangles, heatmap
#
# from scipy.ndimage.measurements import label
#
# def apply_threshold(heatmap, threshold):
#     # Zero out pixels below the threshold
#     heatmap[heatmap <= threshold] = 0
#     # Return thresholded map
#     return heatmap
#
# def draw_labeled_bboxes(img, labels):
#     # Iterate through all detected cars
#     for car_number in range(1, labels[1]+1):
#         # Find pixels with each car_number label value
#         nonzero = (labels[0] == car_number).nonzero()
#         # Identify x and y values of those pixels
#         nonzeroy = np.array(nonzero[0])
#         nonzerox = np.array(nonzero[1])
#         # Define a bounding box based on min/max x and y
#         bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
#         # Draw the box on the image
#         cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
#     # Return the image
#     return img
#
#
# # test heatmap images
# ystart = 400
# ystop = 656
# scale = 1.5
# test_images = glob.glob('test_images/*.jpg')
# plt.figure()
# # i = 1
#
# # out_images = []
# # out_maps = []
# # for imgs in test_images:
# #     t1 = time.time()
# #
# #     img = mpimg.imread(imgs)
# #
# #     draw_img = np.copy(img)
# #     #img = img.astype(np.float32) / 255
# #     draw_img, heatmap = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block,
# #                                     None, None)
# #     heatmap = apply_threshold(heatmap, 1)
# #     labels = label(heatmap)
# #     draw_img = draw_labeled_bboxes(np.copy(img), labels)
# #     out_images.append(draw_img)
# #     out_maps.append(heatmap)
# #
# #     plt.subplot(len(test_images), 3, i)
# #     plt.imshow(img)
# #     plt.title('Original_image')
# #     plt.axis('off')
# #
# #     plt.subplot(len(test_images),3, i+1)
# #     plt.imshow(heatmap, cmap='hot')
# #     plt.title('Heatmap')
# #     plt.axis('off')
# #
# #     plt.subplot(len(test_images), 3, i + 2)
# #     plt.imshow(draw_img)
# #     plt.title('Detected cars')
# #     plt.axis('off')
# #     i = i + 3
#
# i = 0
#
# def process_video_frame_instantaneous(img):
#     #draw_img = np.copy(img)
#     draw_img, heat = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block,
#                                     None, None)
#     heat = apply_threshold(heat, 2)
#     heatmap = np.clip(heat, 0, 255)
#     labels = label(heatmap)
#     draw_img = draw_labeled_bboxes(np.copy(img), labels)
#     return draw_img
#
#
# import imageio
#
# input_video = 'test_video.mp4'
# output_video_folder = './'
# out_detected = imageio.get_writer(output_video_folder + 'output' + input_video, fps=20)
# reader = imageio.get_reader('./' + input_video)
#
# for i, img in enumerate(reader):
#     draw_img = process_video_frame_instantaneous(img)
#     out_detected.append_data(draw_img)
#
# out_detected.close()
#
#
#
#
