# author: Ankith Manjunath
# Date : 12.05.17

#script to perform Vehicle detection and lane detection on real time video


import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
import imageio
import pickle
from pathlib import Path
from helper_functions import *
from jitter_info import Line
from skimage.feature import hog
from vehicle_detec_class import veh_info


input_video = 'project_video.mp4'
output_video_folder = './'
out_color_thresolded = imageio.get_writer(output_video_folder + 'output' + input_video  , fps=20)
reader = imageio.get_reader('./' + input_video)


calibration_data_file = Path('/wide_dist_pickle.p')
dist_pickle = pickle.load(open('./wide_dist_pickle.p', 'rb'))
mtx = dist_pickle['mtx']
dist = dist_pickle['dist']

#load trained model
filename = 'finalized_model.p'

svc= pickle.load(open(filename, 'rb'))


average_frames = 20

#Parameters for lane detection
#kernel size definition
ksize = 3

#threshold for x and y
thresh_x_min = 30
thresh_x_max = 150

h = 720
w = 1280
# define source and destination points for transform
src = np.float32([(575,464),
                  (707,464),
                  (258,682),
                  (1049,682)])
dst = np.float32([(450,0),
                  (w-450,0),
                  (450,h),
                  (w-450,h)])

#HSV threshold for color thresholding
threshold = [[0,70],
             [150,255],
             [220,255]]

#create objects of the left and right lane
left_info = Line(average_frames,h)
right_info = Line(average_frames,h)

def lane_finding(img):
    # frame = cv2.imread(img)
    undistort = undistort_frame(img, mtx, dist)

    M = cv2.getPerspectiveTransform(src, dst)
    # get the inverse perspective transform
    Minv = cv2.getPerspectiveTransform(dst, src)
    # Warp the image using OpenCV warpPerspective()
    frame = img
    warped = cv2.warpPerspective(undistort, M,
                                 (frame.shape[1], frame.shape[0]),
                                 flags=cv2.INTER_LINEAR)

    hsv_img = cv2.cvtColor(warped, code=cv2.COLOR_BGR2HSV)

    # color thresholding
    # separate color channels
    h = hsv_img[:, :, 0]
    s = hsv_img[:, :, 1]
    v = hsv_img[:, :, 2]

    h_threshold = np.zeros_like(h)
    h_threshold[np.logical_and(h > threshold[0][0], h < threshold[0][1])] = 1

    s_threshold = np.zeros_like(s)
    s_threshold[np.logical_and(s > threshold[1][0], s < threshold[1][1])] = 1

    v_threshold = np.zeros_like(s)
    v_threshold[np.logical_and(v > threshold[2][0], v < threshold[2][1])] = 1

    h_s_v_combined_mask = np.logical_or(s_threshold, v_threshold)

    # gradient thresholding
    # gradient threshold in horizontal direction the saturation channel
    gradx_s = abs_sobel_thresh(s,
                               orient='x',
                               thresh_min=thresh_x_min,
                               thresh_max=thresh_x_max)

    # gradient threshold in horizontal direction the value channel
    gradx_v = abs_sobel_thresh(v,
                               orient='x',
                               thresh_min=thresh_x_min,
                               thresh_max=thresh_x_max)

    # combine the binary masks
    combined_gradient = np.logical_or(gradx_s, gradx_v)

    # combine color and gradient masks
    combined_gradient_color = np.logical_or(combined_gradient, h_s_v_combined_mask)

    binary_warped = np.zeros_like(v)
    binary_warped[combined_gradient_color == True] = 1

    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    if i == 0:
        fit_first(binary_warped, left_info, right_info)

    if i != 0:
        fit(binary_warped, left_info, right_info)

    # radius of curvature of left lane
    get_radius_of_curvature(left_info, 719)
    get_radius_of_curvature(right_info, 719)

    dist_to_center(left_info)
    dist_to_center(right_info)

    # update x for left and right lanes
    left_info.update_xfitted()
    right_info.update_xfitted()

    # find the average coefficients of left and right
    left_info.get_best_fit()
    right_info.get_best_fit()

    # do the First last out operation
    left_info.update_circ_buf()
    right_info.update_circ_buf()

    position_offset = car_position_to_center(left_info, right_info)

    # Create an image to draw the lines on
    color_warp = np.zeros_like(warped).astype(np.uint8)
    # color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_info.bestx, left_info.ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_info.bestx, right_info.ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    # plot the color in red if offset from center is larger than 0.5, indicating lane departure warning
    if position_offset > 0.5:
        poly_color = (255, 0, 0)
    else:
        poly_color = (0, 255, 0)
    cv2.fillPoly(color_warp, np.int_([pts]), poly_color)
    # plt.imshow(color_warp)

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undistort, 1, newwarp, 0.3, 0)
    # plt.imshow(result)


    font = cv2.FONT_HERSHEY_SIMPLEX
    print_text = 'Curvature     :' + ' {:0.2f}'.format(
        (right_info.radius_of_curvature + left_info.radius_of_curvature) / 2)
    cv2.putText(result, print_text, (500, 300), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    print_text = 'Center Offset :' + ' {:0.2f}'.format(position_offset)
    cv2.putText(result, print_text, (500, 330), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    return result


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

def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    if conv == 'RGB2HSV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # heatmap
def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap


# find the hog feature of the complete image and subsample the windows
# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
    draw_img = np.copy(img)
    img = img.astype(np.float32) / 255
    rectangles = []
    heatmap = np.zeros_like(img)

    img_tosearch = img[ystart:ystop, :, :]
    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

    ch1 = ctrans_tosearch[:, :, 0]
    ch2 = ctrans_tosearch[:, :, 1]
    ch3 = ctrans_tosearch[:, :, 2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
    nfeat_per_block = orient * cell_per_block ** 2

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))
            test_prediction = svc.predict(hog_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)
                # cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6)
                rectangles.append(
                    ((xbox_left, ytop_draw + ystart), (xbox_left + win_draw, ytop_draw + win_draw + ystart)))
                heatmap[ytop_draw+ystart:ytop_draw+win_draw+ystart,xbox_left:xbox_left+win_draw] += 1
    return rectangles, heatmap

from scipy.ndimage.measurements import label

def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img


def process_video_frame_instantaneous(img,result, detection):
    #draw_img = np.copy(img)
    rectangles, heat = find_cars(img, ystart, ystop, scale, svc, None, orient, pix_per_cell, cell_per_block,
                                    None, None)
    if len(rectangles)> 0:
        detection.update_rectangles(rectangles)
    heat = np.zeros_like(img[:,:,0])

    for rect_set in detection.rectangles:
        heat = add_heat(heat, rect_set)

    heat = apply_threshold(heat, 1 + len(detection.rectangles)//2)
    heatmap = np.clip(heat, 0, 255)
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(result), labels)


    return draw_img

colorspace = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = 'ALL'  # Can be 0, 1, 2, or "ALL"
spatial_feat = False
hist_feat = False
hog_feat = True
y_start_stop = [None, None]
overlap = 0.5
# test heatmap images
ystart = 400
ystop = 656
scale = 1.5
number_of_frames_to_average = 15



for i, img in enumerate(reader):
    result = lane_finding(img)
    if i == 0:
        detection = veh_info(number_of_frames_to_average, img.shape)
        detection.n_avg = 1


    draw_img = process_video_frame_instantaneous(img,result, detection)
    out_color_thresolded.append_data(draw_img)

out_color_thresolded.close()
