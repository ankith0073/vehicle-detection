# author: Ankith Manjunath
# Date : 26.04.17

#Script with helper function to perform lane detection
import numpy as np
import cv2

#Function to undistort image given the intrinsec and extrinsic matrices
def undistort_frame(image, mtx, dist):
    undist = cv2.undistort(image, mtx, dist, None, mtx)
    return undist

def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):

    # Apply the following steps to img
    # 1) Convert to grayscale
    #gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    if orient == 'x':
        sobel = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    else:
        sobel = cv2.Sobel(img, cv2.CV_64F, 0, 1)

    # 3) Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)

    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

    # 5) Create a mask of 1's where the scaled gradient magnitude
    # is > thresh_min and < thresh_max

    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # 6) Return this mask as your binary_output image
    binary_output = np.copy(sxbinary)  # Remove this line
    return binary_output

def fit_first(binary_warped,left_info, right_info):
    ym_per_pix = 3.048 / 100  # meters per pixel in y dimension, lane line is 10 ft = 3.048 meters
    xm_per_pix = 3.7 / 378  # meters per pixel in x dimension, lane width is 12 ft = 3.7 meters
    histogram = np.sum(binary_warped[np.int(binary_warped.shape[0] / 2):, :], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0] / nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 50
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2)
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    #check if points has been detected on the left side
    if leftx.size != 0 and lefty.size != 0:
        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        left_info.detected = True
        left_info.fit_world = np.polyfit(lefty *ym_per_pix,  leftx*xm_per_pix,2)
        #left_info.current_fit.append(left_fit)
        left_info.current_fit[:,0] = left_fit
    else:
        left_info.detected = False

    if rightx.size != 0 and righty.size != 0:
        right_fit = np.polyfit(righty, rightx, 2)
        right_info.detected = True
        right_info.current_fit[:,0] = right_fit
        right_info.fit_world = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)
    else:
        right_info.detected = False


def fit(binary_warped, left_info, right_info):
    # Assume you now have a new warped binary image
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    ym_per_pix = 30/720  # meters per pixel in y dimension, lane line is 10 ft = 3.048 meters
    xm_per_pix = 3.7/700  # meters per pixel in x dimension, lane width is 12 ft = 3.7 meters
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 50

    left_lane_inds = ((nonzerox > (left_info.best_fit[0] * (nonzeroy ** 2) + left_info.best_fit[1] * nonzeroy + left_info.best_fit[2] - margin)) & (
        nonzerox < (left_info.best_fit[0] * (nonzeroy ** 2) + left_info.best_fit[1] * nonzeroy + left_info.best_fit[2] + margin)))
    right_lane_inds = (
        (nonzerox > (right_info.best_fit[0] * (nonzeroy ** 2) + right_info.best_fit[1] * nonzeroy + right_info.best_fit[2] - margin)) & (
            nonzerox < (right_info.best_fit[0] * (nonzeroy ** 2) + right_info.best_fit[1] * nonzeroy + right_info.best_fit[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # check if points has been detected on the left side
    if leftx.size != 0 and lefty.size != 0:
        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        left_info.detected = True
        left_info.current_fit[:, 0] = left_fit
    else:
        left_info.detected = False


    if rightx.size != 0 and righty.size != 0:
        right_fit = np.polyfit(righty, rightx, 2)
        right_info.detected = True
        right_info.current_fit[:, 0] = right_fit
    else:
        right_info.detected = False



def get_radius_of_curvature(poly_data, y_eval):
    ym_per_pix = 3.048 / 100  # meters per pixel in y dimension, lane line is 10 ft = 3.048 meters
    xm_per_pix = 3.7 / 378  # meters per pixel in x dimension, lane width is 12 ft = 3.7 meters

    curverad = ((1 + (2 * poly_data.fit_world[0] * y_eval +  poly_data.fit_world[1]) ** 2) ** 1.5) / np.absolute(2 *  poly_data.fit_world[0])
    poly_data.radius_of_curvature = curverad

def dist_to_center(poly_data):
    ym_per_pix = 3.048 / 100  # meters per pixel in y dimension, lane line is 10 ft = 3.048 meters
    xm_per_pix = 3.7 / 378  # meters per pixel in x dimension, lane width is 12 ft = 3.7 meters
    center_of_image = 1280 / 2
    y_eval = np.max(poly_data.ploty)
    #dist of line to center of lane is dist from the center of the image to the left or right lane line

    #find x cooresponding to y_eval given a polynomial coefficient
    corr_x = poly_data.current_fit[0,0] * y_eval ** 2 + poly_data.current_fit[1,0] * y_eval + poly_data.current_fit[2,0]

    #Pixel distance between center of image and x
    pixel_dist = np.absolute(corr_x - center_of_image)

    #distance in global
    poly_data.dist_to_center = pixel_dist * xm_per_pix

def car_position_to_center(left_info, right_info):
    #distance of lane center to left and right lines
    dist = 1.85

    #leftlane distance
    left_dist = np.absolute (left_info.dist_to_center - dist)

    # rightlane distance
    right_dist = np.absolute(right_info.dist_to_center - dist)

    return (left_dist + right_dist)/2




