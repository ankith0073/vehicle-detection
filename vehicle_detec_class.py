# author: Ankith Manjunath
# Date : 11.05.17

#script to track vehicle detections
import numpy as np

class veh_info():
    def __init__(self,n, size_of_x_points):
        self.n  = n
        self.heatmaps = np.zeros(shape = [n, size_of_x_points[0],size_of_x_points[1],size_of_x_points[2]] , dtype = np.float32)
        self.best_heatmap = np.zeros(shape = size_of_x_points , dtype = np.float32)
        self.n_avg = 0

        self.rectangles = []


    def update_heatmap(self, heatmap):
        self.heatmaps[0] = heatmap
        #find the number of valid rows for averaging
        self.n_avg = (self.n_avg + 1) % self.n


    def update_circ_buf(self):
        for i in range(1,self.n):
            self.heatmaps[self.n - i] = self.heatmaps[self.n - i - 1]
        self.heatmaps[0] = np.zeros(self.heatmaps[0].shape)

    def get_best_fit(self):
        self.best_heatmap = np.sum(self.heatmaps, axis=0)/self.n_avg

    def update_rectangles(self, rectangles):
        self.rectangles.append(rectangles)
        if len(self.rectangles) > self.n:
            self.rectangles = self.rectangles[len(self.rectangles)-self.n:]



