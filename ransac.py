from estimate_homography import *
import numpy as np


class RANSAC:
    _BLUE = [255, 0, 0]
    _GREEN = [0, 255, 0]
    _RED = [0, 0, 255]
    _CYAN = [255, 255, 0]

    _line_thickness = 2
    _radius = 5
    _circ_thickness = 2

    def __init__(self, p=0.99, eps=0.6, n=6, delta=3):
        
        self.n = n
        self.p = p
        self.eps = eps
        self.delta = delta
        self.N = self.compute_N(self.n, self.p, self.eps)


    def compute_N(self, n, p, eps):

        
        N = np.round(np.log(1-p)/np.log(1-(1-eps)**n))
        return N


    def sample_n_datapts(self, n_total, n=6):
        
        idx = np.random.choice(n_total, n, replace=False)

        n_idx = np.setdiff1d(np.arange(n_total), idx)

        return idx, n_idx


    def get_inliers(self, H, pts_in_expected, delta):
        
        pts_in = pts_in_expected[:, 0:2]
        pts_expected = pts_in_expected[:, 2:]

        pts_in = convert_to_homogenous_crd(pts_in, axis=1)  

        est_pts = np.matmul(H, pts_in.T)  

        est_pts = est_pts/est_pts[-1, :]
        est_pts = est_pts.T  

        dst = np.linalg.norm(est_pts[:, 0:2] - pts_expected, axis=1)

        inliers = pts_in_expected[np.where(dst <= delta)]

        outliers = pts_in_expected[np.where(dst > delta)]

        return inliers, outliers



    def run_ransac(self, correspondence):
        
        if isinstance(correspondence, list):
            correspondence = np.array(correspondence)

        
        n_total = correspondence.shape[0]
        self.M = (1-self.eps)*n_total

        print("N: {}, n: {}, M:{}, p: {}, eps: {}, delta: {}".format(self.N, self.n, self.M,
                                                                     self.p, self.eps, self.delta))
        no_iter = 0

        current_inliers = []
        current_inliers_cnt = 0

        current_sample_pts = []
        current_outliers = []

        while no_iter <= self.N:


            idx, n_idx = self.sample_n_datapts(n_total, self.n)

            sample_pts = correspondence[idx]
            other_pts = correspondence[n_idx]

            H = calculate_homography(in_pts=sample_pts[:, 2:], out_pts=sample_pts[:, 0:2])

            inliers, outliers = self.get_inliers(H, other_pts, delta=self.delta)

            inlier_count = inliers.shape[0]

            print("prev_inlier_cnt: {}, new_inlier_cnt: {}".
                  format(current_inliers_cnt, inlier_count))

            if (inlier_count > self.M) and (inlier_count > current_inliers_cnt):
                print(" #### Found better sample of points. Updating #####")
                current_inliers = inliers
                current_outliers = outliers
                current_inliers_cnt = inlier_count
                current_sample_pts = sample_pts

            print(" Done {}/{}".format(no_iter, self.N))

            no_iter += 1

        final_corr_points = np.concatenate((current_sample_pts, current_inliers), axis=0)
        final_H = calculate_homography(in_pts=final_corr_points[:, 2:],
                                                    out_pts=final_corr_points[:, 0:2])

        return current_inliers_cnt, current_inliers, current_outliers, current_sample_pts, final_H


    def draw_lines(self, corr_pts, img_1, img_2, save_path, line_color, pt_color):
        

        h, w, _ = img_1.shape

        img_stack = np.hstack((img_1, img_2))

        for x1, y1, x2, y2 in corr_pts:
            x1_d = int(round(x1))
            y1_d = int(round(y1))

            x2_d = int(round(x2) + w)
            y2_d = int(round(y2))

            cv2.circle(img_stack, (x1_d, y1_d), radius=self._radius, color=pt_color,
                       thickness=self._circ_thickness, lineType=cv2.LINE_AA)

            cv2.circle(img_stack, (x2_d, y2_d), radius=self._radius, color=pt_color,
                       thickness=self._circ_thickness, lineType=cv2.LINE_AA)

            cv2.line(img_stack, (x1_d, y1_d), (x2_d, y2_d), color=line_color,
                     thickness=self._line_thickness)

        cv2.imwrite(save_path, img_stack)




