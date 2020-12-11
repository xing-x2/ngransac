import numpy as np
import cv2
import math
import argparse
import os
import random

import torch
import torch.optim as optim
import ngransac

from network import CNNet
from dataset import SparseDataset
import util
from compare import find_ground_truth

parser = util.create_parser('NG-RANSAC demo for a user defined image pair. Fits an essential matrix (default) or fundamental matrix (-fmat) using OpenCV RANSAC vs. NG-RANSAC.')

parser.add_argument('--image1', '-img1', default='images/demo1.jpg',
	help='path to image 1')

parser.add_argument('--image2', '-img2', default='images/demo2.jpg',
	help='path to image 2')

parser.add_argument('--outimg', '-out', default='demo.png',
	help='demo will store a matching image under this file name')

parser.add_argument('--focallength1', '-fl1', type=float, default=900, 
	help='focal length of image 1 (only used when fitting the essential matrix)')

parser.add_argument('--focallength2', '-fl2', type=float, default=900, 
	help='focal length of image 2 (only used when fitting the essential matrix)')

parser.add_argument('--model', '-m', default='',
	help='model to load, leave empty and the script infers an appropriate pre-trained model from the other settings')

parser.add_argument('--hyps', '-hyps', type=int, default=1000, 
	help='number of hypotheses, i.e. number of RANSAC iterations')

parser.add_argument('--refine', '-ref', action='store_true', 
	help='refine using the 8point algorithm on all inliers, only used for fundamental matrix estimation (-fmat)')

parser.add_argument("--name", required=True)
parser.add_argument("--im2", required=True)

opt = parser.parse_args()

if opt.fmat:
	print("\nFitting Fundamental Matrix...\n")
else:
	print("\nFitting Essential Matrix...\n")

# setup detector
if opt.orb:
	print("Using ORB.\n")
	if opt.nfeatures > 0:
		detector = cv2.ORB_create(nfeatures=8000)
	else:
		detector = cv2.ORB_create()
else:
	if opt.rootsift:
		print("Using RootSIFT.\n")
	else:
		print("Using SIFT.\n")
	if opt.nfeatures > 0:
		detector = cv2.xfeatures2d.SIFT_create(nfeatures=8000, contrastThreshold=1e-5)
	else:
		detector = cv2.xfeatures2d.SIFT_create(nfeatures=8000, contrastThreshold=1e-5)

# loading neural guidence network
model_file = opt.model
if len(model_file) == 0:
	model_file = util.create_session_string('e2e', opt.fmat, opt.orb, opt.rootsift, opt.ratio, opt.session)
	model_file = 'models/weights_' + model_file + '.net'
	print("No model file specified. Inferring pre-trained model from given parameters:")
	print(model_file)

model = CNNet(opt.resblocks)
model.load_state_dict(torch.load(model_file))
model = model.cuda()
model.eval()
print("Successfully loaded model.")

if opt.name == "boat":
    ext = "pgm"
else:
    ext = "ppm"
path1 = "GT_pics/"+opt.name+"/imgs/img1."+ext
path2 = "GT_pics/"+opt.name+"/imgs/img"+opt.im2+"."+ext
print("\nProcessing pair:")
print("Image 1: ", path1)
print("Image 2: ", path2)
gt_path = "ground_truth/"+opt.name+"/"+opt.name+"_1_"+opt.im2+"_TP.txt"
print("GT file: ", gt_path)
# read images
img1_rgb = cv2.imread(path1)
img1 = cv2.cvtColor(img1_rgb, cv2.COLOR_BGR2GRAY)

img2_rgb = cv2.imread(path2)
img2 = cv2.cvtColor(img2_rgb, cv2.COLOR_BGR2GRAY)

# calibration matrices of image 1 and 2, principal point assumed to be at the center
K1 = np.eye(3)
K1[0,0] = K1[1,1] = opt.focallength1
K1[0,2] = img1.shape[1] * 0.5
K1[1,2] = img1.shape[0] * 0.5

K2 = np.eye(3)
K2[0,0] = K2[1,1] = opt.focallength2
K2[0,2] = img2.shape[1] * 0.5
K2[1,2] = img2.shape[0] * 0.5

# detect features
kp1, desc1 = detector.detectAndCompute(img1, None)
kp2, desc2 = detector.detectAndCompute(img2, None)

print("\nFeature found in image 1:", len(kp1))
print("Feature found in image 2:", len(kp2))

# root sift normalization
if opt.rootsift:
	print("Using RootSIFT normalization.")
	desc1 = util.rootSift(desc1)
	desc2 = util.rootSift(desc2)

# feature matching
bf = cv2.BFMatcher()
matches = bf.knnMatch(desc1, desc2, k=2)

good_matches = []
pts1 = []
pts2 = []

#side information for the network (matching ratios in this case)
ratios = []

print("")
if opt.ratio < 1.0:
	print("Using Lowe's ratio filter with", opt.ratio)

for (m,n) in matches:
	if m.distance < opt.ratio*n.distance: # apply Lowe's ratio filter
		good_matches.append(m)
		pts2.append(kp2[m.trainIdx].pt)
		pts1.append(kp1[m.queryIdx].pt)
		ratios.append(m.distance / n.distance)

print("Number of valid matches:", len(good_matches))

pts1 = np.array([pts1])
pts2 = np.array([pts2])

ratios = np.array([ratios])
ratios = np.expand_dims(ratios, 2)

# ------------------------------------------------
# fit fundamental or essential matrix using OPENCV
# ------------------------------------------------
if opt.fmat:

	# === CASE FUNDAMENTAL MATRIX =========================================

	ransac_model, ransac_inliers = cv2.findFundamentalMat(pts1, pts2, ransacReprojThreshold=opt.threshold, confidence=0.999)
else:
	# === CASE ESSENTIAL MATRIX =========================================

	# normalize key point coordinates when fitting the essential matrix
	pts1 = cv2.undistortPoints(pts1, K1, None)
	pts2 = cv2.undistortPoints(pts2, K2, None)

	K = np.eye(3)

	ransac_model, ransac_inliers = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=opt.threshold)

print("\n=== Model found by RANSAC: ==========\n")
print(ransac_model)

print("\nRANSAC Inliers:", ransac_inliers.sum())

# ---------------------------------------------------
# fit fundamental or essential matrix using NG-RANSAC
# ---------------------------------------------------

if opt.fmat:
	# normalize x and y coordinates before passing them to the network
	# normalized by the image size
	util.normalize_pts(pts1, img1.shape)
	util.normalize_pts(pts2, img2.shape)

if opt.nosideinfo:
	# remove side information before passing it to the network
	ratios = np.zeros(ratios.shape)

# create data tensor of feature coordinates and matching ratios
correspondences = np.concatenate((pts1, pts2, ratios), axis=2)
correspondences = np.transpose(correspondences)
correspondences = torch.from_numpy(correspondences).float()

# predict neural guidance, i.e. RANSAC sampling probabilities
log_probs = model(correspondences.unsqueeze(0).cuda())[0] #zero-indexing creates and removes a dummy batch dimension
probs = torch.exp(log_probs).cpu()

out_model = torch.zeros((3, 3)).float() # estimated model
out_inliers = torch.zeros(log_probs.size()) # inlier mask of estimated model
out_gradients = torch.zeros(log_probs.size()) # gradient tensor (only used during training)
rand_seed = 0 # random seed to by used in C++

# run NG-RANSAC
if opt.fmat:

	# === CASE FUNDAMENTAL MATRIX =========================================

	# undo normalization of x and y image coordinates
	util.denormalize_pts(correspondences[0:2], img1.shape)
	util.denormalize_pts(correspondences[2:4], img2.shape)

	incount = ngransac.find_fundamental_mat(correspondences, probs, rand_seed, opt.hyps, opt.threshold, opt.refine, out_model, out_inliers, out_gradients)
else:

	# === CASE ESSENTIAL MATRIX =========================================

	incount = ngransac.find_essential_mat(correspondences, probs, rand_seed, opt.hyps, opt.threshold, out_model, out_inliers, out_gradients)

print("\n=== Model found by NG-RANSAC: =======\n")
print(out_model.numpy())

print("\nNG-RANSAC Inliers: ", int(incount))

# create a visualization of the matching, comparing results of RANSAC and NG-RANSAC
out_inliers = out_inliers.byte().numpy().ravel().tolist()
ransac_inliers = ransac_inliers.ravel().tolist()

def show_matches(img1, img2, k1, k2, out1, out2, target_dim=800.):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    def resize_horizontal(h1, w1, h2, w2, target_height):
        scale_to_align = float(h1) / h2
        current_width = w1 + w2 * scale_to_align
        scale_to_fit = target_height / h1
        target_w1 = int(w1 * scale_to_fit)
        target_w2 = int(w2 * scale_to_align * scale_to_fit)
        target_h = int(target_height)
        return (target_w1, target_h), (target_w2, target_h), scale_to_fit, scale_to_fit * scale_to_align, [target_w1, 0]

    def resize_vertical(h1, w1, h2, w2, target_width):
        scale_to_align = float(w1) / w2
        current_height = h1 + h2 * scale_to_align
        scale_to_fit = target_width / w1
        target_h1 = int(h1 * scale_to_fit)
        target_h2 = int(h2 * scale_to_align * scale_to_fit)
        target_w = int(target_width)
        return (target_w, target_h1), (target_w, target_h2), scale_to_fit, scale_to_fit * scale_to_align, [0, target_h1]

    target_1, target_2, scale1, scale2, offset = resize_vertical(h1, w1, h2, w2, target_dim)

    im1 = cv2.resize(img1_rgb, target_1, interpolation=cv2.INTER_AREA)
    im2 = cv2.resize(img2_rgb, target_2, interpolation=cv2.INTER_AREA)

    h1, w1 = target_1[::-1]
    h2, w2 = target_2[::-1]

    # vis = np.ones((max(h1, h2), w1 + w2, 3), np.uint8) * 255
    vis = np.ones((h1+h2, max(w1, w2), 3), np.uint8) * 255
    vis[:h1, :w1] = im1
    vis[h1:h1+h2, :w2] = im2

    p1 = [np.int32(k * scale1) for k in k1]
    p2 = [np.int32(k * scale2 + offset) for k in k2]
    o1 = [np.int32(out * scale1) for out in out1]
    o2 = [np.int32(out * scale2 + offset) for out in out2]

    # count = 0
    for (x1, y1), (x2, y2) in zip(p1, p2):
        cv2.line(vis, (x1, y1), (x2, y2), [0, 255, 0], 1)
        # count += 1
        # if count > 500:
        #     break
    
    # count = 0
    for (x1, y1), (x2, y2) in zip(o1, o2):
        cv2.line(vis, (x1, y1), (x2, y2), [0, 0, 255], 1)
    #     count += 1
    #     if count > 300:
    #         break

    cv2.imshow("AdaLAM example", vis)
    cv2.waitKey()

km1 = []
km2 = []
# print(len(out_inliers))
for match in good_matches:
	i1 = match.queryIdx
	i2 = match.trainIdx
	km1.append(kp1[i1].pt)
	km2.append(kp2[i2].pt)
km1 = np.array(km1)
km2 = np.array(km2)
# print(km1.shape, km2.shape)

ng_in = []
for i in range(len(out_inliers)):
	if out_inliers[i] == 1:
		ng_in.append(i)

kng1 = km1[ng_in]
kng2 = km2[ng_in]

rs_in = []
for i in range(len(ransac_inliers)):
	if ransac_inliers[i] == 1:
		rs_in.append(i)

krs1 = km1[rs_in]
krs2 = km2[rs_in]


# # print("good_matches\n", good_matches)
# kng1 = np.array(kng1)
# kng2 = np.array(kng2)
# indng = np.array(indng)
# krs1 = np.array(krs1)
# krs2 = np.array(krs2)
# print("kng2\n", kng2)
print("\nngransac:")
true_pos, false_pos = find_ground_truth(kng1, kng2, gt_path, "NGRANSAC")
print("\nransac:")
true_pos, false_pos = find_ground_truth(krs1, krs2, gt_path, "RANSAC")
# pts1 = kng1[true_pos]
# pts2 = kng2[true_pos]
# out1 = kng1[false_pos]
# out2 = kng2[false_pos]

# f = open("NEW_RESULTS.txt", "a")
#     f.write(path + "\n")
#     f.write(str(len(true_pos)) + " true positive out of " + str(len(a1)) + "\n")
#     f.write(str(len(false_pos)) + " false positive out of " + str(len(a1)) + "\n")
#     f.write("\n \n")

# show_matches(img1, img2, pts1, pts2, out1, out2)


match_img_ransac = cv2.drawMatches(img1_rgb, kp1, img2_rgb, kp2, good_matches, None, flags=2, matchColor=(75,180,60), matchesMask = ransac_inliers)
match_img_ngransac = cv2.drawMatches(img1_rgb, kp1, img2_rgb, kp2, good_matches, None, flags=2, matchColor=(200,130,0), matchesMask = out_inliers)
match_img = np.concatenate((match_img_ransac, match_img_ngransac), axis = 0)

cv2.imwrite(opt.outimg, match_img)
print("\nDone. Visualization of the result stored as", opt.outimg)
