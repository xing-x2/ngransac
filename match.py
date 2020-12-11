import cv2 as cv
import numpy as np
import argparse
import sys
from adalam import AdalamFilter
from compare import find_ground_truth


def extract_keypoints(impath):
    im = cv.imread(impath, cv.IMREAD_COLOR)
    d = cv.xfeatures2d.SIFT_create(nfeatures=8000, contrastThreshold=1e-5)
    kp1, desc1 = d.detectAndCompute(im, mask=np.ones(shape=im.shape[:-1] + (1,),
                                                              dtype=np.uint8))
    pts = np.array([k.pt for k in kp1], dtype=np.float32)
    ors = np.array([k.angle for k in kp1], dtype=np.float32)
    scs = np.array([k.size for k in kp1], dtype=np.float32)
    return pts, ors, scs, desc1, im


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

    im1 = cv.resize(img1, target_1, interpolation=cv.INTER_AREA)
    im2 = cv.resize(img2, target_2, interpolation=cv.INTER_AREA)

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
        cv.line(vis, (x1, y1), (x2, y2), [0, 255, 0], 1)
        # count += 1
        # if count > 500:
        #     break
    
    # count = 0
    for (x1, y1), (x2, y2) in zip(o1, o2):
        cv.line(vis, (x1, y1), (x2, y2), [0, 0, 255], 1)
    #     count += 1
    #     if count > 300:
    #         break

    cv.imshow("AdaLAM example", vis)
    cv.waitKey()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--im2", required=True)
    opt = p.parse_args()
    if opt.name == "boat":
        ext = "pgm"
    else:
        ext = "ppm"
    path1 = "GT_pics/"+opt.name+"/imgs/img1."+ext
    path2 = "GT_pics/"+opt.name+"/imgs/img"+opt.im2+"."+ext
    gt_path = "ground_truth/"+opt.name+"/"+opt.name+"_1_"+opt.im2+"_TP.txt"
    print(path1, path2)
    
    # results = function(im1, im2)
    k1, o1, s1, d1, im1 = extract_keypoints(path1) # (opt.im1)
    k2, o2, s2, d2, im2 = extract_keypoints(path2) # (opt.im2)

    matcher = AdalamFilter()
    matches = matcher.match_and_filter(k1=k1, k2=k2,
                                       o1=o1, o2=o2,
                                       d1=d1, d2=d2,
                                       s1=s1, s2=s2,
                                       im1shape=im1.shape[:2], im2shape=im2.shape[:2]).cpu().numpy()

    k1=k1[matches[:, 0]]
    k2=k2[matches[:, 1]]

    true_pos, false_pos = find_ground_truth(k1, k2, gt_path)
    pts1 = k1[true_pos]
    pts2 = k2[true_pos]
    out1 = k1[false_pos]
    out2 = k2[false_pos]
    
    show_matches(im1, im2, pts1, pts2, out1, out2)

    # Save as image and rename

    exit()








