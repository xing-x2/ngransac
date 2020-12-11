import numpy as np
import cv2 as cv
import math

def read_data(file_path):
    f = open(file_path, 'r')
    data = []
    for line in f:
        line_list = line.split(', ')
        data.append([float(coord) for coord in line_list])
    f.close()
    data = np.array(data)
    left = data[:,0:2]
    right = data[:,2:4]
    # return data
    return left, right

def find_ground_truth(a1, a2, path, method):
    # t1 = read_data("Left_TP.txt")
    # t2 = read_data("Right_TP.txt")
    t1, t2 = read_data(path)

    # print(t1.shape, t2.shape)

    F, mask = cv.findFundamentalMat(t1,t2,cv.FM_8POINT)
    # print(F)

    true_pos = []
    false_pos = []
    # for (p1, p2) in zip(a1, a2):
    for i in range(len(a1)):
        p1 = np.reshape(np.append(a1[i], 1), (3,1))
        p2 = np.reshape(np.append(a2[i], 1), (3,1))

        l = F.dot(p1)
        error = abs(p2.T.dot(l))/math.sqrt(l[0]**2+l[1]**2)
        if error < 2:
            true_pos.append(i)
        else:
            false_pos.append(i)
        # print(l)
        # print(math.sqrt(l[1]**2+l[2]**2))
        # print(error)

        # print(p2.T.dot(F.dot(p1)))
        # if abs(p2.T.dot(F.dot(p1))) < 0.01:
        #     true_pos.append(i)
        # else:
        #     false_pos.append(i)
    
    f = open("NEW_RESULTS.txt", "a")
    f.write(method + " " + path + "\n")
    f.write(str(len(true_pos)) + " true positive out of " + str(len(a1)) + "\n")
    f.write(str(len(false_pos)) + " false positive out of " + str(len(a1)) + "\n")
    f.write("\n \n")
    f.close()

    print(len(true_pos), "true positive out of", len(a1))
    print(len(false_pos), "false positive out of", len(a1))
    return true_pos, false_pos