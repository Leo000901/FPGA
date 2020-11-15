import numpy as np
from cv2 import cv2



left_camera_matrix = np.array([[1104.0114, 0.5830, 580.4499],
                               [0., 1100.077, 481.0441],
                               [0., 0., 1.]])
left_distortion = np.array([[-0.4588,0.3514,-0.000108,0.0020,-0.2210]])



right_camera_matrix = np.array([[1114.528,3.5610,595.9592],
                                [0., 1114.098, 500.7778],
                                [0., 0., 1.]])
right_distortion = np.array([[-0.4983,0.5503,-0.0020,0.0026,-0.5252]])

R = np.array([[1.0000, -0.0024, 0.0073],[0.0024,1.0000,-0.0061],[-0.0073,0.0062,1.0000]]) # 旋转关系向量
#R = cv2.Rodrigues(om)[0]  # 使用Rodrigues变换将om变换为R
T = np.array([-114.6016, -1.9783, 4.3376]) # 平移关系向量

size = (640, 480) # 图像尺寸

# 进行立体更正
R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(left_camera_matrix, left_distortion,
                                                                  right_camera_matrix, right_distortion, size, R,
                                                                  T)
# 计算更正map
left_map1, left_map2 = cv2.initUndistortRectifyMap(left_camera_matrix, left_distortion, R1, P1, size, cv2.CV_16SC2)
right_map1, right_map2 = cv2.initUndistortRectifyMap(right_camera_matrix, right_distortion, R2, P2, size, cv2.CV_16SC2)


cv2.namedWindow("left")
cv2.namedWindow("right")
cv2.namedWindow("leftt")
cv2.namedWindow("rightt")
cv2.namedWindow("depth")
cv2.moveWindow("left", 0, 0)
cv2.moveWindow("right", 640, 0)
cv2.moveWindow("leftt", 0, 480)
cv2.moveWindow("rightt", 640, 480)
#cv2.createTrackbar("num", "depth", 0, 10, lambda x: None)
#cv2.createTrackbar("blockSize", "depth", 5, 255, lambda x: None)

videoIn = cv2.VideoCapture(1, cv2.CAP_DSHOW)
videoIn.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
videoIn.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
#cmera2 = cv2.VideoCapture(2)

def draw_min_rect(img,cnt,depth):  # conts = contours
    img = np.copy(img)
    x, y, w, h = cv2.boundingRect(cnt)
    #print(x)
    # x1 = x + w/2
    #print(y)
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)  # blue
    cv2.putText(img,str(round(depth,2)), (x+1,y+(h//2)),cv2.FONT_HERSHEY_COMPLEX_SMALL,0.7,(0,0,255),1)
    # x1 = x + w/2
    # y1 = y + h/2
    return img

#获取点集最大轮廓面积对应的集合
def getMaxAreaContour(cnts):
    if(len(cnts)) == 0:
        return
    maxArea = float()
    maxAreaIdx = 0
    for i in range(len(cnts)):
        temprea = cv2.contourArea(cnts[i])
        if temprea>maxArea:
            maxArea = temprea
            maxAreaIdx = i
    return cnts[maxAreaIdx]

def zuidaliantongyu(img1_rectified, img2_rectified, imgGrayL, imgGrayR, threeD):
    leftret, left = cv2.threshold(imgGrayL, 50, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE) #0是黑
    rightret, right = cv2.threshold(imgGrayR, 50, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)

    
    #left_cnts, hierarchy1 = cv2.findContours(left, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)    #轮廓检测
    #right_cnts, hierarchy2 = cv2.findContours(right, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #left_out = draw_min_rect(img1_rectified, getMaxAreaContour(left_cnts),0)#画出矩形并把矩形的坐标，宽高返回
    #right_out = draw_min_rect(img2_rectified, getMaxAreaContour(right_cnts),0)#画出矩形并把矩形的坐标，宽高返回
    
    # cv2.drawContours(left, left_cnts, -1, (0, 255, 255), 2)
    # cv2.drawContours(right, right_cnts, -1, (0, 255, 255), 2)
    # print(left_cnts)
    # print(right_cnts)
    # areal = []
    # arear = []
    # for i in range(len(left_cnts)):        #提取最大连通域，但是好像代码不太对
    #     areal.append(cv2.contourArea(left_cnts[i]))
    # print(areal)
    # if (len(areal) != 0):    
    #     max_idx = np.argmax(areal)
    # else:
    #     max_idx = 0
    # for i in range(max_idx - 1): 
    #     cv2.fillConvexPoly(left, left_cnts[max_idx], 0)   #这里会报错
    # #cv2.fillConvexPoly(left, left_cnts[max_idx], 255)

    # for i in range(len(right_cnts)):
    #     arear.append(cv2.contourArea(right_cnts[i]))
    # print(arear)
    # if (len(areal) != 0):    
    #     max_idx = np.argmax(areal)
    # else:
    #     max_idx = 0
    # for i in range(max_idx - 1):
    #     cv2.fillConvexPoly(right, right_cnts[max_idx], 0)
    #cv2.fillConvexPoly(right, right_cnts[max_idx], 255)
    
    left_cnts, hierarchy1 = cv2.findContours(left, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)    #轮廓检测
    right_cnts, hierarchy2 = cv2.findContours(right, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #if len(left_cnts) != 1:#轮廓总数

    cntl = getMaxAreaContour(left_cnts)
    x, y, w, h = cv2.boundingRect(cntl)
    x = round( x + w/2 )
    y = round( y + h/2 )
    # sum = 0
    # m = 0
    # for i in range(w):
    #     for j in range(h):
    #         if (x + i) < 480 and (y + j) < 480:
    #             d = threeD[x + i][y + j]
    #             if d[2] > 0:
    #                 sum = sum + d[2]
    #                 m = m + 1
    # if m == 0:
    #     m = 1
    # depth = sum / m
    if x < 480 and y < 480:
        d = threeD[x][y]
        depth = d[2]
    else:
        depth = 0
    #depth = 0
    left_out = draw_min_rect(img1_rectified, getMaxAreaContour(left_cnts), depth)#画出矩形并把矩形的坐标，宽高返回
    right_out = draw_min_rect(img2_rectified, getMaxAreaContour(right_cnts), depth)#画出矩形并把矩形的坐标，宽高返回
    return left_out, right_out, left, right

# def callbackFunc(e, x, y, f, p):
#     if e == cv2.EVENT_LBUTTONDOWN:        
#         print(threeD[y][x])

# cv2.setMouseCallback("depth", callbackFunc, None)

while True:
    ret, frame_vga = videoIn.read()
    #ret2, frame2 = camera2.read()
    frame1 = frame_vga
    frame2 = frame_vga

    if not ret:
        break

    # 根据更正map对图片进行重构
    img1_rectified = cv2.remap(frame1, left_map1, left_map2, cv2.INTER_LINEAR)
    img2_rectified = cv2.remap(frame2, right_map1, right_map2, cv2.INTER_LINEAR)

    # 将图片置为灰度图，为StereoBM作准备
    imgL = cv2.cvtColor(img1_rectified, cv2.COLOR_BGR2GRAY)
    imgR = cv2.cvtColor(img2_rectified, cv2.COLOR_BGR2GRAY)
    #imgGrayL = cv2.equalizeHist(imgL)
    #imgGrayR = cv2.equalizeHist(imgR)
 
# through gausiann filter
    imgGrayL = cv2.GaussianBlur(imgL, (5, 5), 0)  #高斯滤波
    imgGrayR = cv2.GaussianBlur(imgR, (5, 5), 0)


    # 两个trackbar用来调节不同的参数查看效果
    # num = cv2.getTrackbarPos("num", "depth")
    # blockSize = cv2.getTrackbarPos("blockSize", "depth")
    # if blockSize % 2 == 0:
    #     blockSize += 1
    # if blockSize < 5:
    #     blockSize = 5
    
    # 根据Block Maching方法生成差异图（opencv里也提供了SGBM/Semi-Global Block Matching算法，有兴趣可以试试）
    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=25)
    disparity = stereo.compute(imgGrayL, imgGrayR)
    
    disp = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    disp = cv2.GaussianBlur(disp, (5, 5), 0)

    # 将图片扩展至3d空间中，其z方向的值则为当前的距离
    #threeD = cv2.reprojectImageTo3D(dispp.astype(np.float32)/16., Q)
    threeD = cv2.reprojectImageTo3D(disparity.astype(np.float32)/16., Q)
    #threeD = array(threeD, dtype=np.float32)
    print(threeD)
    #print(p[2])
    # print(len(threeD))
    #print(threeD[479][479])
    # print(threeD[3][1])
    left_out, right_out, left, right = zuidaliantongyu(img1_rectified, img2_rectified, imgGrayL, imgGrayR, threeD)
    cv2.imshow("left", left)
    cv2.imshow("right", right)
    cv2.imshow("leftt", left_out)
    cv2.imshow("rightt", right_out)
    cv2.imshow("depth", disp)


    key = cv2.waitKey(1)
    if key == ord("q"):
        break
    elif key == ord("s"):
        cv2.imwrite("./snapshot/BM_left.jpg", imgL)
        cv2.imwrite("./snapshot/BM_right.jpg", imgR)
        cv2.imwrite("./snapshot/BM_depth.jpg", disp)


videoIn.release()
#camera2.release()
cv2.destroyAllWindows()