#对实时物体进行边缘检测··
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2

#定义两个变量，width用于将尺寸转换
pix = None
width = 2
#读取图像，转换图像为灰度图像，并高斯模糊
img = cv2.imread("wtcl.png")
cv2.imshow("img",img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
r = cv2.GaussianBlur(gray, (7, 7), 0)

#边缘检测
edged = cv2.Canny(r, 50, 100)
k = np.ones((5,5),dtype=np.uint8)
edged = cv2.morphologyEx(edged,cv2.MORPH_CLOSE,k)

#轮廓查找
ref_Cnts, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#计算最小外接矩形
for (i, c) in enumerate(ref_Cnts):
    #将大小过小的轮廓舍弃
    if cv2.contourArea(c) <300:
        continue
    rect = cv2.minAreaRect(c)
    points = cv2.boxPoints(rect)
    points =np.int0(points)
    points =perspective.order_points(points)
    cv2.drawContours(img, [points.astype("int")], -1, (0, 0, 255), 2)
    #对最小外接矩形的坐标计算
    (tl, tr, br, bl) = points
    #为计算尺寸运用公式先计算中点坐标
    (tltrX, tltrY) = (tl[0] + tr[0]) * 0.5, (tl[1] + tr[1]) * 0.5
    (blbrX, blbrY) = (bl[0] + br[0]) * 0.5, (bl[1] + br[1]) * 0.5
    (tlblX, tlblY) = (tl[0] + bl[0]) * 0.5, (tl[1] + bl[1]) * 0.5
    (trbrX, trbrY) = (tr[0] + br[0]) * 0.5, (tr[1] + br[1]) * 0.5

    # 分别计算中心点坐标
    cv2.circle(img, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
    cv2.circle(img, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
    cv2.circle(img, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
    cv2.circle(img, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

    # 计算两个中心点之间的欧氏距离，即图片距离
    dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
    dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

    # 初始化测量指标值
    if pix is None:
        pix = dB / width

    # 计算目标的实际大小（宽和高），用英尺来表示
    dimA = dA / pix
    dimB = dB / pix

    # 在图片中绘制结果
    cv2.putText(img, "{:.1f}in".format(dimB),
                (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 2)
    cv2.putText(img, "{:.1f}in".format(dimA),
                (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 2)

cv2.imshow("image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()