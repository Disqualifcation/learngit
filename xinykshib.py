import cv2
import numpy as np
#定义一个函数用于对模块的从左到右排序
def sort(cnts, method="left-to-right"):
    reverse = False
    i = 0
    if method == "right-to-left" or method=="bottom-to-top":
        reverse = True
    if method == "top-to-bottom" or method=="bottom-to-top":
        i = 1
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]#用一个最小的矩形，把找到的形状包起来，然后对最小坐标进行排序
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b : b[1][i], reverse=reverse))
    return cnts, boundingBoxes
#导入图片，转换为灰度图并二值化
img = cv2.imread("xinykmok.jpg")
ref = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ref = cv2.threshold(ref, 10, 255, cv2.THRESH_BINARY_INV)[1]
#轮廓查找并绘制
refCnts, hierarchy = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, refCnts, -1, (0, 0, 255), 3)
refCnts = sort(refCnts, method="left-to-right")[0]
digits = {}
# 计算外接矩形并且resize成合适大小
for (i, c) in enumerate(refCnts):
    (x, y, w, h) = cv2.boundingRect(c)
    roi = ref[y:y + h, x:x + w]
    roi = cv2.resize(roi, (57, 88)) # 每一个数字对应每一个模板
    digits[i] = roi
#核函数生成两个不同的核
rect = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
sq = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

image = cv2.imread("R-C.jpg")
cv2.imshow("image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
#将图片resize成合适大小
image = cv2.resize(image, (300, 200))
#进行灰度图处理并进行一次礼帽处理
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rect)
#sobel算子获取边缘边缘信息
gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
gradX = np.absolute(gradX)
(minVal, maxVal) = (np.min(gradX),np.max(gradX))
gradX = (255*((gradX-minVal)/(maxVal-minVal)))
gradX = gradX.astype("uint8")
#运用两次礼帽操作将数字轮廓间隙缩小，使轮廓几乎规则
gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE,rect)
thresh = cv2.threshold(gradX,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sq)
threshCnts, hierarchy=cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = threshCnts
cur_img = image.copy()
#轮廓绘制
cv2.drawContours(cur_img, cnts, -1, (0,0,255), 3)
locs = []
#for循环遍历，舍弃干扰模块
for (i, c) in enumerate(cnts):
    (x, y, w, h) = cv2.boundingRect(c)
    ar = w/float(h)
    if ar>2.5 and ar<4:
        if(w>40 and w<55) and (h>10 and h<20):
            locs.append((x,y,w,h))
locs = sorted(locs, key=lambda x: x[0])
output = []
#遍历循环，将数字模块圈出并比对模板数字，再将其书写在信用卡数字上方
for (i, (gX, gY, gW, gH)) in enumerate(locs):
    groupOutput = []
    group = gray[gY-5:gY+gH+5, gX-5:gX+gW+5]
    group = cv2.threshold(group, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    digitCnts, hierarchy = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    digitCnts = sort(digitCnts, method="left-to-right")[0]
    for c in digitCnts:
        (x, y, w, h) = cv2.boundingRect(c)
        roi = group[y:y + h, x:x + w]
        roi = cv2.resize(roi, (57, 88))
        scores = []
        for (digit, digitROI) in digits.items():
            result = cv2.matchTemplate(roi, digitROI, cv2.TM_CCOEFF)
            (_, score, _, _) = cv2.minMaxLoc(result)
            scores.append(score)
        groupOutput.append(str(np.argmax(scores)))
    cv2.rectangle(image, (gX-5, gY-5), (gX+gW+5, gY+gH+5), (0, 0, 255), 1)
    cv2.putText(image, "".join(groupOutput), (gX,gY-15), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
    output.extend(groupOutput)
cv2.imshow("image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()



