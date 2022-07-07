import cv2
import sys

(major_v, minor_v, subminor_v) = (cv2.__version__).split('.')

if __name__ == '__main__':


#建立追踪器
    tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
    tracker_type = tracker_types[2]

    if int(minor_v) < 3:
        tracker = cv2.Tracker_create(tracker_type)
    else:
        if tracker_type == 'BOOSTING':
            tracker = cv2.TrackerBoosting_create()
        if tracker_type == 'MIL':
            tracker = cv2.TrackerMIL_create()
        if tracker_type == 'KCF':
            tracker = cv2.TrackerKCF_create()
        if tracker_type == 'TLD':
            tracker = cv2.TrackerTLD_create()
        if tracker_type == 'MEDIANFLOW':
            tracker = cv2.TrackerMedianFlow_create()
        if tracker_type == 'GOTURN':
            tracker = cv2.TrackerGOTURN_create()

    #设置变量video为视频
    video = cv2.VideoCapture("mbgz.mp4")


    # 读取视频
    aks, frame = video.read()
    if not aks:
        print('Cannot read video file')
        sys.exit()

    # 定义初始化边框
    box = (287, 40, 86, 320)

    #用于对目标的框选
    box = cv2.selectROI(frame, False)

    # 使用目标跟踪器跟踪框选的
    ok = tracker.init(frame, box)

    while True:
        # 读取框选时刻
        ok, frame = video.read()
        if not ok:
            break

        # 计算执行时间
        timer = cv2.getTickCount()

        # 追踪结果
        ok, bbox = tracker.update(frame)


        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        # 绘制边框
        if ok:
            # 追踪器运行，通过实时追踪rectangle绘制矩形边框
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
        else:
            #如果无边框输出则追踪器完成追踪
            cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        #
        cv2.putText(frame, tracker_type + " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)


        cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

        # 展示追踪结果
        cv2.imshow("Tracking", frame)

        # 按下esc键退出
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break