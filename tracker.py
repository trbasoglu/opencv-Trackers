import cv2
import sys
import numpy as np

from datetime import datetime
import matplotlib as plt

def showFlow(prev_frame,frame,hsv):
    hsv = np.zeros_like(frame)
    hsv[..., 1] = 255
    next = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_frame, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imshow('frame2', bgr)
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

if __name__ == '__main__':

    # Set up tracker.
    # Instead of MIL, you can also use

    tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'MOSSE', 'CSRT']
    #Mosse,boostingperformans yetersiz. bulamiyor
    #medianflow yetersiz
    #CSRT>TLD>MIL>KCF>BOOSTING,Mean Shift,CamShift,Particle Filter,Kalman Filter,optical Flow
    #tld+ meanshift+,mil+,camshift+,Kalman Filter+,KCF+,boosting+Optical flow+,particle filter
    #csrt-
    #BOOSTING,MIL size not changed
    tracker_type = tracker_types[6]

    if int(minor_ver) < 3:
        tracker = cv2.Tracker_create(tracker_type)
    else:
        if tracker_type == 'BOOSTING':
            tracker = cv2.TrackerBoosting_create()
        if tracker_type == 'MIL':
            tracker = cv2.TrackerMIL_create()
        if tracker_type == 'KCF':
            tracker = cv2.TrackerTLD_create()
        if tracker_type == 'TLD':
            tracker = cv2.TrackerTLD_create()
        if tracker_type == 'MEDIANFLOW':
            tracker = cv2.TrackerMedianFlow_create()
        if tracker_type == 'GOTURN':
            tracker = cv2.TrackerGOTURN_create()
        if tracker_type == 'MOSSE':
            tracker = cv2.TrackerMOSSE_create()
        if tracker_type == "CSRT":
            tracker = cv2.TrackerCSRT_create()

    # Read video
    Path='D:\\cv\\videoplayback.mp4'
    video = cv2.VideoCapture(Path)#'C:\\Users\\trbasoglu\\Documents\\PycharmProjects\\untitled1\\cv\\video1.mp4')

    # Exit if video not opened.
    if not video.isOpened():
        print
        "Could not open video"
        sys.exit()

    # Read first frame.
    ok, frame = video.read()
    if not ok:
        print
        'Cannot read video file'
        sys.exit()

    # Define an initial bounding box
    bbox = (287, 23, 86, 320)

    # Uncomment the line below to select a different bounding box
    print("line")
    bbox = cv2.selectROI('Tracking',frame, True)
    print(bbox,"_____",type(bbox))
    # Initialize tracker with first frame and bounding box
    ok = tracker.init(frame, bbox)
    screenX = np.shape(frame)[1]
    screenY = np.shape(frame)[0]
    print(screenX, "x", screenY)
    centerX =screenX/2
    centerY=screenY/2
    print(centerX, "x", centerY)
    array=[]
    sum=np.sum(bbox)
    i=0
    starttime=cv2.getTickCount()
    prev_frame=frame
    hsv = np.zeros_like(frame[int(bbox[1])-50:int(bbox[1]+bbox[3])+50, int(bbox[0])-50:int(bbox[0]+bbox[2])+50])
    prev_trackWindow=frame[int(bbox[1]):int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2])]
    initialDiagonal=np.sqrt(np.square(np.shape(prev_trackWindow)[0])+np.square(np.shape(prev_trackWindow)[1]))
    avgdiagonal=0
    i=0
    while True:
        # Read a new frame
        i += 1
        print("atlanmadi"+str(i))
        ok, frame = video.read()

        if not ok:
            break

        prvs=prev_frame[int(bbox[1])-50:int(bbox[1]+bbox[3])+50, int(bbox[0])-50:int(bbox[0]+bbox[2])+50]
        next =frame[int(bbox[1])-50:int(bbox[1] + bbox[3])+50, int(bbox[0])-50:int(bbox[0] + bbox[2])+50]
        # print(np.shape(next ))
        # nameOfImage="img_" + str(datetime.now().strftime('%d%m%y%H%M'))+'_'+str(i)+'.jpg'
        # cv2.imwrite(nameOfImage, next)
        # showFlow(cv2.cvtColor(prvs, cv2.COLOR_BGR2GRAY), next,hsv)
        prev_trackWindow=frame[int(bbox[1]):int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2])]
        prevdiagonal=np.sqrt(np.square(np.shape(prev_trackWindow)[0])+np.square(np.shape(prev_trackWindow)[1]))

        # Start timer
        timer = cv2.getTickCount()

        # Update tracker
        ok, bbox = tracker.update(frame)
        print(bbox)
        next_trackwindow = frame[int(bbox[1]):int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2])]
        nextdiagonal = np.sqrt(np.square(np.shape(next_trackwindow)[0]) + np.square(np.shape(next_trackwindow)[1]))
        # if((i%24)!=0):
        avgdiagonal=(avgdiagonal*((i)-1)+nextdiagonal)/(i)
        # else:
        #     avgdiagonal = nextdiagonal
        print((i%24))
        print("prev:" + str(prevdiagonal))
        print("next:"+str(nextdiagonal))
        print("avg:" + str(avgdiagonal))
        if (nextdiagonal >initialDiagonal):
            cv2.putText(frame, " Closing: " + str(nextdiagonal / initialDiagonal), (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (80, 170, 50), 2)
        elif (nextdiagonal < initialDiagonal):
            cv2.putText(frame, " Moving Away: " + str(nextdiagonal / initialDiagonal), (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (80, 170, 50), 2)
        else:
            cv2.putText(frame, "Same", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (80, 170, 50), 2)
        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
        time= (cv2.getTickCount()-starttime)/ cv2.getTickFrequency()
        # Draw bounding box
        if ok:
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))

            bbox_center_x=(p1[0]+p2[0])/2
            bbox_center_y=(p1[1]+p2[1])/2
            vectorX=centerX-bbox_center_x
            vectorY=centerY-bbox_center_y
            # if (nextdiagonal > initialDiagonal):
            #     cv2.putText(frame, "Closing", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 170, 50), 1)
            # elif (nextdiagonal == initialDiagonal):
            #     cv2.putText(frame, "Same", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 170, 50), 1)
            # else:
            #     cv2.putText(frame, "Move away", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 170, 50), 1)
            cv2.putText(frame, "vector : X:" + str(vectorX)+"Y:"+str(vectorY), (50,180), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (80, 170, 50), 2);
            cv2.putText(frame, "Position : p1:" + str(p1) + "p2:" + str(p2)+"center:"+str(bbox_center_x)+","+str(bbox_center_y), (50, 200), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 170, 150), 1);
            cv2.rectangle(frame, p1, p2, (0, 255, 0), 2, 1)
            cv2.line(frame, (int(bbox_center_x), int(bbox_center_y)), (int(centerX), int(centerY)), (255, 0, 0), 1)
            array.append(np.sum(bbox))
            #print(sum,",",np.sum(bbox),",",np.min(array),",",np.max(array))
            k = cv2.waitKey(1) & 0xff
            if(k==32):
                cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255),2)
                bbox = cv2.selectROI('Tracking',frame, True)

                # Initialize tracker with first frame and bounding box
                if int(minor_ver) < 3:
                    tracker = cv2.Tracker_create(tracker_type)
                else:
                    if tracker_type == 'BOOSTING':
                        tracker = cv2.TrackerBoosting_create()
                    if tracker_type == 'MIL':
                        tracker = cv2.TrackerMIL_create()
                    if tracker_type == 'KCF':
                        tracker = cv2.TrackerTLD_create()
                    if tracker_type == 'TLD':
                        tracker = cv2.TrackerTLD_create()
                    if tracker_type == 'MEDIANFLOW':
                        tracker = cv2.TrackerMedianFlow_create()
                    if tracker_type == 'GOTURN':
                        tracker = cv2.TrackerGOTURN_create()
                    if tracker_type == 'MOSSE':
                        tracker = cv2.TrackerMOSSE_create()
                    if tracker_type == "CSRT":
                        tracker = cv2.TrackerCSRT_create()
                ok = tracker.init(frame, bbox)
                screenX = np.shape(frame)[1]
                screenY = np.shape(frame)[0]
                print(screenX, "x", screenY)
                centerX = screenX / 2
                centerY = screenY / 2
                print(centerX, "x", centerY)
                array = []
                sum = np.sum(bbox)
                print(sum)
                starttime = cv2.getTickCount()
        else:
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            bbox = cv2.selectROI('Tracking',frame, True)

            # Initialize tracker with first frame and bounding box
            if int(minor_ver) < 3:
                tracker = cv2.Tracker_create(tracker_type)
            else:
                if tracker_type == 'BOOSTING':
                    tracker = cv2.TrackerBoosting_create()
                if tracker_type == 'MIL':
                    tracker = cv2.TrackerMIL_create()
                if tracker_type == 'KCF':
                    tracker = cv2.TrackerTLD_create()
                if tracker_type == 'TLD':
                    tracker = cv2.TrackerTLD_create()
                if tracker_type == 'MEDIANFLOW':
                    tracker = cv2.TrackerMedianFlow_create()
                if tracker_type == 'GOTURN':
                    tracker = cv2.TrackerGOTURN_create()
                if tracker_type == 'MOSSE':
                    tracker = cv2.TrackerMOSSE_create()
                if tracker_type == "CSRT":
                    tracker = cv2.TrackerCSRT_create()
            ok = tracker.init(frame, bbox)
            screenX = np.shape(frame)[1]
            screenY = np.shape(frame)[0]
            print(screenX, "x", screenY)
            centerX = screenX / 2
            centerY = screenY / 2
            print(centerX, "x", centerY)
            array = []
            sum = np.sum(bbox)
            print(sum)
        # Display tracker type on frame
        cv2.putText(frame, tracker_type + " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

        # Display FPS on frame
        cv2.putText(frame, "FPS : " + str(int(fps))+"frame: "+str(i)+"time:"+str(int(time)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

        # Display result
        cv2.imshow("Tracking", frame)

        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        print("time="+str(int(time)))
        prev_frame=frame
        if k == 27: break
