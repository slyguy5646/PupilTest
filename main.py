from pupil_apriltags import Detector, Detection
import cv2
import numpy as np

at = Detector(
    nthreads=20
)

cameraMatrix = (1386.5059636831022, 1380.3287074149298, 674.3138223396147, 431.0716867204328)


cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    if not success:
        break

    tagDetection = at.detect(img=gray, estimate_tag_pose=True, camera_params=cameraMatrix, tag_size=0.060325)

    print(tagDetection)

pose_R = np.array([[ 0.81650962, -0.10073156, -0.56847621],
 [ 0.00372407,  0.98555959, -0.16928802],
 [ 0.57731982,  0.13610825,  0.80509401]])

pose_T = np.array([[-0.67561307],
 [-0.1634916 ],
 [ 2.1544972 ]])