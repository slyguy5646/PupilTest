from pupil_apriltags import Detector, Detection
import cv2
import numpy as np
from numba import jit



    # hello = cv2.namedWindow("Image")

def main():    
    at = Detector(
        nthreads=100
    )

    cameraMatrix = (1386.5059636831022, 1380.3287074149298, 674.3138223396147, 431.0716867204328)


    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        #cv2.imshow('frame', frame)
        if not success:
            break

        tagDetection = at.detect(img=gray, estimate_tag_pose=True, camera_params=cameraMatrix, tag_size=0.060325)

        print(tagDetection)
            # cv2.imshow(hello, gray)
            # k = cv2.waitKey(1)

            # if k == 27:
            #     break
        

if __name__ == "__main__":
    main()

