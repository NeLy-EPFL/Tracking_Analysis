import cv2
print(cv2.__version__)

import numpy as np

videoCapture = cv2.VideoCapture("/Users/maximehilbig/Documents/EPFL/Année 3/Semester project/SLEAP/FirstVideoMatthias/Mp4Videos/multiMazeTrimmed1.mp4")
#videoCapture = cv2.VideoCapture("/Users/maximehilbig/Documents/EPFL/Année 3/Semester project/SLEAP/FirstVideoMatthias/Mp4Videos/multiMazeTrimmed1copy30sec.mp4")
prevCircle = None



frameList = []
while True:
    ret, frame = videoCapture.read()
    if not ret: break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurFrame = cv2.GaussianBlur(gray, (5, 5), 0)   #(5,5) is the kernel size, more or less blur

    circles = cv2.HoughCircles(blurFrame, cv2.HOUGH_GRADIENT, 1, 125, param1=40, param2=30, minRadius=10, maxRadius=30) #fourth argument distance between centers of circles
    #list of circles found are stored in circles                          param1 is sensitivity, if two high will detect not many circles, too low will detect too many circles
                                                                            #param2 is the minimum number of edges, the higher the more accurate the circle detection


    if circles is not None:
        circles = np.uint16(np.around(circles))

        #shape of circles
        print(circles.shape)




        iteration = 0
        listCircle = []
        for i in circles[0, :]:
            iteration += 1
            print("Coordinates of circle", iteration, ":", i[0], i[1], i[2])

            cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 3) #draw the outer circle
            cv2.circle(frame, (i[0], i[1]), 2, (255, 0, 255), 3) #draws the center of the circle
            #create a list of the coordinates of the circles and at it to the listFrame
            listCircle.append([i[0], i[1], i[2]])
        # flatten the list of circles
        listCircle = [item for sublist in listCircle for item in sublist]
        frameList.append(listCircle)


    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break    #to quit teh video press q

# export the frameList to a csv file with one row per frame, and one column per coordinate of the circle
import csv
with open('frameList.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(frameList)

videoCapture.release()
cv2.destroyAllWindows()