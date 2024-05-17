import numpy as np
import math



numberofBalls = 6    #number of balls in the video

#read data from a csv file
import csv
with open('/Users/maximehilbig/PycharmProjects/openCVPlayground/frameList.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    frameList = []
    for row in reader:
        #take element wise and not row wise
        frameList.append([int(i) for i in row[0].split(',')])



print(frameList)
inputIwantToWorkon = []
#get the number of elements in frameList
numberOfFrames = len(frameList)

for i in range(numberOfFrames):
    listCircle = frameList[i]
    #regroup the coordinates of a circle
    listCircle = [listCircle[j:j + 3] for j in range(0, len(listCircle), 3)]
    inputIwantToWorkon.append(listCircle)


#now if I want to access the coordinates of frame i, I write inputIwantToWorkon[i] and I can write len(inputIwantToWorkon[i]) to get the number of balls detected in frame i

#determine the first frame where all the balls are detected
firstFrameWithRightNumberOfBalls = 0
for n in range(0, numberOfFrames):
    if (len(inputIwantToWorkon[n]) == numberofBalls):
        firstFrameWithRightNumberOfBalls = n
        break
print(firstFrameWithRightNumberOfBalls)

for frameNumber in range(firstFrameWithRightNumberOfBalls, numberOfFrames):

    #if(frameNumber == 6220)    to debug
        #print("here")
    numberOfBallsDetectedOnTheFrame = len(inputIwantToWorkon[frameNumber]) #for each frame I determine how many balls were detected
    if (numberOfBallsDetectedOnTheFrame > numberofBalls): #in this case too many balls were detected, we have to remove the wrong predictions
        numberofBallsAlreadyDeleted = 0
        numberofBallsIneedToDelete = numberOfBallsDetectedOnTheFrame - numberofBalls


        for i in range(0, numberOfBallsDetectedOnTheFrame):
            erase = True
            i = i - numberofBallsAlreadyDeleted #because we delete elements from the list, the index of the elements change, if I dont do this, outofBound error
            for j in range(0, 6):
                euclidianDistance = math.dist([inputIwantToWorkon[frameNumber][i][0], inputIwantToWorkon[frameNumber][i][1]], [inputIwantToWorkon[frameNumber-1][j][0], inputIwantToWorkon[frameNumber-1][j][1]])
                if euclidianDistance < 12:     #we compare the distance between the center of the circle of the current frame and the center of the circle of the previous frame, if a frame is too far from all the previous frames, it is likely to be a wrong prediction
                    erase = False
            if erase:
                inputIwantToWorkon[frameNumber].remove(inputIwantToWorkon[frameNumber][i]) #we remove the wrong prediction(s) so that we have the right number of balls
                numberofBallsAlreadyDeleted += 1
                if (numberofBallsAlreadyDeleted == numberofBallsIneedToDelete):
                    break


    elif (numberOfBallsDetectedOnTheFrame < numberofBalls): #in this case too few balls were detected, we have to add the missing balls
        numberofBallsAlreadyAdded = 0
        numberofBallsIneedToAdd = numberofBalls - numberOfBallsDetectedOnTheFrame
        for i in range(0, 6):
            add = True
            for j in range(0, numberOfBallsDetectedOnTheFrame):
                euclidianDistance = math.dist([inputIwantToWorkon[frameNumber][j][0],inputIwantToWorkon[frameNumber][j][1]], [inputIwantToWorkon[frameNumber-1][i][0], inputIwantToWorkon[frameNumber-1][i][1]]) #we compare the distance between the center of the circle of the current frame and the center of the circle of the previous frame, if a frame is too far from all the previous frames, it is likely that this is the ball we are looking for
                if euclidianDistance < 13: #12 was not big enough
                    add = False
            if add:
                inputIwantToWorkon[frameNumber].append(inputIwantToWorkon[frameNumber-1][i])
                numberofBallsAlreadyAdded += 1
                if(numberofBallsAlreadyAdded == numberofBallsIneedToAdd):
                    break
    print(frameNumber)

#export the created matrix to a csv file
import csv
with open('PostAnalysisFrameList.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(inputIwantToWorkon)



