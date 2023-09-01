import cv2
import h5py

import GetLabels

print(cv2.__version__)

startFrame = 0   # only variables you need to change to run the code
endFrame = 5000     # start and end frame of the abstract that you want to record

# raise an error if the startFrame is greater than the endFrame
if startFrame > endFrame:
    raise ValueError('The startFrame is greater than the endFrame')


videoCapture = cv2.VideoCapture("/Volumes/Ramdya-Lab/DURRIEU_Matthias/Experimental_data/Optogenetics/Optobot/MultiMazeBiS_15_Steel_Wax/Female_Starved_noWater/221116/102044_s0a0_p6-0/MultiMazeBiS_15_Steel_Wax_Female_Starved_noWater_p6-0_80fps_Trimmed_smol_rotated.mp4") #TODO change the pathname
# get the frame rate of the video, needed for the video writer
fps = videoCapture.get(cv2.CAP_PROP_FPS)
print(fps)
# get the frame size of the video, needed for the video writer
frameSize = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print(frameSize)

# create a videowriter object for mp4
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 40.0, (832, 832))

frameNumber = 0
while True:
    ret, frame = videoCapture.read()
    if not ret:
        break

    frameNumber += 1
    print(frameNumber)

    if startFrame <= frameNumber <= endFrame:

        # get the skeleton of the frame
        GetLabels.draw_entire_skeleton(frame, frameNumber)
        out.write(frame)
        cv2.imshow("frame", frame)

    if frameNumber > endFrame:
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  # to quit the video press q

videoCapture.release()
out.release()
cv2.destroyAllWindows()