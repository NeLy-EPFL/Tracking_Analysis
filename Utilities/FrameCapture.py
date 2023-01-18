# Program To Read video
# and Extract Frames

import cv2
from pathlib import Path

videopath = Path('/home/durrieu/Desktop/VideosForSibo/Video3/Arena5_trimmed.mp4')
# Function to extract frames


cap = cv2.VideoCapture(videopath.as_posix())
count = 0
last = 0

while True:
	ret, frame = cap.read() # Grab frame
	this = cap.get(1)
	if ret == True:

		#cv2.imshow('Arena6', frame)
		cv2.imwrite(videopath.parent.as_posix()+"/Video3_frames/frame%d.jpg" % count, frame)
		count += 1

		if cv2.waitKey(1) == 27:
			exit(0)
	if last >= this:
		break
	last = this
cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)
