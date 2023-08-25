import cv2
from pathlib import Path

def extract_frame(video_path, frame_index, output_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Set the position of the video to the desired frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

    # Read the frame from the video
    ret, frame = cap.read()

    # Check if the frame was successfully read
    if ret:
        # Save the frame as an image
        cv2.imwrite(output_path, frame)
    else:
        print('Error: Could not read frame from video')

    # Release the video capture
    cap.release()


vidpath = Path('/mnt/labserver/DURRIEU_Matthias/Experimental_data/MultiMazeRecorder/Videos/230721_Feedingstate_4_PM_Videos_Tracked/arena5/corridor3/corridor3.mp4')
outpath = Path('/mnt/labserver/DURRIEU_Matthias/Pictures/ImageGrab.png')

extract_frame(vidpath.as_posix(), 4000, outpath.as_posix())

