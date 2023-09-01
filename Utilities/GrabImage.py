import cv2
from pathlib import Path

def extract_frame(video_path, frame_index, output_path):
    """Extract a single frame from a video file and save it as a PNG image.
    Arguments:
        video_path: Path to the video file.
        frame_index: Index of the frame to extract.
        output_path: Path to save the output image.
        
        Returns:
            None        
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Set the position of the video to the desired frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    
    # Check if the frame_index is valid, i.e. if it is within the bounds of the video
    if frame_index < 0 or frame_index >= cap.get(cv2.CAP_PROP_FRAME_COUNT):
        print('Error: Invalid frame index')
        return

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

