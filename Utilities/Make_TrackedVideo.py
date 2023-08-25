import h5py
from pathlib import Path
import cv2
import numpy as np

ballpath = Path('/mnt/labserver/DURRIEU_Matthias/Experimental_data/MultiMazeRecorder/Videos/230721_Feedingstate_4_PM_Videos_Tracked/arena5/corridor3/corridor3_tracked.000_corridor3.analysis.h5')

flypath = Path('/mnt/labserver/DURRIEU_Matthias/Experimental_data/MultiMazeRecorder/Videos/230721_Feedingstate_4_PM_Videos_Tracked/arena5/corridor3/flytrack.000_corridor3.analysis.h5')

vidpath = Path('/mnt/labserver/DURRIEU_Matthias/Experimental_data/MultiMazeRecorder/Videos/230721_Feedingstate_4_PM_Videos/arena5/corridor3/corridor3.mp4')

outpath = Path('/mnt/labserver/DURRIEU_Matthias/Videos/ballflytrack_sample.mp4')

# import the h5 files

with h5py.File(ballpath.as_posix(), "r") as f:
            dset_names = list(f.keys())
            ball_locs = f["tracks"][:].T
            node_names = [n.decode() for n in f["node_names"][:]]
            
with h5py.File(flypath.as_posix(), "r") as f:
            dset_names = list(f.keys())
            fly_locs = f["tracks"][:].T
            node_names = [n.decode() for n in f["node_names"][:]]
            
yball = ball_locs[:, :, 1, :]
xball = ball_locs[:, :, 0, :]

yfly = fly_locs[:, :, 1, :]
xfly = fly_locs[:, :, 0, :]

def create_marked_video(video_path, xfly, yfly, xball, yball, start_frame, end_frame, output_path):
    # Open the input video file
    cap = cv2.VideoCapture(video_path)
    
    # Get the video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Create the output video file
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Set the position of the video to the start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Create a window to display the frames
    #cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
    
    # Process each frame of the video
    for i in range(start_frame, end_frame+1):
        # Read the next frame from the input video
        ret, frame = cap.read()
        
        # Check if the frame was successfully read
        if not ret:
            break
        
        # Get the positions of the fly and ball in this frame
        fly_pos = (int(xfly[i]), int(yfly[i]))
        ball_pos = (int(xball[i]), int(yball[i]))
        
        # Draw a circle at each position
        cv2.circle(frame, fly_pos, 10, (0, 0, 255), -1)
        cv2.circle(frame, ball_pos, 10, (255, 0, 0), -1)
        
        # Write the frame to the output video
        out.write(frame)
        
        # Display the frame in the window
        #cv2.imshow('Frame', frame)
        
        # Wait for 1 millisecond to update the window
        cv2.waitKey(1)
    
    # Release the video capture and writer
    cap.release()
    out.release()
    
    # Destroy the window
    #cv2.destroyAllWindows()

create_marked_video(vidpath.as_posix(), xfly, yfly, xball, yball, 2000, 3200, outpath.as_posix())