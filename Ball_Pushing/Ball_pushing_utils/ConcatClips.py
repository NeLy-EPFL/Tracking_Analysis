from moviepy.editor import concatenate_videoclips, TextClip, CompositeVideoClip, VideoFileClip
import os

def concatenate_clips(input_folder):
    clips = []
    for filename in os.listdir(input_folder):
        if filename.endswith(".mp4"):
            clip = VideoFileClip(os.path.join(input_folder, filename))
            txt_clip = TextClip(filename, fontsize=24, color='white')
            txt_clip = txt_clip.set_position('bottom').set_duration(clip.duration)
            video = CompositeVideoClip([clip, txt_clip])
            clips.append(video)
    final_clip = concatenate_videoclips(clips)
    final_clip.write_videofile("/mnt/labserver/DURRIEU_Matthias/Experimental_data/MultiMazeRecorder/Grids/Newcrop.mp4")

concatenate_clips("/mnt/labserver/DURRIEU_Matthias/Experimental_data/MultiMazeRecorder/Grids/Newcrop")




    

