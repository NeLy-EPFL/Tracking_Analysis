from pathlib import Path

data_path = "/mnt/labserver/DURRIEU_Matthias/Experimental_data/MultiMazeRecorder/Videos/"  # Update this with the path to your data

for main_folder in Path(data_path).iterdir():
    if not main_folder.is_dir():
        continue
    print(f"Checking structure of folder: {main_folder.name}")
    # Check if the main folder contains multiple videos
    video_count = len(list(main_folder.glob('*.mp4')))
    if video_count > 1:
        print('Multiple videos found in main folder. Moving videos to their respective folders...')
        # Create arena and corridor folders
        for arena in range(1, 10):
            arena_folder = main_folder / f'arena{arena}'
            arena_folder.mkdir(exist_ok=True)
            for corridor in range(1, 7):
                corridor_folder = arena_folder / f'corridor{corridor}'
                corridor_folder.mkdir(exist_ok=True)

        # Move videos to their respective folders
        for video in main_folder.glob('*.mp4'):
            video_name = video.name
            arena, corridor = video_name.split('.')[0].split('_')[0], f'corridor_{video_name.split(".")[0].split("_")[2]}'
            corridor = corridor.replace('_', '')
            destination_folder = main_folder / arena / corridor
            video.rename(destination_folder / video_name)
        
        print(f'Reordering of videos in {main_folder.name} complete.')

