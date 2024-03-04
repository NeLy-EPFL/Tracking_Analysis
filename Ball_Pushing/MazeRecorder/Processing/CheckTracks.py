from pathlib import Path

datapath = Path(
    "/mnt/labserver/DURRIEU_Matthias/Experimental_data/MultiMazeRecorder/Videos/"
)

for experiment in datapath.iterdir():
    if experiment.is_dir() and "_Checked" in experiment.name or "_Tracked" in experiment.name:
        all_corridors_tracked = True
        for arena in experiment.iterdir():
            if arena.is_dir():
                for corridor in arena.iterdir():
                    if corridor.is_dir():
                        slp_files_ball = list(corridor.glob("*_tracked_ball.slp"))
                        h5_files_ball = list(corridor.glob("*_tracked_ball*.h5"))
                        slp_files_fly = list(corridor.glob("*_tracked_fly.slp"))
                        h5_files_fly = list(corridor.glob("*_tracked_fly*.h5"))
                        if not (slp_files_ball and h5_files_ball and slp_files_fly and h5_files_fly):
                            all_corridors_tracked = False
                            print(f"Corridor {corridor.name} of arena {arena.name} in experiment {experiment.name} is missing tracking files.")
                            break
                if not all_corridors_tracked:
                    break
        if all_corridors_tracked:
            if "_Checked" in experiment.name:
                print(f"Experiment {experiment.name} is fully processed. Renaming...")
                new_name = experiment.name.replace("_Checked", "_Tracked")
                new_experiment_path = datapath / new_name
                experiment.rename(new_experiment_path)
            elif "_Tracked" in experiment.name:
                print(f"Experiment {experiment.name} is fully processed.")
