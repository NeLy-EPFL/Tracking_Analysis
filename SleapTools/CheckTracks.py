from pathlib import Path

datapath = Path(
    "/mnt/labserver/DURRIEU_Matthias/Experimental_data/MultiMazeRecorder/Videos/"
)

for experiment in datapath.iterdir():
    if experiment.is_dir():
        all_corridors_tracked = True
        for arena in experiment.iterdir():
            if arena.is_dir():
                for corridor in arena.iterdir():
                    if corridor.is_dir():
                        slp_files = list(corridor.glob("*.slp"))
                        h5_files = list(corridor.glob("*.h5"))
                        if not (slp_files and h5_files):
                            all_corridors_tracked = False
                            break
                if not all_corridors_tracked:
                    break
        if all_corridors_tracked:
            print(f"Experiment {experiment.name} is fully processed. Renaming...")
            new_name = experiment.name.replace("_Checked", "_Tracked")
            new_experiment_path = datapath / new_name
            experiment.rename(new_experiment_path)