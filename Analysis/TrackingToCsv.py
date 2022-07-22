import pandas as pd
import os

DataPath = "/Volumes/Ramdya-Lab/DURRIEU_Matthias/Experimental_data/MultiSensory_Project/GatedArenas_Agar"
Data = pd.DataFrame(
    columns=(
        "Date",
        "Training",
        "Starvation",
        "Relative Time Left",
        "Relative Time Right",
        "Relative Time far Left",
        "Relative Time far Right",
    )
)

for dirpath, dirnames, filenames in os.walk(DataPath):
    if "Results" in dirnames:
        dirnames.remove("Results")
    for filename in [f for f in filenames if f.endswith(".csv")]:
        print(dirpath)

        Out = []
        file_path = os.path.join(dirpath, filename)
        print(file_path)
        df = pd.read_csv(file_path)
        X = df["pos_x"]
        Y = df["pos_y"]
        Center_X = min(X) + ((max(X) - min(X)) / 2)
        Center_Y = min(Y) + ((max(Y) - min(Y)) / 2)
        Quarter_X = min(X) + ((Center_X - min(X)) / 2)
        TQuarter_X = Center_X + ((max(X) - Center_X) / 2)

        Left = len(X[X < Center_X])
        FarLeft = len(X[X < Quarter_X])
        Right = len(X[X > Center_X])
        FarRight = len(X[X > TQuarter_X])

        RelTimeLeft = Left / len(X)
        RelTimeFarLeft = FarLeft / len(X)
        RelTimeRight = Right / len(X)
        RelTimeFarRight = FarRight / len(X)

        VisitsLeft_gate = []

        timer = 0

        for ypos in df.pos_y[df.pos_x < 250]:
            if (ypos < 400 and ypos > 250) or (ypos < 600 and ypos > 475) in df.pos_y:
                timer += 1
                # print (timer)
            elif timer != 0:
                VisitsLeft_gate.append(timer)
                timer = 0

        VisitsLeft_gate_Front = []

        timer = 0

        for ypos in df.pos_y[(df.pos_x < 350) & df.pos_x > 200]:
            if (ypos < 500 and ypos > 300) in df.pos_y:
                timer += 1
                # print (timer)
            elif timer != 0:
                VisitsLeft_gate_Front.append(timer)
                timer = 0

        VisitsRight_gate = []

        timer = 0

        for ypos in df.pos_y[df.pos_x > 575]:
            if (ypos < 400 and ypos > 250) or (ypos < 600 and ypos > 475) in df.pos_y:
                timer += 1
                # print (timer)
            elif timer != 0:
                VisitsRight_gate.append(timer)
                timer = 0

        VisitsRight_gate_Front = []

        timer = 0

        for ypos in df.pos_y[(df.pos_x > 500) & (df.pos_x < 650)]:
            if (ypos < 500 and ypos > 300) in df.pos_y:
                timer += 1
                # print (timer)
            elif timer != 0:
                VisitsRight_gate_Front.append(timer)
                timer = 0

        VisitsTop_gate = []

        timer = 0

        for xpos in df.pos_x[df.pos_y < 275]:
            if (xpos < 350 and xpos > 250) or (xpos < 600 and xpos > 500) in df.pos_x:
                timer += 1
                # print (timer)
            elif timer != 0:
                VisitsTop_gate.append(timer)
                timer = 0

        VisitsTop_gate_Front = []

        timer = 0

        for xpos in df.pos_x[(df.pos_y < 350) & (df.pos_y > 200)]:
            if (xpos < 500 and xpos > 300) in df.pos_x:
                timer += 1
                # print (timer)
            elif timer != 0:
                VisitsTop_gate_Front.append(timer)
                timer = 0

        Peeks_Left = sum(1 for i in VisitsLeft_gate if i > 160)
        Peeks_Right = sum(1 for i in VisitsRight_gate if i > 160)
        Peeks_Top = sum(1 for i in VisitsTop_gate if i > 160)
        LongPeeks_Left = sum(1 for i in VisitsLeft_gate if i > 320)
        LongPeeks_Right = sum(1 for i in VisitsRight_gate if i > 320)
        LongPeeks_Top = sum(1 for i in VisitsTop_gate if i > 320)
        Face_Left = sum(1 for i in VisitsLeft_gate_Front if i > 160)
        Face_Right = sum(1 for i in VisitsRight_gate_Front if i > 160)
        Face_Top = sum(1 for i in VisitsTop_gate_Front if i > 160)

        Out.append(
            {
                "Date": "22-03-04" if ("220304" in dirpath) else "22-03-10",
                "Training": "Trained" if ("Trained" in dirpath) else "Ctrl",
                "Starvation": "Overnight no Water"
                if ("noWater" in dirpath)
                else "Overnight with Water",
                "Reinforced_side": "Right"
                if ("RightRew" in dirpath)
                else "Left"
                if ("LeftRew" in dirpath)
                else "Empty",
                "Relative Time Left": RelTimeLeft,
                "Relative Time Right": RelTimeRight,
                "Relative Time far Left": RelTimeFarLeft,
                "Relative Time far Right": RelTimeFarRight,
                "Peeks Left": Peeks_Left,
                "Peeks Right": Peeks_Right,
                "Peeks Top": Peeks_Top,
                "Face Left": Face_Left,
                "Face Right": Face_Right,
                "Face Top": Face_Top,
                "Long Peeks Left": LongPeeks_Left,
                "Long Peeks Right": LongPeeks_Right,
                "Long Peeks Top": LongPeeks_Top,
            }
        )
        Out = pd.DataFrame(Out)
        Data = Data.append(Out)

Data.to_csv(DataPath + "/Results/DataSet.csv")
