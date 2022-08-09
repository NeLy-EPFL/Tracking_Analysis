import pandas as pd
import numpy as np
import ast
import holoviews as hv
import panel as pn
import colorcet

hv.extension("bokeh")

Data = pd.read_csv(
    "/Volumes/Ramdya-Lab/DURRIEU_Matthias/Experimental_data/MultiSensory_Project/GatedArenas/Results/DataSetNew.csv"
)

# Prep dataset from data
Melted = pd.melt(
    Data,
    id_vars=["Training", "ObjectsReinforced", "Reinforced_side", "Date", "Fly"],
    value_name="Durations",
    value_vars=[
        "Durations Left Corner",
        "Durations Right Corner",
        "Durations Top Corner",
        "Durations Left Front",
        "Durations Right Front",
        "Durations Top Front",
    ],
    var_name="Variable",
)

conditions = [
    (
        Melted["Reinforced_side"].str.contains("Right")
        & (Melted["Variable"].str.contains("Right"))
    ),
    (
        Melted["Reinforced_side"].str.contains("Left")
        & (Melted["Variable"].str.contains("Left"))
    ),
    (
        Melted["Reinforced_side"].str.contains("Right")
        & (Melted["Variable"].str.contains("Left"))
    ),
    (
        Melted["Reinforced_side"].str.contains("Left")
        & (Melted["Variable"].str.contains("Right"))
    ),
    (Melted["Variable"].str.contains("Top")),
]

values = [
    "Rewarded Side",
    "Rewarded Side",
    "Punished Side",
    "Punished Side",
    "Empty Side",
]
Melted["Condition"] = np.select(conditions, values)

conditions = [
    ((Melted["Variable"].str.contains("Corner"))),
    ((Melted["Variable"].str.contains("Front"))),
]
values = [
    "Corner",
    "Front",
]
Melted["Location"] = np.select(conditions, values)

ThreshSlider = pn.widgets.IntSlider(
    name="ThreshSlider", value=80, start=60, end=270, step=10
)

Condis = list(Melted["Condition"].unique())
Condis.append("All")

Condition = pn.widgets.RadioButtonGroup(options=Condis)

Locs = list(Melted["Location"].unique())
Locs.append("All")

Location = pn.widgets.Select(options=Locs)

Dates = list(Melted["Date"].unique())
Dates.insert(0, "All")

Date = pn.widgets.Select(options=Dates)


def slider_callback(Date, Condition, Location, ThreshSlider):
    if (Condition == "All") & (Location == "All"):
        Subset = Melted

    elif (Condition == "All") & (Location != "All"):
        Subset = Melted[(Melted["Location"] == Location)]

    elif (Condition != "All") & (Location == "All"):
        Subset = Melted[(Melted["Condition"] == Condition)]
    elif (Condition != "All") & (Location != "All"):
        Subset = Melted[
            (Melted["Condition"] == Condition) & (Melted["Location"] == Location)
        ]

    if Date == "All":
        Subset = Subset
    else:
        Subset = Subset[Subset["Date"] == Date]

    for index, row in Subset.iterrows():
        # print(row['Durations Left Corner'])

        # print (1 for i in row['Durations Left Corner'])
        Subset.loc[index, "Peeks"] = sum(
            1 for i in ast.literal_eval(row["Durations"]) if i > ThreshSlider
        )
        # Data_noWater_Simple['Peeks Left'][rows]= sum(1 for i in Data['Durations Left Corner'][rows] if i > param)
    # print(Data['Peeks Left'])
    box = hv.BoxWhisker(data=Subset, kdims=["Training"], vdims=["Peeks"]).opts(
        framewise=True,
        ylim=(0, 40),
        box_fill_alpha=0,
        invert_axes=True,
        invert_yaxis=True,
        fontscale=2,
        # box_line_color="gray",
    )
    points = hv.Scatter(data=Subset, kdims=["Training"], vdims=["Peeks"]).opts(
        framewise=True,
        cmap=colorcet.b_glasbey_category10,
        invert_axes=True,
        invert_yaxis=True,
        ylim=(0, 40),
        color="Training",
        jitter=0.4,
        size=6,
        alpha=0.5,
        fontscale=2,
        tools=["hover"],
    )

    return box * points


dmap = hv.DynamicMap(
    pn.bind(
        slider_callback,
        Date=Date,
        Condition=Condition,
        Location=Location,
        ThreshSlider=ThreshSlider,
    )
)

app = pn.Row(
    pn.WidgetBox(
        "# Gated Arena peeking dashboard",
        "##Threshold slider",
        ThreshSlider,
        "###Date of experiment",
        Date,
        "###Focal Gate",
        Condition,
        "###Location#",
        Location,
    ),
    pn.Spacer(width=100),
    dmap.opts(
        width=600,
        height=600,
        framewise=True,
    ),
).servable()

app
