import pandas as pd
import numpy as np
import ast
import holoviews as hv
import panel as pn
import colorcet
import bokeh

hv.extension("bokeh")

Data = pd.read_csv(
    "/Volumes/Ramdya-Lab/DURRIEU_Matthias/Experimental_data/MultiSensory_Project/GatedArenas/Results/DataSetAugust22.csv"
)
Data = Data[Data['Test Starvation'] == "Overnight no Water"]
Data = Data[Data["Training Starvation"] == "Not starved"]#.reset_index()
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
    "Rewarded",
    "Rewarded",
    "Punished",
    "Punished",
    "Empty",
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
    name="ThreshSlider", value=80, start=60, end=270, step=10,
    #width=300,
)

Melted = Melted.loc[Melted["Location"] == 'Corner']

Condis = list(Melted["Condition"].unique())
Condis.append("All")

Condition = pn.widgets.RadioButtonGroup(options=Condis,
                                        )
Objs = list(Melted["ObjectsReinforced"].unique())
Objs.append("All")

Object = pn.widgets.RadioButtonGroup(options=Objs,
                                     )


Locs = list(Melted["Location"].unique())
Locs.append("All")

Location = pn.widgets.Select(options=Locs)

Dates = list(Melted["Date"].unique())
Dates.insert(0, "All")

Date = pn.widgets.Select(options=Dates)


def slider_callback(Condition, ThreshSlider, Object, Date):

    if Condition == "All":
        Subset = Melted

    else:
        Subset = Melted.loc[(Melted["Condition"] == Condition)]

    if Date == "All":
        Subset = Subset
    else:
        Subset = Subset.loc[Subset["Date"] == Date]

    if Object == "All":
        Subset = Subset
    else:
        Subset = Subset.loc[Subset["ObjectsReinforced"] == Object]

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
        ylim=(0, 30),
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
        ylim=(0, 30),
        color="Training",
        jitter=0.4,
        size=5,
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
        Object=Object,
        #Location=Location,
        ThreshSlider=ThreshSlider,
    )
)

app = pn.Row(
    pn.WidgetBox(
        "# Gated Arenas Food objects peeking dashboard",
        "###Threshold slider",
        ThreshSlider,
        "###Object rewarded",
        Object,
        "###Focal Gate",
        Condition,
        "###Date of experiment",
        Date,
        #"###Location#",
        #Location,
    ),
    pn.Spacer(width=50),
    dmap.opts(
        width=600,
        height=600,
        framewise=True,
    ), sizing_mode='stretch_both',
).servable()

#app

app.save('/Volumes/Ramdya-Lab/DURRIEU_Matthias/Experimental_data/MultiSensory_Project/GatedArenas/Results/Boxplots_Peeking.html',
         embed=True,
         max_states=1000,
         max_opts=5000, # default : 3; used to limit options especially for slider like objects that can take lots of values
         #embed_json=True
        )
