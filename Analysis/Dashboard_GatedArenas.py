import pandas as pd
import numpy as np
import ast

import bokeh.layouts
import bokeh.models
import bokeh.plotting

from bokeh.palettes import Category10
import itertools

colors = itertools.cycle(Category10[10])
def color_gen():
    yield from itertools.cycle(Category10[10])
color = color_gen()
# Read in data

TimeData = pd.read_csv(
    "/Volumes/Ramdya-Lab/DURRIEU_Matthias/Experimental_data/MultiSensory_Project/GatedArenas/Results/DataSetNew.csv"
)

# Prep dataset from data
TimeData = pd.read_csv(
    "/Volumes/Ramdya-Lab/DURRIEU_Matthias/Experimental_data/MultiSensory_Project/GatedArenas/Results/DataSetNew.csv"
)

# Prep dataset from data
TimeMelted = pd.melt(
    TimeData,
    id_vars=["Training", "ObjectsReinforced", "Reinforced_side", "Date", "Fly"],
    value_name="Values",
    value_vars=[
        "Visits Left Corner",
        "Durations Left Corner",
        "Visits Right Corner",
        "Durations Right Corner",
        "Visits Top Corner",
        "Durations Top Corner",
        "Visits Left Front",
        "Durations Left Front",
        "Visits Right Front",
        "Durations Right Front",
        "Visits Top Front",
        "Durations Top Front",
    ],
    var_name="Variable",
)

TimeMelted[["Values",]] = TimeMelted[
    [
        "Values",
    ]
].applymap(ast.literal_eval)
TimeMelted = TimeMelted.explode("Values")#.drop_duplicates()#.dropna()#.reset_index()
TimeMelted["EventIndex"] = TimeMelted.groupby(["Fly", "Variable"]).cumcount()
TimeMelted = TimeMelted.drop_duplicates().dropna()

conditions = [
    (TimeMelted["Variable"].str.contains("Durations")),
    (TimeMelted["Variable"].str.contains("Visits")),
]

values = ["Durations", "Visits"]
TimeMelted["Kind"] = np.select(conditions, values)

conditions = [
    (
        TimeMelted["Reinforced_side"].str.contains("Right")
        & (TimeMelted["Variable"].str.contains("Right")
    )
    ),
    (
        TimeMelted["Reinforced_side"].str.contains("Left")
        & (TimeMelted["Variable"].str.contains("Left")
    )
    ),
    (
        TimeMelted["Reinforced_side"].str.contains("Right")
        & (TimeMelted["Variable"].str.contains("Left")
    )
    ),
    (
        TimeMelted["Reinforced_side"].str.contains("Left")
        & (TimeMelted["Variable"].str.contains("Right")
     )
    ),
    (TimeMelted["Variable"].str.contains("Top")
     ),

]

values = [
    "Rewarded Side",
    "Rewarded Side",
    "Punished Side",
    "Punished Side",
    "Empty Side",
]
TimeMelted["Condition"] = np.select(conditions, values)

conditions = [
    (
            (TimeMelted["Variable"].str.contains("Corner"))
    ),
    (
            (TimeMelted["Variable"].str.contains("Front")))

]
values = [
    "Corner",
    "Front",
]
TimeMelted["Location"] = np.select(conditions, values)

TimeMelted = TimeMelted#.reset_index()

TimeMelted = (
    TimeMelted.reset_index()
    .pivot_table(
        index=["Condition","Fly","EventIndex","Training", "ObjectsReinforced", ],
        columns="Kind",
        values="Values",
    )
    .reset_index()
)

DataSet = TimeMelted

# Produce the subsets

ReinforcedVisit = DataSet[DataSet["Condition"] == "Rewarded Side"]

PunishedVisits = DataSet[DataSet["Condition"] == "Punished Side"]

EmptyVisits = DataSet[DataSet["Condition"] == "Empty Side"]

# Options for x and y selectors

Subset_selector = bokeh.models.Select(
    title="Subsets",
    options=TimeMelted["Condition"].unique().tolist() + ['All'],
    value="All",
    width=200
)

xy_options = list(
    DataSet.columns[ReinforcedVisit.columns.isin(["Visits", "Durations"])]
)  #'ObjectsReinforced','Training', 'Fly'

# Define the selector widgets

x_selector = bokeh.models.Select(
    title="x",
    options=xy_options,
    value="Visits",
    width=200,
)

y_selector = bokeh.models.Select(
    title="y",
    options=xy_options,
    value="Durations",
    width=200,
)

colorby_selector = bokeh.models.Select(
    title="color by",
    options=[
        "none",
        "ObjectsReinforced",
        "Training",
        "Fly",
    ],
    value="none",
    width=200,
)

# Column data source
source = bokeh.models.ColumnDataSource(
    dict(x=DataSet["Visits"],
         y=DataSet["Durations"],
         Fly=DataSet['Fly'],
         Training=DataSet['Training'],
         ObjectsReinforced=DataSet['ObjectsReinforced'],
         #Conditions=DataSet['Condition']

         )
)



# Add a column for colors; for now, all bokeh's default blue
source.data["color"] = ["#1f77b3"] * len(DataSet)

# Make the plot
p = bokeh.plotting.figure(
    frame_height=600,
    frame_width=600,
    x_axis_label=x_selector.value,
    y_axis_label=y_selector.value,
    tooltips=[('Fly', '@{Fly}'),
                  ('Training state', '@Training'),
                  ('Object reinforced', '@ObjectsReinforced')
                 ],
    #legend_group=colorby_selector.value
)

# Populate glyphs
circle = p.circle(source=source, x="x", y="y", color="color",
                  #legend="legend",
                   )


def gfmt_callback(attr, new, old):
    """Callback for updating plot of GMFT results."""
    # Update data subset

    if Subset_selector.value == "Rewarded Side":
        DataSet = TimeMelted[TimeMelted['Condition'] == 'Rewarded Side']

    elif Subset_selector.value == "Punished Side":
        DataSet = TimeMelted[TimeMelted['Condition'] == 'Punished Side']

    elif Subset_selector.value == "Empty Side":
        DataSet = TimeMelted[TimeMelted['Condition'] == 'Empty Side']

    elif Subset_selector.value == "All":
        DataSet = TimeMelted

    # Update color column
    if colorby_selector.value == "none":
        source.data["color"] = ["#1f77b3"] * len(DataSet)

    elif colorby_selector.value == "ObjectsReinforced":
        source.data["color"] = [
            "#1f77b3" if Obj == "Blue" else "#ff7e0e"
            for Obj in DataSet["ObjectsReinforced"]
        ]

    elif colorby_selector.value == "Training":
        source.data["color"] = [
            "#1f77b3" if Training == "Trained" else "#ff7e0e"
            for Training in DataSet["Training"]
        ]


    # Update x-data and axis label
    source.data["x"] = DataSet[x_selector.value]
    p.xaxis.axis_label = x_selector.value

    # Update x-data and axis label
    source.data["y"] = DataSet[y_selector.value]
    p.yaxis.axis_label = y_selector.value


# Connect selectors to callback
Subset_selector.on_change("value", gfmt_callback)
colorby_selector.on_change("value", gfmt_callback)
x_selector.on_change("value", gfmt_callback)
y_selector.on_change("value", gfmt_callback)

# Build the layout
gfmt_layout = bokeh.layouts.row(
    p,
    bokeh.layouts.Spacer(width=15),
    bokeh.layouts.column(
        Subset_selector,
        bokeh.layouts.Spacer(height=15),
        x_selector,
        bokeh.layouts.Spacer(height=15),
        y_selector,
        bokeh.layouts.Spacer(height=15),
        colorby_selector,
    ),
)


def gfmt_app(doc):
    doc.add_root(gfmt_layout)


# Build the app in the current doc
gfmt_app(bokeh.plotting.curdoc())

# To serve, execute : bokeh serve --show Dashboard_GatedArenas.py
