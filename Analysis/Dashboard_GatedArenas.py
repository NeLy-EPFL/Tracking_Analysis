import pandas as pd
import numpy as np
import ast

import bokeh.layouts
import bokeh.models
import bokeh.plotting

# Read in data
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
TimeMelted = TimeMelted.explode("Values").drop_duplicates()

conditions = [
    (TimeMelted["Variable"].str.contains("Durations")),
    (TimeMelted["Variable"].str.contains("Visits")),
]

values = ["Durations", "Visits"]
TimeMelted["Kind"] = np.select(conditions, values)

conditions = [
    (
        TimeMelted["Reinforced_side"].str.contains("Right")
        & (TimeMelted["Variable"].str.contains("Right"))
    ),
    (
        TimeMelted["Reinforced_side"].str.contains("Left")
        & (TimeMelted["Variable"].str.contains("Left"))
    ),
    (
        TimeMelted["Reinforced_side"].str.contains("Right")
        & (TimeMelted["Variable"].str.contains("Left"))
    ),
    (
        TimeMelted["Reinforced_side"].str.contains("Left")
        & (TimeMelted["Variable"].str.contains("Right"))
    ),
    (TimeMelted["Variable"].str.contains("Top")),
]

values = [
    "Rewarded Side",
    "Rewarded Side",
    "Punished Side",
    "Punished Side",
    "Empty Side",
]
TimeMelted["Condition"] = np.select(conditions, values)

# Produce the subsets

ReinforcedVisit = TimeMelted[TimeMelted["Condition"] == "Rewarded Side"]

ReinforcedVisit.loc[
    ReinforcedVisit["Variable"].str.contains("Visits"), "Variable"
] = "Visits"
ReinforcedVisit.loc[
    ReinforcedVisit["Variable"].str.contains("Durations"), "Variable"
] = "Durations"

ReinforcedVisit = (
    ReinforcedVisit.reset_index()
    .pivot_table(
        index=["Fly", "Training", "ObjectsReinforced"],
        columns="Variable",
        values="Values",
    )
    .reset_index()
)

PunishedVisits = TimeMelted[TimeMelted["Condition"] == "Punished Side"]

PunishedVisits.loc[
    PunishedVisits["Variable"].str.contains("Visits"), "Variable"
] = "Visits"
PunishedVisits.loc[
    PunishedVisits["Variable"].str.contains("Durations"), "Variable"
] = "Durations"

PunishedVisits = (
    PunishedVisits.reset_index()
    .pivot_table(
        index=["Fly", "Training", "ObjectsReinforced"],
        columns="Variable",
        values="Values",
    )
    .reset_index()
)

EmptyVisits = TimeMelted[TimeMelted["Condition"] == "Empty Side"]

EmptyVisits.loc[EmptyVisits["Variable"].str.contains("Visits"), "Variable"] = "Visits"
EmptyVisits.loc[
    EmptyVisits["Variable"].str.contains("Durations"), "Variable"
] = "Durations"

EmptyVisits = (
    EmptyVisits.reset_index()
    .pivot_table(
        index=["Fly", "Training", "ObjectsReinforced"],
        columns="Variable",
        values="Values",
    )
    .reset_index()
)

# Options for x and y selectors

xy_options = list(
    ReinforcedVisit.columns[ReinforcedVisit.columns.isin(["Visits", "Durations"])]
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
    dict(x=ReinforcedVisit[x_selector.value], y=ReinforcedVisit[y_selector.value])
)

# Add a column for colors; for now, all bokeh's default blue
source.data["color"] = ["#1f77b3"] * len(ReinforcedVisit)

# Make the plot
p = bokeh.plotting.figure(
    frame_height=250,
    frame_width=250,
    x_axis_label=x_selector.value,
    y_axis_label=y_selector.value,
    legend_group=colorby_selector.value
)

# Populate glyphs
circle = p.circle(source=source, x="x", y="y", color="color", legend="legend")


def gfmt_callback(attr, new, old):
    """Callback for updating plot of GMFT results."""
    # Update color column
    if colorby_selector.value == "none":
        source.data["color"] = ["#1f77b3"] * len(ReinforcedVisit)
    elif colorby_selector.value == "ObjectReinforced":
        source.data["color"] = [
            "#1f77b3" if objects == "Blue" else "#ff7e0e"
            for objects in ReinforcedVisit["ObjectsReinforced"]
        ]
    elif colorby_selector.value == "Training":
        source.data["color"] = [
            "#1f77b3" if Training == "Trained" else "#ff7e0e"
            for Training in ReinforcedVisit["Training"]
        ]

    elif colorby_selector.value == "Training":
        source.data["color"] = [
            "#1f77b3" if Training == "Trained" else "#ff7e0e"
            for Training in ReinforcedVisit["Training"]
        ]

    # Update x-data and axis label
    source.data["x"] = ReinforcedVisit[x_selector.value]
    p.xaxis.axis_label = x_selector.value

    # Update x-data and axis label
    source.data["y"] = ReinforcedVisit[y_selector.value]
    p.yaxis.axis_label = y_selector.value


# Connect selectors to callback
colorby_selector.on_change("value", gfmt_callback)
x_selector.on_change("value", gfmt_callback)
y_selector.on_change("value", gfmt_callback)

# Build the layout
gfmt_layout = bokeh.layouts.row(
    p,
    bokeh.layouts.Spacer(width=15),
    bokeh.layouts.column(
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
