import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import ast

import iqplot

import bokeh.io
import bokeh.models
import bokeh.plotting


Data = pd.read_csv('/Volumes/Ramdya-Lab/DURRIEU_Matthias/Experimental_data/MultiSensory_Project/GatedArenas/Results/DataSetNew.csv')

Data["Peeks Left"] = [0] * len(Data)

source = bokeh.models.ColumnDataSource(dict(Training=Data["Training"],
                                            LCDurations= Data['Durations Left Corner'],
                                            LPeeks=Data['Peeks Left']))

ThreshSlider = bokeh.models.Slider(
    title="Threshold",
    start=0,
    end=600,
    step=10,
    value=160
)



def slider_callback(attr, old, new):
    param = ThreshSlider.value

    for row in source.data:
        source.data["LPeeks"] = sum(1 for i in ast.literal_eval(source.data['LCDurations'][row]) if i > param)


    #Peeks_Right = sum(1 for i in Durations_Corner_Right if i > 160)
    #Peeks_Top = sum(1 for i in Durations_Corner_Top if i > 160)

ThreshSlider.on_change("value", slider_callback)

p_box = iqplot.stripbox(data=source,
                   q="LPeeks",
                   cats="Training",
                   )

Box_layout = bokeh.layouts.row(
    p_box,
    bokeh.models.Spacer(width=15),
    bokeh.layouts.column(
        ThreshSlider,
        width=200,
    ),
)

def Box_App(doc):
    doc.add_root(Box_layout)

# Build the app in the current doc
Box_App(bokeh.plotting.curdoc())