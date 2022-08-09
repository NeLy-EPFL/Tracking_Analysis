import pandas as pd

import ast
import iqplot
import bokeh.layouts
import bokeh.models
import bokeh.plotting
import holoviews as hv
import panel as pn
import colorcet

from holoviews.operation.timeseries import rolling

hv.extension('bokeh')

Data = pd.read_csv(
    "/Volumes/Ramdya-Lab/DURRIEU_Matthias/Experimental_data/MultiSensory_Project/GatedArenas/Results/DataSetNew.csv"
)



ThreshSlider = pn.widgets.IntSlider(name='ThreshSlider', value=80, start=60, end=270, step=10)

def slider_callback(ThreshSlider):
    for index, row in Data.iterrows():
        # print(row['Durations Left Corner'])

        # print (1 for i in row['Durations Left Corner'])
        Data.loc[index, "Peeks Left"] = sum(
            1 for i in ast.literal_eval(row["Durations Left Corner"]) if i > ThreshSlider
        )
        # Data_noWater_Simple['Peeks Left'][rows]= sum(1 for i in Data['Durations Left Corner'][rows] if i > param)
    # print(Data['Peeks Left'])
    box = hv.BoxWhisker(data=Data,
                        kdims="Training",
                        vdims="Peeks Left").opts(framewise=True,
                                                 ylim=(0, 40),
                                                 box_fill_alpha=0,
                                                 invert_axes=True,
                                                 invert_yaxis=True,
                                                 # box_line_color="gray",
                                                 )
    points = hv.Scatter(data=Data,
                        kdims="Training",
                        vdims="Peeks Left").opts(framewise=True,
                                                 cmap=colorcet.b_glasbey_category10,
                                                 invert_axes=True,
                                                 invert_yaxis=True,
                                                 ylim=(0, 40),
                                                 color="Training",
                                                 jitter=0.4,
                                                 )

    return box * points


dmap = hv.DynamicMap(pn.bind(slider_callback, ThreshSlider=ThreshSlider))
# dmap = hv.DynamicMap(slider_callback)

app = pn.Row(pn.WidgetBox('## Threshold Explorer', ThreshSlider),
             dmap.opts(width=500,
                       framewise=True,
                       )).servable()

pn.serve(app)
