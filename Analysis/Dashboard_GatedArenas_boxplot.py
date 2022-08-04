ThreshSlider = bokeh.models.Slider(
    title="Peeks",
    start=0,
    end=600,
    step=10,
    value=160
)


def slider_callback(attr, old, new):
    param = ThreshSlider.value

    Data['Peekings_Left'] = sum(1 for i in Data['Durations_Corner_Left'] if i > param)
    #Peeks_Right = sum(1 for i in Durations_Corner_Right if i > 160)
    #Peeks_Top = sum(1 for i in Durations_Corner_Top if i > 160)

slider.on_change("value", slider_callback)

p_box = iqplot.stripbox(data=Data,
                   q="Peekings_Left",
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

def induction_app(doc):
    doc.add_root(induction_layout)