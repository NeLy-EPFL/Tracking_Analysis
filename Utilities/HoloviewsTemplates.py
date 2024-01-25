from bokeh.models import HoverTool

# My main template

hv_main = {
    "boxwhisker": {
        "box_fill_color": None,
        "box_line_color": "black",
        "outlier_fill_color": None,
        "outlier_line_color": None,
        "framewise": True,
    },
    "scatter": {
        "jitter": 0.3,
        "color": "Nickname",
        "alpha": 0.5,
        "size": 6,
        "cmap": "Category10",
        "framewise": True,
    },
    "plot": {
        "width": 750,
        "height": 500,
        "show_legend": False,
        "xlabel": "",
        "ylabel": "Number of Events",
        "invert_axes": True,
        "show_grid": True,
        "fontscale": 1.5,
        "title": "",
        "toolbar": None,
    },
}
