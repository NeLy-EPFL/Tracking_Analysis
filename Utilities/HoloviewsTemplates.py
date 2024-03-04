from bokeh.models import HoverTool

import holoviews as hv
import Utils
import Processing
import datetime
from icecream import ic
import pandas as pd

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
        "color": "label",
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
        "invert_axes": True,
        "show_grid": True,
        "fontscale": 1.5,
        "title": "",
    },
}

# Custom Jitterboxplot function


def compute_controls_bs_ci(
    data, metric, genotypes=["TNTxZ2018", "TNTxZ2035", "TNTxM6", "TNTxM7"]
):
    """
    Compute a 95% bootstrap confidence interval for a given metric for flies with specified genotypes. The default usage is to compute the confidence interval for the control genotypes, which are the ones set as default in the genotypes argument.

    Args:
        data (pandas DataFrame): The dataset to compute the confidence interval from.
        metric (str): The metric to compute the confidence interval for.
        genotypes (list): A list of control genotypes.

    Returns:
        np.ndarray: The lower and upper confidence interval bounds.
    """

    # Filter the dataset for flies with control genotypes
    control_data = data[data["Genotype"].isin(genotypes)]

    # Check if there are any control flies in the dataset
    if control_data.empty:
        ic("No flies with control genotypes found in the dataset.")
        return None

    # Drop rows with NaN values in the metric column
    control_data = control_data.dropna(subset=[metric])

    # Compute the bootstrap confidence interval for the given metric
    ci = Processing.draw_bs_ci(control_data[metric].values)

    return ci


def jitter_boxplot(
    data,
    vdim,
    metadata,
    bs_controls=True,
    plot_options=hv_main,
    show=True,
    save=False,
    outpath=None,
):
    """
    Generate a jitter boxplot for a given metric. The jitter boxplot is a combination of a boxplot and a scatterplot. The boxplot shows the distribution of the metric for each brain region, while the scatterplot shows the value of the metric for each fly.
    The plot includes a bootstrap confidence interval for the control genotypes that is displayed as a shaded area behind the boxplot.

    Args:
        data (pandas DataFrame): The dataset to plot.
        vdim (str): The metric to plot. It should be a column in the dataset.
        metadata (list): A list of metadata columns in the dataset.
        plot_options (dict, optional): A dictionary containing the plot options. Defaults to hv_main which is MD's base template for holoviews plots found in Holoviews_Templates.py.
        show (bool, optional): Whether to display the plot (Currently only tested on Jupyter notebooks). Defaults to True.
        save (bool, optional): Whether to save the plot as an html file. Defaults to False.
        outpath (str, optional): The path where to save the plot. Defaults to None. If None is provided, the plot is saved in a generic location with a timestamp and information on the metric plotted.

    Returns:
        holoviews plot: A jitter boxplot.
    """

    # Filter out groups where the vdim column is all NaN
    data = data.groupby("Brain region").filter(lambda x: x[vdim].notna().any())

    # Clean the data by removing NaN values for this metric
    data = data.dropna(subset=[vdim])

    # Convert vdim to numeric
    data[vdim] = pd.to_numeric(data[vdim], errors="coerce")

    # Get the metadata for the tooltips
    tooltips = [
        ("Fly", "@fly"),
        (vdim.capitalize(), f"@{vdim}"),
    ]

    # Add the metadata to the tooltips
    for var in metadata:
        tooltips.append((var.capitalize(), f"@{var}"))

    hover = HoverTool(tooltips=tooltips)

    # Compute the bootstrap confidence interval for the metric
    if bs_controls:
        bs_ci = compute_controls_bs_ci(data, vdim)
    else:
        bs_ci = None

    # Get the limits for the y axis
    y_min = data[vdim].min()
    y_max = data[vdim].max()

    hv_boxplot = (
        hv.BoxWhisker(
            data=data,
            vdims=vdim,
            kdims=["label", "Brain region"],
        )
        .groupby("Brain region")
        .opts(**plot_options["boxwhisker"], ylim=(y_min, y_max))
    )

    hv_scatterplot = (
        hv.Scatter(
            data=data,
            vdims=[vdim] + metadata + ["fly"],
            kdims=["label", "Brain region"],
        )
        .groupby("Brain region")
        .opts(**plot_options["scatter"], tools=[hover], ylim=(y_min, y_max))
    )

    if bs_ci is not None:
        # Create an Area plot for the confidence interval
        hv_bs_ci = hv.HSpan(bs_ci[0], bs_ci[1]).opts(fill_alpha=0.2, color="red")
        hv_jitter_boxplot = (hv_bs_ci * hv_boxplot * hv_scatterplot).opts(
            ylabel=f"{vdim}", **plot_options["plot"]
        )

    else:
        ic("No control data to generate the confidence interval.")
        hv_jitter_boxplot = (hv_boxplot * hv_scatterplot).opts(
            ylabel=f"{vdim}", **plot_options["plot"]
        )

    if show:
        hv.render(hv_jitter_boxplot)
    if save:
        if outpath is None:
            now = datetime.datetime.now()  # get current date and time
            date_time = now.strftime("%Y%m%d_%H%M")  # format as a string

            output_path = (
                Utils.get_labserver()
                / "Experimental_data"
                / "MultiMazeRecorder"
                / "Plots"
                / f"{vdim}_{date_time}.html"
            )

        hv.save(hv_jitter_boxplot, output_path)

    return hv_jitter_boxplot
