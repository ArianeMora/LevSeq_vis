import streamlit as st
import re
import pandas as pd
from threading import Thread
import plotly.express as px
import plotly.graph_objects as go
from levseq_vis_dev.seqfit import (
    normalise_calculate_stats,
    process_plate_files,
    append_xy,
    prep_aa_order,
    prep_single_ssm,
    get_parent2sitedict,
    get_single_ssm_site_df,
    get_x_label,
    get_y_label,
)

from streamlit.runtime.scriptrunner import add_script_run_ctx

# """
# The base of this app was developed from:
# https://share.streamlit.io/streamlit/example-app-csv-wrangler/
# """


# Set the config for the plotly charts
config = {
    "toImageButtonOptions": {
        "format": "svg",  # Set the download format to SVG
    }
}


def _max_width_():
    max_width_str = f"max-width: 1800px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>    
    """,
        unsafe_allow_html=True,
    )


st.title("LevSeq sequence function pairing")
st.subheader("Beta mode, for issues post [here](https://github.com/fhalab/LevSeq)")

c1, c2 = st.columns([6, 6])

df, seq_variant, fitness_files = None, None, None

with c1:
    seq_variant = st.file_uploader(
        "Variant sequencing file (output from LevSeq)",
        key="1",
        help="To activate 'wide mode', go to the hamburger menu > Settings > turn on 'wide mode'",
    )
with c2:
    fitness_files = st.file_uploader(
        "Fitness files",
        key="2",
        accept_multiple_files=True,  # Allow multiple files
        help="To activate 'wide mode', go to the hamburger menu > Settings > turn on 'wide mode'",
    )

with c1:
    if seq_variant is not None:
        seq_variant = pd.read_csv(seq_variant)

with c2:

    fit_df_list = []  # List to store all the fitness files
    products_set = set()  # Set to hold unique product names
    plate_names = []  # List to hold the names of the plates

    if fitness_files:
        for fitness in fitness_files:
            try:
                # Read the file into a DataFrame without assuming any header
                df = pd.read_csv(fitness, header=None)

                # Check if the first entry contains "Compound name (signal)"
                if "Compound name (signal)" in str(df.iloc[0, 0]):
                    # Use the second row as the header
                    df.columns = df.iloc[1]  # Set the second row as the header
                    df = (
                        df[2:].reset_index(drop=True).copy()
                    )  # Drop the first two rows (metadata and headers)
                else:
                    # Use the first row as the header
                    df.columns = df.iloc[0]  # Set the first row as the header
                    df = (
                        df[1:].reset_index(drop=True).copy()
                    )  # Drop the first row (headers)

                plate_df = df.dropna(subset=["Compound Name"])
                plate_name = str(fitness.name).split(".")[0]

                # add a column for the plate name
                plate_df["Plate"] = plate_name

                fit_df_list.append(plate_df)  # Append DataFrame to the list
                plate_names.append(plate_name)  # Extract plate names
                products_set.update(
                    plate_df["Compound Name"].dropna().values
                )  # Add products to the set

            except Exception as e:
                st.warning(f"Could not read file {fitness.name}: {e}")

        # Check if there are valid DataFrames in the list
        if fit_df_list:
            fit_df = pd.concat(fit_df_list, ignore_index=True)
        else:
            st.warning("No valid fitness files were uploaded or parsed.")

        # Display multi-select dropdown for products
        if products_set:
            products = st.multiselect(
                "Select products",
                sorted(products_set),  # Sort for better UI experience
                help="Select one or more products to filter.",
            )
        else:
            st.warning("No products found in the fitness files.")

padd3, c0, padd4 = st.columns([1, 6, 1])


def make_alignment_plot(df):
    # Streamlit app
    st.title("Alignment Count Plot")
    st.write("This is a histogram showing alignment counts categorized by type.")

    # Define custom color mapping
    color_mapping = {
        "Empty": "lightgrey",
        "Low": "#A6A7AC",
        "Deletion": "#6E6E6E",
        "Truncated": "#FCE518",
        "Parent": "#97CA43",
        "Variant": "#3A578F",
    }

    # Create the Plotly histogram
    fig = px.histogram(
        df,
        x="Alignment Count",
        color="Type",
        color_discrete_map=color_mapping,
        barmode="stack",
        category_orders={
            "Type": ["Empty", "Low", "Deletion", "Truncated", "Parent", "Variant"]
        },
        title="Alignment Counts",
    )

    # Customize layout
    fig.update_layout(
        xaxis_title="Alignment Count",
        yaxis_title="Count",
        title_x=0.5,
        legend_title="Type",
        template="simple_white",
    )

    # Display the plot in Streamlit
    st.plotly_chart(fig, config=config)


def make_scatter_plot(df, parents_list):
    # If there are more than 1 in the fitness, plot both.
    # ---------------------------------------------------------
    if len(products) > 1:
        prod_1 = products[0]
        prod_2 = products[1]
    else:
        prod_1 = products[0]
        prod_2 = "Alignment Count"

    st.title("Scatter plot of multiple features")
    st.write(
        "Either your first two features otherwise the only feature and the alignment count."
    )

    # Define the color palette as a dictionary for Plotly
    color_mapping = {
        "Empty": "lightgrey",
        "Low": "#A6A7AC",
        "Deletion": "#6E6E6E",
        "Truncated": "#FCE518",
        "Parent": "#97CA43",
        "Variant": "#3A578F",
    }

    SHPAE_LIST = ["circle", "diamond", "triangle-up", "square", "cross"]

    # Define the order of the legend with parents first
    type_order = ["Empty", "Low", "Deletion", "Truncated", "Parent", "Variant"]

    shape_mapping = {
        p: s for p, s in zip(parents_list, SHPAE_LIST[: len(parents_list)])
    }

    df["size"] = [
        m + 1 if isinstance(m, int) and m > 0 else 1 for m in df["# Mutations"]
    ]

    # Create the Plotly scatter plot
    fig = px.scatter(
        df,
        x=prod_1,
        y=prod_2,
        color="Type",
        color_discrete_map=color_mapping,
        symbol="Parent_Name",
        symbol_map=shape_mapping,
        size="size",
        category_orders={
            "Parent_Name": parents_list,
            "Type": type_order,
        },
        hover_data=[
            "Type",
            "Parent_Name",
            "Plate",
            "Well",
            "amino-acid_substitutions",
            "# Mutations",
            prod_1,
            prod_2,
        ],
        title=f"{prod_1} vs {prod_2}",
    )

    # Update axis ranges and ticks
    fig.update_xaxes(title=f"{prod_1}")
    fig.update_yaxes(title=f"{prod_2}")

    # Adjust legend position
    fig.update_layout(
        legend=dict(title="Type", x=1.05, y=1, xanchor="left", yanchor="top")
    )
    st.plotly_chart(fig, config=config)


def plot_bar_point(
    df,
    x,
    y,
    x_label=None,
    y_label=None,
    title=None,
    if_max=False,
    bar_color=None,
    colorscale=None,
    highlight_label=None,  # New: Value to highlight
    highlight_color="white",  # New: Color for the highlighted bar
    showlegend=True,
):

    # Group data by `x` and calculate mean for bars
    bar_data = df.groupby(x).mean()[y].reset_index()

    if bar_color:
        bar_kwargs = dict(marker_color=bar_color)
    elif colorscale:
        # Define bar colors conditionally
        bar_colors = [
            highlight_color if label == highlight_label else value
            for label, value in zip(bar_data[x], bar_data[y])
        ]

        # Define bar line styles conditionally
        bar_lines = [
            {"color": "gray", "width": 2}
            if val == highlight_label
            else {"color": "white", "width": 0}
            for val in bar_data[x]
        ]

        bar_kwargs = dict(
            marker=dict(
                color=bar_colors,  # Color based on y-values
                colorscale=colorscale,  # Use the specified colorscale
                line=dict(
                    color=[line["color"] for line in bar_lines],
                    width=[line["width"] for line in bar_lines],
                ),  # Add outlines for highlighted bar
                showscale=True,  # Show color scale legend
                colorbar=dict(
                    title=dict(
                        text=get_y_label(y),
                        side="right",  # Move title to the left of the color bar
                        # standoff=15,  # Adjust the spacing
                    ),
                    thickness=15,  # Set the length of the color bar
                    outlinewidth=0,  # Remove the border of the color bar
                ),  # Add title to the color scale
            )
        )
    else:
        bar_kwargs = dict()

    # Create bar plot
    bar_trace = go.Bar(x=bar_data[x], y=bar_data[y], name="Average", **bar_kwargs)

    # Create scatter plot
    scatter_trace = go.Scatter(
        x=df[x],
        y=df[y],
        mode="markers",
        name="Points",
        marker=dict(size=8, color="gray", opacity=0.6),
        text=df[["Plate", "Well"]].astype(str).agg(", ".join, axis=1),  # Tooltip info
    )

    # Add max points (if requested)
    traces = [bar_trace, scatter_trace]
    if if_max:
        max_data = df.loc[df.groupby(x)[y].idxmax()]
        max_trace = go.Scatter(
            x=max_data[x],
            y=max_data[y],
            mode="markers",
            name="Max",
            marker=dict(size=10, color="orange"),
            text=max_data[["Plate", "Well"]].astype(str).agg(", ".join, axis=1),
        )
        traces.append(max_trace)

    # Combine all traces
    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title or f"{x} vs {y}",
        xaxis=dict(
            title=x_label or get_x_label(x),
            showline=True,
            linewidth=1,
            linecolor="gray",
        ),
        yaxis=dict(
            title=y_label or get_y_label(y),
            showline=True,
            linewidth=1,
            linecolor="gray",  # Color of the axis line
        ),
        showlegend=showlegend,
    )

    return fig


def agg_parent_plot(df, ys):
    # Create plots for each metric
    plots = [
        plot_bar_point(
            df,
            x="Parent_Name",
            y=y,
            bar_color="#97CA43",
            title=f"{get_y_label(y)} across parents",
            if_max=True,
        )
        for y in ys
        if y in df.columns
    ]

    if not plots:
        return None

    # Display plots in Streamlit
    for plot in plots:
        st.plotly_chart(plot, config=config)


def agg_mut_plot(sites_dict, single_ssm_df, ys):
    """
    Create individual plots for each parent and their respective sites.

    Args:
    - sites_dict (dict): A dictionary where keys are parents and values are lists of sites.
    - single_ssm_df (pd.DataFrame): DataFrame containing the single site mutation data.
    - ys (list): List of columns to plot.
    """
    for parent, sites in sites_dict.items():
        st.subheader(f"Parent: {parent}")  # Section title for each parent

        for site in sites:
            # Preprocess the site-specific data
            site_df = prep_aa_order(
                get_single_ssm_site_df(single_ssm_df, parent=parent, site=site)
            )

            if site_df.empty:
                st.warning(f"No data available for Parent: {parent}, Site: {site}")
                continue  # Skip if there's no data for the site

            site_info = (
                site_df["parent_aa_loc"].unique()[0] if not site_df.empty else "Unknown"
            )

            st.markdown(f"**Site: {site_info}**")  # Add site-specific label

            # Generate plots for each `y` value
            for y in ys:
                if y in site_df.columns:
                    fig = plot_bar_point(
                        site_df,
                        x="mut_aa",
                        y=y,
                        title=f"Parent: {parent}, Site: {site_info}, Metric: {get_y_label(y)}",
                        if_max=False,
                        highlight_label=site_info[0],  # New: Value to highlight
                        highlight_color="white",  # New: Color for the highlighted bar
                        colorscale="RdBu_r",
                        showlegend=False,
                    )
                    st.plotly_chart(fig, config=config)


def plot_single_ssm_avg(single_ssm_df, ys):
    # Preprocess data for each parent
    parents = single_ssm_df["Parent_Name"].unique()

    # Check if the data is empty
    if single_ssm_df.empty:
        st.warning("No data available to plot.")
        return

    # Create a container for all the plots
    all_plots = []

    for parent_name in parents:
        for y in ys:
            if y in single_ssm_df.columns:
                sliced_df = prep_aa_order(
                    single_ssm_df[single_ssm_df["Parent_Name"] == parent_name]
                )
                # Skip if sliced_df is empty
                if sliced_df.empty:
                    continue

                # Prepare heatmap data
                heatmap_data = sliced_df.pivot_table(
                    index="mut_aa", columns="parent_aa_loc", values=y, aggfunc="mean"
                ).reindex(
                    columns=sorted(
                        sliced_df["parent_aa_loc"].dropna().unique(),
                        key=lambda x: int(re.search(r"(\d+)$", str(x)).group())
                        if re.search(r"(\d+)$", str(x))
                        else 0,
                    )
                )

                # Create Plotly heatmap
                fig = px.imshow(
                    heatmap_data.T,
                    color_continuous_scale="RdBu_r",
                    labels={
                        "x": "Amino acid substitutions",
                        "y": "Position",
                        "color": y,
                    },
                    title=f"Average Single Site Substitution for {parent_name}",
                )

                # Update colorbar settings
                fig.update_coloraxes(
                    colorbar_title=get_y_label(y),
                    colorbar=dict(
                        title_side="right",  # Move title to the left of the color bar
                    ),
                )

                # Remove horizontal gridlines
                fig.update_layout(
                    xaxis=dict(showgrid=False),  # Disable gridlines for x-axis
                    yaxis=dict(showgrid=False),  # Disable gridlines for y-axis
                )
                all_plots.append(fig)

    # Render all plots in Streamlit
    for fig in all_plots:
        st.plotly_chart(fig, config=config)


def plot_embxy(df: pd.DataFrame, product: str):

    """
    Function to plot the x, y embedding coordinates colored by a specified metric using Plotly.

    Args:
    - df : pd.DataFrame
        A pandas DataFrame containing the data for the x, y embedding coordinates.
    - product : str
        The column name to be used for coloring the scatter plot.

    Returns:
    - fig : plotly.graph_objs.Figure
        A Plotly figure object containing the scatter plot.
    """

    # Create a scatter plot using Plotly Express
    fig = px.scatter(
        df,
        x="x_coordinate",
        y="y_coordinate",
        color=product,
        hover_data=["ID", "amino-acid_substitutions", "Parent_Name"],
        title=f"{product} in sequence embedding space",
        color_continuous_scale="RdBu_r",
        labels={
            "x_coordinate": "Embedding x",
            "y_coordinate": "Embedding y",
            product: product,
        },
    )
    # Customize layout
    fig.update_layout(
        width=800,
        height=600,
        coloraxis_colorbar={"title": product},
    )
    return fig


def agg_embxy(df: pd.DataFrame, products: list):
    """
    Function to aggregate the x, y embedding coordinates colored by different metrics using Plotly.

    Args:
    - df : pd.DataFrame
        A pandas DataFrame containing the data for the x, y embedding coordinates.
    - products : list
        A list of column names for which the x, y embedding coordinates are to be colored.

    Returns:
    - figs : list
        A list of Plotly figure objects containing the scatter plots.
    """

    plots = [plot_embxy(df, product) for product in products if product in df.columns]

    if not plots:
        return None

    # Display plots in Streamlit
    for plot in plots:
        st.plotly_chart(plot, config=config)


def seqfit_runner():
    st.info(f"Running LevSeq using your files! Results will appear below shortly.")

    if fit_df is None or seq_variant is None:
        st.error("Please upload both Sequence and Fitness files.")
        return

    # Process variants
    df = process_plate_files(
        products=products, fit_df=fit_df, seq_df=seq_variant, plate_names=plate_names
    ).copy()

    fold_products = [f"{p}_fold" for p in products]
    parents_list = df["Parent_Name"].unique()

    # ------------------------ Display paired seq function data as a table
    st.title("Joined sequence function data (fold change wrt parent per plate)")
    st.dataframe(df)  # Interactive table with scrollbars

    # ------------------------ Stats
    value_columns = products
    stats_df = pd.DataFrame()
    for value in value_columns:
        stats_df = pd.concat(
            [
                stats_df,
                normalise_calculate_stats(
                    df,
                    [value],
                    normalise="standard",
                    stats_method="mannwhitneyu",
                    parent_label="#PARENT#",
                ),
            ]
        )

    stats_df = stats_df.sort_values(
        by="amount greater than parent mean", ascending=False
    )
    # ------------------------ Display stats data as a table
    st.title("Statistics on the function data")
    st.dataframe(stats_df)  # Interactive table with scrollbars

    # -------------------------- Make visualisations

    make_alignment_plot(df)
    make_scatter_plot(df, parents_list)
    # st.bokeh_chart(gen_seqfitvis(df, products))
    # Generate parent plot
    st.header("Parent Aggregation")
    agg_parent_plot(df, ys=fold_products)

    # Generate mutation plots
    st.header("Mutation Aggregation")
    single_ssm_df = prep_single_ssm(df)
    plot_single_ssm_avg(single_ssm_df, ys=fold_products)

    # Generate single SSM plots
    st.header("Single SSM Heatmap")
    agg_mut_plot(
        sites_dict=get_parent2sitedict(single_ssm_df),
        single_ssm_df=single_ssm_df,
        ys=fold_products,
    )

    # Generate embedding plot
    st.header("Sequence Embedding")
    st.subheader("esm2_t12_35M_UR50D PCA (will take a while)")
    agg_embxy(append_xy(df, products=fold_products), products=fold_products)

    st.subheader("Done LevSeq!")


def run_run():
    thread = Thread(target=seqfit_runner, args=())
    add_script_run_ctx(thread)
    thread.start()
    thread.join()


with c0:
    if c1 is not None and c2 is not None:
        st.button("Run LevSeq <3", on_click=run_run)
    else:
        st.info(
            f"""
               Upload a variant file and a fitness file.
            """
        )
        st.stop()