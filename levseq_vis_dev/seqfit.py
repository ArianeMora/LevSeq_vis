"""
A script for visualzing the sequnece-fitness relationship
"""

import re
import os

from copy import deepcopy

# Get them w.r.t to a mutation
from scipy.stats import mannwhitneyu
from tqdm import tqdm
import pandas as pd
import numpy as np

from sklearn.decomposition import PCA

import esm
import torch

import ninetysix as ns


# Amino acid code conversion
AA_DICT = {
    "Ala": "A",
    "Cys": "C",
    "Asp": "D",
    "Glu": "E",
    "Phe": "F",
    "Gly": "G",
    "His": "H",
    "Ile": "I",
    "Lys": "K",
    "Leu": "L",
    "Met": "M",
    "Asn": "N",
    "Pro": "P",
    "Gln": "Q",
    "Arg": "R",
    "Ser": "S",
    "Thr": "T",
    "Val": "V",
    "Trp": "W",
    "Tyr": "Y",
    "Ter": "*",
}


ALL_AAS = list(AA_DICT.values())
AA_NUMB = len(ALL_AAS)

# Create a dictionary that links each amino acid to an index
AA_TO_IND = {aa: i for i, aa in enumerate(ALL_AAS)}

# Set up cuda variables
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def normalise_calculate_stats(
    processed_plate_df,
    value_columns,
    normalise="standard",
    stats_method="mannwhitneyu",
    parent_label="#PARENT#",
    normalise_method="median",
):
    parent = parent_label
    # if nomrliase normalize with standard normalisation
    normalised_value_columns = []
    normalised_df = pd.DataFrame()
    if normalise:
        for plate in set(processed_plate_df["Plate"].values):
            for value_column in value_columns:
                sub_df = processed_plate_df[processed_plate_df["Plate"] == plate].copy()
                parent_values = sub_df[sub_df["amino-acid_substitutions"] == parent][
                    value_column
                ].values
                # By default use the median
                if normalise_method == "median":
                    parent_mean = np.median(parent_values)
                else:
                    parent_mean = np.mean(parent_values)
                parent_sd = np.std(parent_values)

                # For each plate we normalise to the parent of that plate
                sub_df[f"{value_column} plate standard norm"] = (
                    sub_df[value_column].values - parent_mean
                ) / parent_sd
                normalised_value_columns.append(f"{value_column} plate standard norm")
                normalised_df = pd.concat([normalised_df, sub_df])
    else:
        normalised_df = processed_plate_df

    normalised_value_columns = list(set(normalised_value_columns))
    processed_plate_df = normalised_df

    sd_cutoff = 1.5  # The number of standard deviations we want above the parent values
    # Now for all the other mutations we want to look if they are significant, first we'll look at combinations and then individually
    grouped_by_mutations = processed_plate_df.groupby("amino-acid_substitutions")

    rows = []
    for mutation, grp in tqdm(grouped_by_mutations):
        # Get the values and then do a ranksum test
        if mutation != parent:
            for value_column in normalised_value_columns:
                parent_values = list(
                    processed_plate_df[
                        processed_plate_df["amino-acid_substitutions"] == parent
                    ][value_column].values
                )
                if normalise_method == "median":
                    parent_mean = np.median(parent_values)
                else:
                    parent_mean = np.mean(parent_values)
                parent_sd = np.std(parent_values)

                vals = list(grp[value_column].values)
                U1, p = None, None
                # Now check if there are 3 otherwise we just do > X S.D over - won't be sig anyway.
                if len(grp) > 2:
                    # Do stats
                    U1, p = mannwhitneyu(parent_values, vals, method="exact")
                mean_vals = np.mean(vals)
                std_vals = np.std(vals)
                median_vals = np.median(vals)
                sig = mean_vals > ((sd_cutoff * parent_sd) + parent_mean)
                rows.append(
                    [
                        value_column,
                        mutation,
                        len(grp),
                        mean_vals,
                        std_vals,
                        median_vals,
                        mean_vals - parent_mean,
                        sig,
                        U1,
                        p,
                    ]
                )
    stats_df = pd.DataFrame(
        rows,
        columns=[
            "value_column",
            "amino-acid_substitutions",
            "number of wells with amino-acid substitutions",
            "mean",
            "std",
            "median",
            "amount greater than parent mean",
            f"greater than > {sd_cutoff} parent",
            "man whitney U stat",
            "p-value",
        ],
    )
    return stats_df


def checkNgen_folder(folder_path: str) -> str:

    """
    Check if the folder and its subfolder exists
    create a new directory if not
    Args:
    - folder_path: str, the folder path
    """
    # get rid of the very first / if it exists
    if folder_path[0] == "/":
        folder_path = folder_path[1:]

    # if input path is file
    if bool(os.path.splitext(folder_path)[1]):
        folder_path = os.path.dirname(folder_path)

    split_list = os.path.normpath(folder_path).split("/")
    for p, _ in enumerate(split_list):
        subfolder_path = "/".join(split_list[: p + 1])
        if not os.path.exists(subfolder_path):
            print(f"Making {subfolder_path} ...")
            os.mkdir(subfolder_path)
    return folder_path


def work_up_lcms(
    file,
    products,
    substrates=None,
    drop_string=None,
):
    """Works up a standard csv file from Revali.
    Parameters:
    -----------
    file: string
        Path to the csv file
    products: list of strings
        Name of the peaks that correspond to the product
    substrates: list of strings
        Name of the peaks that correspond to the substrate
    drop_string: string, default 'burn_in'
        Name of the wells to drop, e.g., for the wash/burn-in period that are not samples.
    Returns:
    --------
    plate: ns.Plate object (DataFrame-like)
    """

    if isinstance(file, str):
        # Read the first row of the file to inspect
        first_row = pd.read_csv(file, nrows=2, header=None)

        # Check if the first entry contains "Compound name (signal)"
        if "Compound name (signal)" in str(first_row.iloc[0, 0]):
            header_row = 1  # The actual header is on the second row
        else:
            header_row = 0  # The first row is the actual header

        # Load the CSV with the determined header row
        df = pd.read_csv(file, header=header_row)
    else:
        # Change to handling both
        df = file
    # Convert nans to 0
    df = df.fillna(0)
    # Only grab the Sample Acq Order No.s that have a numeric value
    index = [True for _ in df["Sample Acq Order No"]]
    for i, value in enumerate(df["Sample Acq Order No"]):
        try:
            int(value)
        except ValueError:
            index[i] = False
    # Index on this
    df = df[index]

    def fill_vial_number(series):
        for i, row in enumerate(series):
            if pd.isna(row):
                series[i] = series[i - 1]
        return series

    df["Sample Vial Number"] = fill_vial_number(df["Sample Vial Number"].copy())
    # Drop empty ones!
    df = df[df["Sample Vial Number"] != 0]
    # Remove unwanted wells
    df = df[df["Sample Name"] != drop_string]
    # Get wells

    df.insert(
        0, "Well", df["Sample Vial Number"].apply(lambda x: str(x).split("-")[-1])
    )

    # Create minimal DataFrame
    df = df[["Well", "Plate", "Compound Name", "Area"]].reset_index(drop=True)
    # Pivot table; drop redundant values by only taking 'max' with aggfunc
    # (i.e., a row is (value, NaN, NaN) and df is 1728 rows long;
    # taking max to aggregate duplicates gives only (value) and 576 rows long)
    df = df.pivot_table(
        index=["Well", "Plate"], columns="Compound Name", values="Area", aggfunc="max"
    ).reset_index()
    # Get rows and columns
    df.insert(
        1, "Column", df["Well"].apply(lambda x: int(x[1:]) if x[1:].isdigit() else None)
    )
    df.insert(1, "Row", df["Well"].apply(lambda x: x[0]))
    # Set values as floats
    cols = products + substrates if substrates is not None else products
    for col in cols:
        df[col] = df[col].astype(float)
    plate = ns.Plate(df, value_name=products[-1]).set_as_location("Plate", idx=3)
    plate.values = products
    return plate


# Function to process the plate files
def process_plate_files(
    products: list, seq_df: pd.DataFrame, fit_df: pd.DataFrame, plate_names: list
) -> pd.DataFrame:

    """
    Process the plate files to extract relevant data for downstream analysis.
    Assume the same directory contains the plate files with the expected names.
    The expected filenames are constructed based on the Plate values in the input CSV file.
    The output DataFrame contains the processed data for the specified products
    and is saved to a CSV file named 'seqfit.csv' in the same dirctory.

    Args:
    - products : list
        The name of the product to be analyzed. ie ['pdt']
    - seq_df : pd.DataFrame
        A pandas DataFrame containing the sequence data.
    - fit_df : pd.DataFrame
        A pandas DataFrame containing the fitness data.

    Returns:
    - pd.DataFrame
        A pandas DataFrame containing the processed data.
    - str
        The path of the output CSV file containing the processed data.
    """

    seq_columns = [
        "Plate",
        "Well",
        "amino-acid_substitutions",
        "nt_sequence",
        "aa_sequence",
        "Alignment Count",
        "Average mutation frequency",
    ]

    seq_df = seq_df[seq_columns].copy()

    seq_fit_column_order = (
        [
            "Plate",
            "Well",
            "Parent_Name",
            "amino-acid_substitutions",
            "# Mutations",
            "Type",
        ]
        + products
        + [p + "_fold" for p in products]
        + [
            "nt_sequence",
            "aa_sequence",
            "Alignment Count",
            "Average mutation frequency",
        ]
    )

    # Create an empty list to store the processed plate data
    processed_data = []

    # Iterate over unique Plates and search for corresponding CSV files in the current directory
    for plate in plate_names:

        plate_seq = seq_df[seq_df["Plate"] == plate].reset_index(drop=True).copy()
        # extract the fit data for the plate
        plate_fit = fit_df[fit_df["Plate"] == plate].reset_index(drop=True).copy()

        # Work up data to plate object
        plate_object = work_up_lcms(plate_fit, products)

        # Extract attributes from plate_object as needed for downstream processes
        if hasattr(plate_object, "df"):
            # Assuming plate_object has a dataframe-like attribute 'df' that we can work with
            plate_fit_df = plate_object.df
            plate_fit_df["Plate"] = plate  # Add the plate identifier for reference

            # Merge filtered_df with plate_df to retain amino-acid_substitutionss and nt_sequence columns
            merged_df = pd.merge(
                plate_fit_df, plate_seq, on=["Plate", "Well"], how="outer"
            )
            processed_data.append(merged_df)

    # Concatenate all dataframes if available
    if processed_data:
        processed_df = pd.concat(processed_data, ignore_index=True)
    else:
        return pd.DataFrame(seq_fit_column_order)

    # Ensure all entries in 'Mutations' are treated as strings
    processed_df["amino-acid_substitutions"] = processed_df[
        "amino-acid_substitutions"
    ].astype(str)

    # Match plate to parent
    parent_dict, plate2parent = match_plate2parent(processed_df)
    processed_df["Parent_Name"] = processed_df["Plate"].map(plate2parent)

    # apply the norm function to all plates
    processed_df = (
        processed_df.groupby("Plate")
        .apply(norm2parent, products=products)
        .reset_index(drop=True)
        .copy()
    )

    processed_df["# Mutations"] = [
        len(str(m).split("_")) if m not in ["#N.A.#", "#PARENT#", "-", "#LOW#"] else 0
        for m in processed_df["amino-acid_substitutions"].values
    ]

    # do not count the stop codon at the very end
    processed_df["Type"] = [
        m if "*" not in str(v)[:-1] else "#TRUNCATED#"
        for m, v in processed_df[["amino-acid_substitutions", "aa_sequence"]].values
    ]

    processed_df["Type"] = [
        v if v != "-" else "#DELETION#" for v in processed_df["Type"].values
    ]
    processed_df["Type"] = [
        v if str(v)[0] == "#" else "#VARIANT#" for v in processed_df["Type"].values
    ]
    processed_df["Type"] = [
        v if v != "#DELETION#" else "Deletion" for v in processed_df["Type"].values
    ]
    processed_df["Type"] = [
        v if v != "#VARIANT#" else "Variant" for v in processed_df["Type"].values
    ]
    processed_df["Type"] = [
        v if v != "#PARENT#" else "Parent" for v in processed_df["Type"].values
    ]
    processed_df["Type"] = [
        v if v != "#TRUNCATED#" else "Truncated" for v in processed_df["Type"].values
    ]
    processed_df["Type"] = [
        v if v != "#LOW#" else "Low" for v in processed_df["Type"].values
    ]
    processed_df["Type"] = [
        v if v != "#N.A.#" else "Empty" for v in processed_df["Type"].values
    ]

    # Return the processed DataFrame for downstream processes
    return processed_df[seq_fit_column_order].reset_index(drop=True).copy()


def match_plate2parent(df: pd.DataFrame) -> dict:

    """
    Find plate names correpsonding to each parent sequence.

    Args:
    - df : pd.DataFrame
        A pandas DataFrame containing the data for a single plate.
        The DataFrame should have the following columns:
        - "Plate" : str
            The plate identifier.
        - "Well" : str
            The well identifier.
        - "Mutations" : str
            The mutations in the well.
    - parent_dict : dict
        A dictionary containing the parent name for each aa_varient.

    Returns:
    - dict
        A dictionary containing the plate names for each parent sequence.
    """

    # get the parent nt_sequence
    parent_aas = (
        df[df["amino-acid_substitutions"] == "#PARENT#"][
            ["amino-acid_substitutions", "aa_sequence"]
        ]
        .drop_duplicates()["aa_sequence"]
        .tolist()
    )

    parent_dict = {f"Parent-{i+1}": parent for i, parent in enumerate(parent_aas)}

    # get the plate names for each parent
    parent2plate = {
        p_name: df[df["aa_sequence"] == p_seq]["Plate"].unique().tolist()
        for p_name, p_seq in parent_dict.items()
    }

    # reverse the dictionary to have plate names as keys and rasie flag if there are multiple parents for a plate
    plate2parent = {}
    for parent, plates in parent2plate.items():
        for plate in plates:
            if plate in plate2parent:
                raise ValueError(f"Multiple parents found for plate {plate}")
            else:
                plate2parent[plate] = parent

    return parent_dict, plate2parent


def detect_outliers_iqr(series: pd.Series) -> pd.Index:

    """
    Calculate the Interquartile Range (IQR) and
    determine the lower and upper bounds for outlier detection.

    The IQR is a measure of statistical dispersion and
    is calculated as the difference between the third quartile (Q3)
    and the first quartile (Q1) of the data

    Args:
    - series : pandas.Series
        A pandas Series containing the data for which the IQR and bounds are to be calculated.

    Returns:
    - tuple
        A tuple containing the lower bound and upper bound for outlier detection.

    Example:
    --------
    >>> import pandas as pd
    >>> data = pd.Series([10, 12, 14, 15, 18, 20, 22, 23, 24, 25, 100])
    >>> calculate_iqr_bounds(data)
    (-1.0, 39.0)
    """

    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    return series[(series < lower_bound) | (series > upper_bound)].index


def norm2parent(plate_df: pd.DataFrame, products: list) -> pd.DataFrame:

    """
    For each given plate,
    normalize the pdt values of a plate to the mean of the parent
    without the outliers.

    Args:
    - plate_df : pd.DataFrame
        A pandas DataFrame containing the data for a single plate.
        The DataFrame should have the following columns:
        - "Plate" : str
            The plate identifier.
        - "Mutations" : str
            The mutations in the well.
        - "pdt" : float
            The pdt value for the well.

    Returns:
    - pd.DataFrame
        A pandas DataFrame containing the normalized pdt values.
    """

    # get all the parents from the df
    parents = (
        plate_df[plate_df["amino-acid_substitutions"] == "#PARENT#"]
        .reset_index(drop=True)
        .copy()
    )

    for product in products:
        filtered_parents = (
            parents.drop(index=detect_outliers_iqr(parents[product]))
            .reset_index(drop=True)
            .copy()
        )

        # normalize the whole plate to the mean of the filtered parent
        plate_df[product + "_fold"] = (
            plate_df[product] / filtered_parents[product].mean()
        )

    return plate_df


def process_mutation(mutation: str) -> pd.Series:
    # Check if mutation is #PARENT#
    if mutation == "#PARENT#":
        return pd.Series([0, [(None, None, None)]])  # Return 0 sites and NaN details

    # Split by "_" to get number of sites
    sites = mutation.split("_")
    num_sites = len(sites)

    # Extract details if it matches the pattern
    details = []
    for site in sites:
        match = re.match(r"^([A-Z])(\d+)([A-Z*])$", site)
        if match:
            parent_aa, site_number, mutated_aa = match.groups()
            details.append((parent_aa, site_number, mutated_aa))
        else:
            details.append((None, None, None))

    return pd.Series([num_sites, details])


def up2stopcodon(sequence: str) -> str:

    """
    Preprocesses the amino acid sequence
    by removing everything after the first '*' (stop codon).
    """

    if "*" in sequence:
        sequence = sequence.split("*")[0]  # Take everything before the first '*'
    return sequence


def get_max_layer(model_name: str) -> int:
    """
    Get the maximum layer number for the specified model.

    ie esm1_t6_43M_UR50S has 6 layers
    esm2_t12_35M_UR50D has 12 layers
    """

    return int(model_name.split("_")[1][1:])


def append_xy(
    df: pd.DataFrame,
    products: list,
    model_name="esm2_t12_35M_UR50D",
    output_file=None,
    batch_size=32,
):

    # Remove the "Unnamed: 0" column if it exists
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # Create the ID column as the combination of `Plate` and `Well`
    df["ID"] = df["Plate"] + "-" + df["Well"]
    df = df[
        ["ID"] + [col for col in df.columns if col != "ID"]
    ]  # Reorder to make ID the first column

    # Filter valid sequences from the `aa_sequence` column
    df = (
        df[df["Type"].isin(["Parent", "Variant", "Truncated"])]
        .dropna(subset=["aa_sequence"])
        .copy()
    )

    # Preprocess sequences to handle stop codons
    df["processed_aa_sequence"] = df["aa_sequence"].apply(up2stopcodon)

    if "esm" in model_name:

        max_layer = get_max_layer(model_name)

        # Load the ESM-2 model
        model, alphabet = esm.pretrained.load_model_and_alphabet_hub(model_name)
        batch_converter = alphabet.get_batch_converter()

        model.eval()  # disable dropout for deterministic results
        model.to(DEVICE)

        # Prepare sequences for embedding
        # Extract data into the desired format
        formatted_data = [
            (row["ID"], row["processed_aa_sequence"]) for _, row in df.iterrows()
        ]

        # Initialize results
        all_sequence_representations = []
        all_batch_labels = []

        # Process in batches
        for i in tqdm(range(0, len(formatted_data), batch_size)):
            batch = formatted_data[i : i + batch_size]
            batch_labels, batch_strs, batch_tokens = batch_converter(batch)
            batch_tokens = batch_tokens.to(DEVICE)

            with torch.no_grad():
                token_representations = (
                    model(batch_tokens, repr_layers=[max_layer], return_contacts=False)[
                        "representations"
                    ][max_layer]
                    .cpu()
                    .numpy()
                )

            for j, tokens_len in enumerate(
                (batch_tokens != alphabet.padding_idx).sum(1)
            ):
                all_sequence_representations.append(
                    token_representations[j, 1 : tokens_len - 1].mean(0)
                )
                all_batch_labels.append(batch_labels[j])

            # Clear memory after each batch
            torch.cuda.empty_cache()

        # Convert the embeddings to a numpy array
        sequence_embeddings = np.array(all_sequence_representations)

        print(f"Number of sequences: {len(all_batch_labels)}")

    # all else do onehot
    else:
        all_sequence_representations = []

        for seq in tqdm(df["processed_aa_sequence"].tolist()):
            # padding: (top, bottom), (left, right)
            all_sequence_representations.append(
                np.pad(
                    np.array(np.eye(AA_NUMB)[[AA_TO_IND[aa] for aa in seq]]),
                    pad_width=(
                        (0, df["processed_aa_sequence"].apply(len).max() - len(seq)),
                        (0, 0),
                    ),
                )
            )

        # flatten emb
        all_sequence_representations = np.array(all_sequence_representations)
        # encoded_mut_seqs.reshape(encoded_mut_seqs.shape[0], -1)
        sequence_embeddings = all_sequence_representations.reshape(
            np.array(all_sequence_representations).shape[0], -1
        )
        all_batch_labels = df["ID"].tolist()

    print(f"sequence_embeddings shape: {sequence_embeddings.shape}")

    # Dimensionality Reduction using PCA
    pca = PCA(n_components=2)
    xy_coordinates = pca.fit_transform(sequence_embeddings)

    # Add x, y coordinates back to the dataframe
    xy_df = pd.DataFrame(
        xy_coordinates, columns=["x_coordinate", "y_coordinate"], index=all_batch_labels
    ).rename_axis("ID")

    return pd.merge(
        df.set_index("ID"), xy_df, left_index=True, right_index=True, how="left"
    ).reset_index()


def prep_single_ssm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the data for a single sitessm summary plot.

    Args:
    - df: pd.DataFrame, input full dataframe

    Returns:
    - pd.DataFrame, output dataframe
    """

    # slice out single site SSM and add in parentAA, site, and mutAA columns
    single_ssm_df = df[
        (df["# Mutations"] <= 1) & (df["Type"].isin(["Parent", "Variant"]))
    ].copy()

    # Apply function to the column
    single_ssm_df[["num_sites", "mut_dets"]] = single_ssm_df[
        "amino-acid_substitutions"
    ].apply(process_mutation)

    # Expand the single entry in Details for these rows into three columns
    single_ssm_df[["parent_aa", "site_numb", "mut_aa"]] = pd.DataFrame(
        single_ssm_df["mut_dets"].apply(lambda x: x[0]).tolist(),
        index=single_ssm_df.index,
    )

    single_ssm_df["parent_aa_loc"] = (
        single_ssm_df["parent_aa"] + single_ssm_df["site_numb"]
    )

    # fill nan site numbers with 0 and convert to int
    single_ssm_df["site_numb"] = single_ssm_df["site_numb"].fillna(0).astype(int)

    return single_ssm_df


def get_single_ssm_site_df(
    single_ssm_df: pd.DataFrame, parent: str, site: str
) -> pd.DataFrame:
    """
    Get the single site SSM data for a given site with appended parent data.

    Args:
    - single_ssm_df: pd.DataFrame, input single site SSM dataframe
    - parent: str, parent to filter the data on
    - site: str, site to filter the data on

    Returns:
    - pd.DataFrame, output dataframe
    """

    # get the site data
    site_df = (
        single_ssm_df[
            (single_ssm_df["Parent_Name"] == parent)
            & (single_ssm_df["parent_aa_loc"] == site)
        ]
        .reset_index(drop=True)
        .copy()
    )

    # get parents from those plates
    site_parent_df = (
        single_ssm_df[
            (single_ssm_df["amino-acid_substitutions"] == "#PARENT#")
            & (single_ssm_df["Plate"].isin(site_df["Plate"].unique()))
        ]
        .reset_index(drop=True)
        .copy()
    )

    # rename those site_numb, mut_aa, parent_aa_loc None or NaN to corresponding parent values
    site_parent_df["mut_aa"] = site_parent_df["mut_aa"].fillna(
        site_df["parent_aa"].values[0]
    )
    site_parent_df["site_numb"] = site_parent_df["site_numb"].fillna(
        site_df["site_numb"].values[0]
    )
    site_parent_df["parent_aa_loc"] = site_parent_df["parent_aa_loc"].fillna(
        site_df["parent_aa_loc"].values[0]
    )

    # now merge the two dataframes
    return pd.concat([site_parent_df, site_df]).reset_index(drop=True).copy()


def prep_aa_order(df: pd.DataFrame, add_na: bool = False) -> pd.DataFrame:
    """
    Prepare the data for a single sitessm summary plot.

    Args:
    - df: pd.DataFrame, input full dataframe

    Returns:
    - pd.DataFrame, output dataframe
    """

    # Define the order of x-axis categories
    x_order = list(AA_DICT.values())

    if add_na:
        x_order += ["#N.A.#"]

    # Convert `Mutations` to a categorical column with specified order
    df["mut_aa"] = pd.Categorical(df["mut_aa"], categories=x_order, ordered=True)

    # Sort by the `x_order`, filling missing values
    return (
        df.sort_values("mut_aa", key=lambda x: x.cat.codes)
        .reset_index(drop=True)
        .copy()
    )


def get_parent2sitedict(df: pd.DataFrame) -> dict:

    """
    Get a dictionary of parent to site mapping for single site mutants.

    Args:
    - df : pd.DataFrame

    Returns:
    - dict
        A dictionary containing the parent sequence and site number for each parent.
    """

    site_dict = deepcopy(
        df[["Parent_Name", "parent_aa_loc"]]
        .drop_duplicates()
        .dropna()
        .groupby("Parent_Name")["parent_aa_loc"]
        .apply(list)
        .to_dict()
    )

    # Sort the site list for each parent as an integer
    for parent, sites in site_dict.items():
        # Ensure each site is processed as a string and sorted by the integer part
        site_dict[parent] = sorted(sites, key=lambda site: int(str(site)[1:]))

    return site_dict


def get_x_label(x: str):

    """
    Function to return the x-axis label based on the input string.
    """

    if "mut_aa" in x.lower():
        clean_x = x.replace("mut_aa", "Amino acid substitutions")
    else:
        clean_x = x.replace("_", " ").capitalize()

    return clean_x


def get_y_label(y: str):

    """
    Function to return the y-axis label based on the input string.
    """
    clean_y = ""

    # if "pdt" in y.casefold():  # Case-insensitive check
    #     clean_y = y.replace("pdt", "Product", case=False)
    if "area" in y.casefold():  # Case-insensitive check
        clean_y = y.replace("area", "Yield", case=False)
    elif y == "fitness_ee2/(ee1+ee2)":
        clean_y = "ee2/(ee1+ee2)"
    elif y == "fitness_ee1/(ee1+ee2)":
        clean_y = "ee1/(ee1+ee2)"
    else:
        clean_y = y

    # normalize the y label
    if "_norm" in y.lower() or "_fold" in y.lower():
        clean_y = f"Normalized {clean_y.replace('_fold', '').replace('_norm', '')}"

    # if "fold" in y.lower():
    #     clean_y = f"{clean_y}"
    return clean_y