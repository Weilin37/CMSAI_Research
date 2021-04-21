"""
A module that anonymizes data (replacing patient ids with random alphanumeric characters)
"""

import os
import json
import random
import string
import time
import pandas as pd


def get_fpaths(data_dir, data_types):
    """
    Get all csv filenames from directory.
    Args:
        data_dir(str): Data directory
        data_types(list): List of data types to read
    Returns:
        List of csv file paths
    """
    fpaths = []
    for data_type in data_types:
        my_data_dir = os.path.join(data_dir, data_type)
        fnames = os.listdir(my_data_dir)
        fnames = [fname for fname in fnames if fname.endswith(".csv")]
        fnames = sorted(fnames)
        fpaths += [os.path.join(my_data_dir, fname) for fname in fnames]
    return fpaths


def get_unique_values(fpaths, column):
    """
    Get all unique values from all files in a given column.
    Args:
        fpaths(list): List file paths
        column(str): Column chosen
    Returns:
        List of unique values
    """
    print("Getting unique values for the {} column...".format(column))
    values = set()
    for fpath in fpaths:
        print("Processing {}...".format(fpath))
        df = pd.read_csv(fpath, low_memory=False)
        ids = set(df[column].tolist())
        values.update(ids)
    print("Success!")
    return list(values)


def _generate_strings(num_values=1000, length=16):
    """
    Generate random strings for patient ids.
    Args:
        num_values(int): Number of strings to be generated
        length(int): String length
    Returns:
        List of the generated strings
    """

    def _generate_unique_string(generated_set, length):
        """
        Generate unique random string for patient id
        Source: https://stackoverflow.com/questions/2511222/efficiently-generate-a-16-character-alphanumeric-string
        """
        # x = ''.join(random.choices(string.ascii_letters + string.digits, k=length))
        x = "".join(random.choices(string.ascii_uppercase + string.digits, k=length))
        set_len = len(x)
        generated_set.add(x)
        while set_len == len(generated_set):
            x = "".join(
                random.choices(string.ascii_uppercase + string.digits, k=length)
            )
            generated_set.add(x)
        return generated_set

    gen_set = set()
    for i in range(num_values):
        gen_set = _generate_unique_string(gen_set, length)
        if i % 10000 == 0:
            print(
                "{}% of the strings already generated!".format((i * 100.0) / num_values)
            )
    return list(gen_set)


def get_all_mappings(unique_values, column, output_dir, string_length):
    """
    Get mappings from the given unique values for a column to be anonymized.
    Args:
        unique_values(list): List of unique values in a given column
        column(str): Selected column
        output_dir(str): Output directory
        string_length(int): Random string length
    Returns:
        Path where all the mappings are saved
    """
    print("Getting all mappings of values with randomly generated strings...")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    num_values = len(unique_values)
    rand_values = _generate_strings(num_values, string_length)
    all_mappings = dict(zip(unique_values, rand_values))

    output_path = os.path.join(output_dir, column + "_mappings.json")

    if os.path.exists(output_path):
        raise ValueError(
            "Error! {} already exists. Please remove it and try again.".format(
                output_path
            )
        )

    with open(output_path, "w") as fp:
        json.dump(all_mappings, fp)
    print("Success! Mappings saved to {}!".format(output_path))
    return output_path


def read_mappings(mappings_path):
    """
    Read the patients mappings
    Args:
        mappings_path(str): Mappings path
    Returns:
        Dict of mappings
    """
    with open(mappings_path, "r") as fp:
        mappings = json.load(fp)
    (keys, values) = zip(*mappings.items())
    keys = [int(key) for key in keys]
    mappings = dict(zip(keys, values))
    return mappings


def apply_mappings(filepaths, mappings, column, re_column2=None):
    """
    Add mapping columns.
    Args:
        filepaths(list): List of file paths.
        mappings(dict): Dictionary of mappings
        column(str): Column name
        re_column2(str): Second column selected
    Returns:
        None
    """

    def _map_values(row, mappings, column, re_column2):
        """Get mapping for a dataframe row's specific column"""
        value = row[column]
        new_map = [mappings[value]]

        if re_column2 is not None:
            value2 = row[re_column2]
            value2 = value2.split("_")[-1]
            new_value2 = "{}_{}".format(new_map[0], value2)
            new_map.append(new_value2)
        return new_map

    for fpath in filepaths:
        print("Anonymizing {}...".format(fpath))
        df = pd.read_csv(fpath, low_memory=False)
        new_values = df.apply(_map_values, axis=1, args=(mappings, column, re_column2))

        new_values1 = [val[0] for val in new_values]
        df[column] = new_values1
        if re_column2 is not None:
            new_values2 = [val[1] for val in new_values]
            df[re_column2] = new_values2

        # Create output_dir
        output_dir, fname = fpath.rsplit("/", 1)
        output_dir = output_dir.replace(
            "Raw", "Anonymized"
        )  # Save output in Anonymized folder

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_path = os.path.join(output_dir, fname)
        df.to_csv(output_path, index=False)
    print("Data Successfully Anonymized!")


if __name__ == "__main__":
    ANONYMIZED_COLUMN = "patient_id"
    RANDOM_STRING_LENGTH = 9

    ## AE
    ROOT_DIR = "/home/ec2-user/SageMaker/CMSAI/modeling/tes/data/anonymize/AE/Data/"
    RAW_DIR = os.path.join(ROOT_DIR, "Raw")
    OUTPUT_DIR = os.path.join(ROOT_DIR, "Anonymized")
    RE_COLUMN2 = None

    DATA_TYPES = ["365NoDeath", "365TestPhase"]
    MAPPINGS_PATH = os.path.join(
        OUTPUT_DIR, "{}_mappings.json".format(ANONYMIZED_COLUMN)
    )

    print("ANONYMIZING AE DATA...")
    # Creating the patient_id mappings dictionary
    filepaths = get_fpaths(RAW_DIR, DATA_TYPES)
    unique_values = get_unique_values(filepaths, ANONYMIZED_COLUMN)
    mappings_path = get_all_mappings(
        unique_values, ANONYMIZED_COLUMN, OUTPUT_DIR, RANDOM_STRING_LENGTH
    )

    # Apply the mappings
    filepaths = get_fpaths(RAW_DIR, DATA_TYPES)
    mappings = read_mappings(MAPPINGS_PATH)
    apply_mappings(filepaths, mappings, ANONYMIZED_COLUMN, RE_COLUMN2)

    # ================================================================================

    # RE
    ROOT_DIR = "/home/ec2-user/SageMaker/CMSAI/modeling/tes/data/anonymize/RE/Data/"
    RAW_DIR = os.path.join(ROOT_DIR, "Raw")
    OUTPUT_DIR = os.path.join(ROOT_DIR, "Anonymized")

    DATA_TYPES = ["365", "365TestPhase"]
    MAPPINGS_PATH = os.path.join(
        OUTPUT_DIR, "{}_mappings.json".format(ANONYMIZED_COLUMN)
    )
    RE_COLUMN2 = "discharge_id"

    print("ANONYMIZING RE DATA...")
    # Creating the patient_id mappings dictionary
    filepaths = get_fpaths(RAW_DIR, DATA_TYPES)
    unique_values = get_unique_values(filepaths, ANONYMIZED_COLUMN)
    mappings_path = get_all_mappings(
        unique_values, ANONYMIZED_COLUMN, OUTPUT_DIR, RANDOM_STRING_LENGTH
    )

    # Apply the mappings
    filepaths = get_fpaths(RAW_DIR, DATA_TYPES)
    mappings = read_mappings(MAPPINGS_PATH)
    apply_mappings(filepaths, mappings, ANONYMIZED_COLUMN, RE_COLUMN2)

    # ================================================================================

    # RE With Dates
    ROOT_DIR = (
        "/home/ec2-user/SageMaker/CMSAI/modeling/tes/data/anonymize/REWithDates/Data/"
    )
    RAW_DIR = os.path.join(ROOT_DIR, "Raw")
    OUTPUT_DIR = os.path.join(ROOT_DIR, "Anonymized")

    DATA_TYPES = ["365", "365TestPhase"]
    MAPPINGS_PATH = os.path.join(
        OUTPUT_DIR, "{}_mappings.json".format(ANONYMIZED_COLUMN)
    )
    RE_COLUMN2 = "discharge_id"

    print("ANONYMIZING RE WITH DATE DATA...")
    # Creating the patient_id mappings dictionary
    filepaths = get_fpaths(RAW_DIR, DATA_TYPES)
    unique_values = get_unique_values(filepaths, ANONYMIZED_COLUMN)
    mappings_path = get_all_mappings(
        unique_values, ANONYMIZED_COLUMN, OUTPUT_DIR, RANDOM_STRING_LENGTH
    )

    # Apply the mappings
    filepaths = get_fpaths(RAW_DIR, DATA_TYPES)
    mappings = read_mappings(MAPPINGS_PATH)
    apply_mappings(filepaths, mappings, ANONYMIZED_COLUMN, RE_COLUMN2)

    print("ANONYMIZATION SUCCESSFULLY COMPLETED!")
