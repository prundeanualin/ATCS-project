import os
import requests

############################
########### SCAN ###########
############################

SCAN_DIR = "SCAN"
SCAN_DATASET_FILEPATH = os.path.join(SCAN_DIR, "SCAN_dataset.csv")
SCAN_EXAMPLES_FILEPATH = os.path.join(SCAN_DIR, "SCAN_examples_{}.txt")

SCAN_DATASET_BASE_URL = "https://raw.githubusercontent.com/taczin/SCAN_analogies/blob/main/data/"
EXAMPLES_BASE_URL = "https://raw.githubusercontent.com/prundeanualin/ATCS-project/blob/main/data/"

EXAMPLES = ["baseline"]


def get_datasets_if_not_present():
    # For SCAN
    if os.path.exists(SCAN_DATASET_FILEPATH):
        print("SCAN datasets already downloaded.")
    else:
        url_dataset = SCAN_DATASET_BASE_URL + SCAN_DATASET_FILEPATH
        response = requests.get(url_dataset)
        if response.status_code == 200:
            csv_content = response.text
            with open(SCAN_DATASET_FILEPATH, "w+") as csv_file:
                csv_file.write(csv_content)

            print("SCAN dataset file downloaded successfully.")
        else:
            raise Exception("Failed to download SCAN dataset file. Status code:", response.status_code)

        for ex in EXAMPLES:
            example_filepath = SCAN_EXAMPLES_FILEPATH.format(ex)
            response = requests.get(EXAMPLES_BASE_URL + example_filepath)
            if response.status_code == 200:
                csv_content = response.text
                with open(example_filepath, "w+") as csv_file:
                    csv_file.write(csv_content)

                print("SCAN examples file downloaded successfully.")
            else:
                raise Exception("Failed to download SCAN examples file. Status code:", response.status_code)
