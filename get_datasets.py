import os
import requests

############################
########### SCAN ###########
############################

SCAN_DIR = "data/SCAN"
SCAN_DATASET_FILEPATH = os.path.join(SCAN_DIR, "SCAN_dataset.csv")
SCAN_EXAMPLES_FILEPATH = os.path.join(SCAN_DIR, "SCAN_examples_{}.txt")

SCAN_DATASET_URL = "https://raw.githubusercontent.com/taczin/SCAN_analogies/main/data/SCAN_dataset.csv"

EXAMPLE_CATEGORIES = ["baseline"]


# For BATS dataset
BATS_FOLDER = 'data/BATS'
BATS_EXAMPLE_FILE = 'data/BATS/BATS_example.pkl'


def get_datasets_if_not_present():
    # For SCAN
    if os.path.exists(SCAN_DATASET_FILEPATH):
        print("SCAN datasets already downloaded.")
    else:
        response = requests.get(SCAN_DATASET_URL)
        if response.status_code == 200:
            csv_content = response.text
            with open(SCAN_DATASET_FILEPATH, "w+") as csv_file:
                csv_file.write(csv_content)

            print("SCAN dataset file downloaded successfully.")
        else:
            raise Exception("Failed to download SCAN dataset file. Status code:", response.status_code)
