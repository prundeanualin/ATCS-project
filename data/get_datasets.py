import os
import requests

SCAN_DIR = "SCAN"
SCAN_DATASET_FILEPATH = os.path.join(SCAN_DIR, "SCAN_dataset.csv")
SCAN_EXAMPLES_FILEPATH = os.path.join(SCAN_DIR, "SCAN_examples.txt")

url_scan_dataset = "https://raw.githubusercontent.com/prundeanualin/ATCS-project/blob/main/data/SCAN/SCAN_dataset.csv"
response = requests.get(url_scan_dataset)
if response.status_code == 200:
    csv_content = response.text
    with open(SCAN_DATASET_FILEPATH, "w+") as csv_file:
        csv_file.write(csv_content)

    print("SCAN dataset file downloaded successfully.")
else:
    raise Exception("Failed to download SCAN dataset file. Status code:", response.status_code)

url_scan_examples = "https://raw.githubusercontent.com/prundeanualin/ATCS-project/blob/main/data/SCAN/SCAN_examples.txt"
response = requests.get(url_scan_examples)
if response.status_code == 200:
    csv_content = response.text
    with open(SCAN_EXAMPLES_FILEPATH, "w+") as csv_file:
        csv_file.write(csv_content)

    print("SCAN examples file downloaded successfully.")
else:
    raise Exception("Failed to download SCAN examples file. Status code:", response.status_code)