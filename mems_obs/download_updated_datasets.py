import os
import sys

import gdown 

if __name__ == "__main__":

    # Download the full data from the models
    print ("Downloading the Adroit updated datasets")
    file_id = '1GA6cfzcUZ-luZR0kuCXSwd5fhf1anLX9'
    destination_final = 'mems_obs/updated_datasets'
    os.makedirs(destination_final, exist_ok=True)

    os.system(f'gdown https://drive.google.com/uc?id={file_id}')

    # Unzip the file
    os.system(f'unzip updated_datasets.zip -d {destination_final}')

    # Remove both the original and the file after moving.
    os.remove("updated_datasets.zip")

