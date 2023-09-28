"""
Credits: https://github.com/felipecode/coiltraine/blob/master/docs/exploring_limitations.md
"""

import os
import sys

import tarfile
import gdown 

if __name__ == "__main__":

    # Download the full data from the models
    print ("Downloading the coil models checkpoints   1.1 GB")
    file_id = '1AzSIkmGETGSNLBtWxoTLGjqLlXNVgkEH'
    destination_final = 'data/models'

    os.system(f'gdown https://drive.google.com/uc?id={file_id}')

    tf = tarfile.open("nocrash_basic.tar.gz")
    tf.extractall(destination_final)
    # Remove both the original and the file after moving.
    os.remove("nocrash_basic.tar.gz")

