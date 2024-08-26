import os
import zipfile
import shutil

if __name__ == '__main__':
    data_url = 'https://zenodo.org/records/10390295/files/arcade.zip'
    data_path = 'data/arcade.zip'

    # Create the data directory
    if not os.path.exists('data'):
        os.makedirs('data')

    if not os.path.exists(data_path):
        print('[+] Downloading data...')
        os.system(f'wget {data_url} -O {data_path}') # Download the data
        print('\tData downloaded.')

        print('[+] Extracting data...')
        with zipfile.ZipFile(data_path, 'r') as zip_ref:
            zip_ref.extractall('data') # Extract the data
            print('\tData extracted.')

    # Remove the (redundant) stenosis folder
    if os.path.exists('data/arcade/stenosis'):
        shutil.rmtree('data/arcade/stenosis')
