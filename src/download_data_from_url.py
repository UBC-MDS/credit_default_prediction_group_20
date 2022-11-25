# author: Arjun Radhakrishnan
# date: 2022-11-18

"""
Download an excel data file from the web and save it as an excel or a csv in the local file system.

Usage:
  download_data_from_url.py --url=<url> --download_path=<download_path> [--file_name=<file_name>] [--file_type=<file_type>]

Options:
  --url=<url>                           URL of the Excel File to be downloaded.
  --download_path=<download_path>       Local File Path where the file should be downloaded.
  --file_name=<file_name>               Name of the locally downloaded file [default: data].
  --file_type=<file_type>               Type or Extension of the locally saved file [default: csv].

Example:
From the root of the repository, the below command could be used to save a file to data/raw folder:
python ./src/download_data_from_url.py --url "<sample_url>" --download_path "./data/raw"

"""

import os
import requests
from docopt import docopt
import pandas as pd
import re

opt = docopt(__doc__)


def check_if_url_exists(url):
    """
    Checks if the url provided is an acutual URL.

    Parameters:
    ------
    url: (str)
    the input url

    Returns:
    -------
     Bool indicating the validity of the url
    """

    # Regex from https://www.geeksforgeeks.org/check-if-an-url-is-valid-or-not-using-regular-expression/

    regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    return re.search(regex, url) is not None


def main(url, download_path, file_name, file_type):
    """
    Driver function to download a file from the web
    and save it in the local file system.

    Parameters
    ----------
    url : string
        The URL from where the excel can be downloaded from.
    download_path : string
        The path where the file needs to be saved.
    file_name : string
        The name of the locally saved file.
    file_type : string
        The type in which the file will be locally saved.
    """

    assert check_if_url_exists(url), "It is not a valid url"

    # Test if the URL exists and returns Status OK
    try:
        request = requests.get(url, timeout=30)
        if request.status_code != 200:
            print("Unable to download from the web url.")
            return

    except requests.exceptions.ConnectionError as ex:
        print("Website at the provided url does not exist.")
        print(ex)
        return

    # Read data
    data = pd.read_excel(url)

    # Make necessary directories to the required path
    if not os.path.exists(download_path):
        os.makedirs(download_path)

    # Construct full file path
    file_name += "." + file_type
    full_path = os.path.join(download_path, file_name)

    # Save file based on user preference.
    if file_type == "xlsx" or file_type == "xls":
        pd.DataFrame.to_excel(data, full_path, index=False)
    else:
        pd.DataFrame.to_csv(data, full_path, index=False)

    assert os.path.isfile(full_path), "The downloaded file does not exist"


# Execute only when run as a script.
if __name__ == "__main__":
    main(opt["--url"], opt["--download_path"], opt["--file_name"], opt["--file_type"])
