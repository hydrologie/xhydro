#!/usr/bin/env python

# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation,
# version 2.1 of the License.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this library; if not, write to the
# Free Software Foundation, Inc., 59 Temple Place - Suite 330,
# Boston, MA 02111-1307, USA.
#
# Author:  Vincent Fortin, Environment and Climate Change Canada, 2025-12-12
# Email:   vincent.fortin@ec.gc.ca
# Purpose: To read instantaneous streamflow data published by various
#          providers. Currently supported providers are ECCC and CEHQ.
#
# How to use it:
#
# Step 1:
# Create an object of class GetFlow_ECCC or GetFlow_CEHQ depending
# on the data provider that you want to use. Pass the station ID
# and the destination directory for the files that will be downloaded
# to the class constructor. Examples:
#
# obj = GetFlow_ECCC(station='04CA002', path='data')
# obj = GetFlow_CEHQ(station='073503', path='data')
# 
# Step 2: Call the method get_data() to obtain an xArray dataset
# containing all available instantaneous discharge for that station 
# ds = obj.get_data()
#
# Step 3: There is no step 3, you are done!
# Well, now the fun begins and you might want to do something smart
# using your dataset "ds". It will have a time dimension and a variable
# named "Discharge". It may have other columns depending on what other
# information is found in the files for that provider (such as quality
# control flags).
#
# If you want to contribute by adding a provider:
#
# Step 1: Create a class that inherits from GetFlow
# Step 2: Redefine the method obtain_file_list(self) to obtain the list
#         of files (possibly a single one) to download in order to
#         obtain the data for the station of interest.
#         This method should store the result in the attribute file_list.
# Step 3: Redefine the method url(self, filename) that provides the url
#         to use to download a single file of the list file_list.
# Step 4: Redefine the method decode_file(self, filepath). This method
#         processes a single file (after it has been downloaded) and
#         returns an xArray dataset with a time dimension, and at least
#         a variable named "Discharge". The dataset can include more
#         columns (optional).


# Dependencies
import os
import requests
import xarray as xr
import pandas as pd
from bs4 import BeautifulSoup
import re
from io import StringIO
import pytz

# Generic class to read instantaneous streamflow
class GetFlow:
    """
    Base class for downloading and decoding hydrometric discharge data.
    """

    # Set at instantiation in GetFlow Class
    station = None          # Station ID
    path = '.'              # Local path where downloaded datafiles will be saved

    # Set at instantiation in child class
    compression = None      # Compression method used on downloaded files
    file_list = []          # List of files to download and store at location "path"

    # Data obtained after downloading and processing datafiles
    data = None             # xArray dataset containing decoded data
        
    def __init__(self, station=None, path='.'):
        """Initialize GetFlow with station and output directory."""
        # Initialize base class with station identifier and output directory
        self.station = station
        self.path = path

    def obtain_file_list(self):
        # Must be implemented by child classes to populate self.file_list
        raise NotImplementedError()

    def url(self, filename):
        # Must be implemented by child classes to build the url from the elements of the file list
        raise NotImplementedError()

    def decode_file(self, filepath):
        # Must be implemented by child classes to decode downloaded files
        raise NotImplementedError()

    def download_file(self, url, path):
        # Download a file from URL and save it to disk, handling compression if needed
        response = requests.get(url)
        response.raise_for_status()
        if self.compression is None:
            # Write as plain text
            with open(path, "w", encoding="utf-8") as f:
                f.write(response.text)
        else:
            # Write as binary for compressed files
            with open(path, "wb") as f:
                f.write(response.content)

    def get_data(self):
        # Top-level method to retrieve and decode all relevant files
        self.obtain_file_list()
        ds_list = []
        for filename in self.file_list:
            basename = os.path.basename(filename)
            filepath = self.path + "/" + basename

            # Skip download if file already exists locally
            if os.path.exists(filepath):
                print(f"File already exists: {basename}, skipping.")
            else:
                print(f"Downloading {basename} ... ", end="")
                try:
                    print(self.url(filename))
                    self.download_file(self.url(filename), filepath)
                    print("Done.")
                except Exception as e:
                    print(f"Failed! ({e})")
                    raise RuntimeError

            # Process the file and decode into an xarray Dataset
            if os.path.exists(filepath):
                print(f"Processing {basename} ... ", end="")
                try:
                    ds_list.append(self.decode_file(filepath))
                    print("Done.")
                except Exception as e:
                    print(f"Failed! ({e})")
                    raise RuntimeError
            else:
                raise FileNotFoundError()

        # Concatenate all datasets along the time dimension
        self.data = xr.concat(ds_list, dim='time')

        # Add metadata to discharge variable
        self.data["Discharge"].attrs["units"] = "m³/s"
        self.data["Discharge"].attrs["long_name"] = "Instantaneous discharge"

        return self.data

class GetFlow_ECCC(GetFlow):
    """Retrieve instant discharge data from Environment and Climate Change Canada (ECCC)."""

    # Static configuration for ECCC data access
    website = 'https://collaboration.cmc.ec.gc.ca/cmc/hydrometrics/www/UnitValueData/Discharge'
    compression = 'xz'
    prefix_filename = 'Discharge.Working@'
    suffix_filename = f'.csv.{compression}'

    def __init__(self, station=None, path=None, quality='corrected'):
        # Initialize base class
        super().__init__(station, path)
        # Quality flag defines corrected or raw series
        assert(quality in ('corrected', 'raw'))
        self.quality = quality

    def obtain_file_list(self):
        # Build the directory URL based on station
        self.file_list = []
        webdir=f'{self.website}/{self.quality}/{self.station[0:2]}'
        print(f"Searching for data on {webdir}")

        response = requests.get(webdir)
        if response.status_code == 200:
            html_content = response.text
            soup = BeautifulSoup(html_content, "html.parser")
            links = soup.find_all("a")

            # Pattern to capture the start date embedded in filenames
            pattern = re.compile(f"{self.prefix_filename}{self.station}\\.(?P<start_date>\\d+)_{self.quality}{self.suffix_filename}")

            for link in links:
                file_name = link.get("href")
                if file_name:
                    match = pattern.search(file_name)
                    if match:
                        start_date = match.group("start_date")
                        # Only one file for each station in ECCC instant data
                        self.file_list = [f"{self.prefix_filename}{self.station}.{start_date}_{self.quality}{self.suffix_filename}"]

        if self.file_list == []:
            raise FileNotFoundError()

        print(f"Found this file on ECCC server: {self.file_list[0]}")

    def url(self, filename):
        # Build full URL to download the file
        return f'{self.website}/{self.quality}/{self.station[0:2]}/{filename}'

    def decode_file(self, filepath):
        # Load compressed CSV using pandas
        df = pd.read_csv(
            filepath,
            comment='#',
            header=0,
            compression=self.compression,
            dtype={'Value': 'float', 'Grade': 'int', 'Qualifiers': 'str'}
        )

        # Rename columns to standard names
        df.columns = ['time','LocalTime','Discharge','Approval','Grade','Qualifiers']

        # Convert time from ISO string ending in Z to UTC timestamp
        df['time'] = pd.to_datetime(df['time'].str.replace('Z', '+00:00'))
        df.pop('LocalTime')  # Remove unused local-time column

        # Sort and index by timestamp
        df.sort_values(by='time', inplace=True)
        df.set_index('time', inplace=True)

        # Convert to xarray Dataset
        ds = df.to_xarray()
        return ds

class GetFlow_CEHQ(GetFlow):
    """Retrieve instant discharge data from Québec's CEHQ archive."""

    website_fiche = 'https://www.cehq.gouv.qc.ca/hydrometrie/historique_donnees/fiche_instantanee.asp?NoStation='
    website_data = 'https://www.cehq.gouv.qc.ca/depot/historique_donnees_instantanees'
    compression = None

    def obtain_file_list(self):
        self.file_list = []
        # Scrape station page to find yearly data files
        webdir=f'{self.website_fiche}{self.station}'
        print(f"Searching for data on {webdir}")

        response = requests.get(webdir)
        if response.status_code == 200:
            html_content = response.text
            soup = BeautifulSoup(html_content, "html.parser")

            # Links that contain exactly 4 digits (year)
            links = soup.find_all("a", string=re.compile(r'^\d{4}$'))

            for link in links:
                file_name = link.get("href")
                if file_name:
                    self.file_list.append(os.path.basename(file_name))

        if self.file_list == []:
            raise FileNotFoundError()

        print(f"Found these files on CEHQ server:")
        for file in self.file_list:
            print(file)

    def url(self, filename):
        # Build file URL
        return f'{self.website_data}/{filename}'

    def decode_file(self, filepath):
        # Read the raw text file
        with open(filepath, encoding="utf-8") as f:
            lines = f.readlines()

        # Find header line identifying the start of the fixed-width table
        header_idx = None
        for i, line in enumerate(lines):
            if line.strip().startswith("Station        Date"):
                header_idx = i
                break

        if header_idx is None:
            raise ValueError(f"Table header not found in {filepath}")

        # Extract everything after the header
        data_lines = lines[header_idx+1:]
        table_str = "".join(data_lines)

        # Define fixed-width column specs
        colspecs = [
            (0, 15),   # Station
            (15, 35),  # Date with time
            (35, 47),  # Discharge (m³/s)
            (47, None) # Qualifiers / remarks
        ]
        columns = ["Station", "LocalTime", "Discharge", "Qualifiers"]

        # Parse table with pandas
        try:
            df = pd.read_fwf(
                StringIO(table_str),
                sep=r"\s{2,}",
                engine="python",
                dtype={"Station": str},
                names=columns,
                colspecs=colspecs
            )
            df.pop("Station")
        except Exception as e:
            raise RuntimeError(f"Error parsing file {filepath}: {e}")

        # Convert local time (QC EST, no DST) to UTC
        df["time"] = pd.to_datetime(df["LocalTime"], format="%Y/%m/%d %H:%M", errors="coerce")
        df["time"] = df["time"].dt.tz_localize(pytz.FixedOffset(-5 * 60))
        df["time"] = df["time"].dt.tz_convert("UTC")
        df.pop("LocalTime")

        # Clean numeric columns
        df["Discharge"] = pd.to_numeric(df["Discharge"], errors="coerce")
        df["Qualifiers"] = df.get("Qualifiers", pd.Series([None] * len(df))).replace("", pd.NA)

        # Sort and index by time
        df.sort_values(by='time', inplace=True)
        df.set_index('time', inplace=True)

        # Convert to xarray Dataset
        ds = df.to_xarray()
        return ds
