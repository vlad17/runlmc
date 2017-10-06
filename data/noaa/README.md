# NOAA weather data

Raw station location data was retrieved from https://www.ncdc.noaa.gov/homr/reports/platforms. It is `wbanmasterlist.psv.zip`.

Raw climate data was retrieved from https://www.ncdc.noaa.gov/data-access/land-based-station-data/land-based-datasets/quality-controlled-local-climatological-data-qclcd. It is `2006_daily.tar.gz`.

`clean.py` is a Python 3 script that should be run from this directory and requires `pandas` and [basemap](https://matplotlib.org/basemap/users/installing.html) installed. It spits out `stations.pdf`, a plot of the climate station locations and `noaa.csv`, the cleaned CSV for this data.

Input: latitude, longitude, time (days since start of 2006)
Output: average daily temp, wind speed, pressure at sea level, and dew point.

