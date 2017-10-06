import datetime
import numpy as np
import os
import pandas as pd
import tempfile
import tarfile
from zipfile import ZipFile

station_zip = 'wbanmasterlist.psv.zip'
climate_tar = '2006_daily.tar.gz'

assert station_zip in os.listdir(), station_zip
assert climate_tar in os.listdir(), climate_tar

with tempfile.TemporaryDirectory() as tmp:
    print('expanding', station_zip, 'and', climate_tar, 'into', tmp)
    with ZipFile(station_zip) as stations:
        stations.extractall(tmp)
    with tarfile.open(climate_tar) as climate:
        climate.extractall(tmp)
    climate_dir = os.path.join(tmp, '2006_daily')
    stations_file = os.path.join(tmp, 'wbanmasterlist.psv')

    print('reading in raw data')
    stations = pd.read_csv(stations_file, sep='|', header=0)
    climates = [os.path.join(climate_dir, f) for f in os.listdir(climate_dir)]
    daily_dfs = [pd.read_csv(f, header=0, encoding='ISO-8859-1')
                 for f in climates]
    for df in daily_dfs:
        df.columns = df.columns.str.strip()
    climate = pd.concat(daily_dfs, ignore_index=True)

print('cleaning climate data')
climate['time'] = pd.to_datetime(climate['YearMonthDay'], format='%Y%m%d')
climate['time'] = (climate['time'] - pd.to_datetime('2006-01-01')
                   ) / datetime.timedelta(days=1)
oldcols = ['Wban Number', 'Avg Temp', 'Avg Dew Pt',
           'Pressure Avg Sea Level', 'Wind Avg Speed', 'time']
newcols = ['wban', 'temp', 'dew', 'pressure', 'wind', 'time']
climate = climate.rename(columns=dict(zip(oldcols, newcols)))
climate = climate[newcols]
print('climate NAs per column')
print(len(climate) - climate.count())

print('cleaning station data')
oldcols = ['WBAN_ID', 'LOCATION']
newcols = ['wban', 'loc']
stations = stations.rename(columns=dict(zip(oldcols, newcols)))
stations = stations[newcols]
print('stations NAs per column')
print(len(stations) - stations.count())

stations = stations.dropna()
missing = len(set(climate.wban) - set(stations.dropna().wban))
tot = len(set(climate.wban))
print('stations in climate missing location data', missing, 'of', tot)

joined = climate.merge(stations, on='wban', how='left')
tot = len(joined)


numerics = ['temp', 'dew', 'pressure', 'wind', 'time']
for name in numerics:
    cleanstr = joined[name].astype('str').str.replace(r'[^\d\.]', '')
    joined[name] = pd.to_numeric(cleanstr)

joined = joined.dropna()
print('left-joined climate/stations data has',
      len(joined), 'of', tot, 'clean rows')


print('parsing lattitude/longitude')

import re

p1 = re.compile(
    r"""(\d+)\D(\d+)\D+(\d+)\D*(N|S)\W+(\d+)\D(\d+)\D(\d+)\D*(E|W)""")
p2 = re.compile(r"""(\d+)\D+(\d+)\D+(N|S)\W+(\d+)\D+(\d+)\D+(E|W)""")
lats = []
lons = []
badparse = 0
for i in joined['loc']:
    m = p1.match(i)
    if m:
        lat = float(m.group(1)) + float(m.group(2)) / \
            60 + float(m.group(3)) / 60 / 60
        lat *= 1 if m.group(4) == 'N' else -1
        lon = float(m.group(5)) + float(m.group(6)) / \
            60 + float(m.group(7)) / 60 / 60
        lon *= 1 if m.group(8) == 'E' else -1
        lats.append(lat)
        lons.append(lon)
        continue
    m = p2.match(i)
    if m:
        lat = float(m.group(1)) + float(m.group(2)) / 60
        lat *= 1 if m.group(3) == 'N' else -1
        lon = float(m.group(4)) + float(m.group(5)) / 60
        lon *= 1 if m.group(6) == 'E' else -1
        lats.append(lat)
        lons.append(lon)
        continue
    if i == """47123'56"N 120*12'24"W""":
        # this entry probably had 1 as a typo for *
        # other values are unreasonable for US lat/lon
        lats.append(47 + 23 / 60 + 56 / 3600)
        lons.append(-1 * (120 + 12 / 60 + 24 / 3600))
        continue
    print('NO MATCH', i)
    lats.append(np.nan)
    lons.append(np.nan)
    badparse += 1


joined['lat'] = lats
joined['lon'] = lons
joined = joined.drop('loc', axis=1)

print('# of bad lat/lon parses:', badparse)


fname = 'noaa.csv'
print('exporting cleaned noaa dataset to', fname)
joined.to_csv(fname, index=False)

fname = 'stations.pdf'
print('plotting stations with weather data on', fname)

import matplotlib as mpl
mpl.use('Agg')
# https://stackoverflow.com/questions/44488167
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import pandas as pd

lat = np.array(lats)
lon = np.array(lons)

# determine range to print based on min, max lat and lon of the data
margin = 2  # buffer to add to the range
lat_min, lon_min = 24.344974, -126.592164
lat_max, lon_max = 52.735210, -54.079873
lat_min = min(min(lat) - margin, lat_min)
lat_max = max(max(lat) + margin, lat_max)
lon_min = min(min(lon) - margin, lon_min)
lon_max = max(max(lon) + margin, lon_max)

# create map using BASEMAP
m = Basemap(llcrnrlon=lon_min,
            llcrnrlat=lat_min,
            urcrnrlon=lon_max,
            urcrnrlat=lat_max,
            lat_0=(lat_max - lat_min) / 2,
            lon_0=(lon_max - lon_min) / 2,
            projection='merc',
            resolution='h',
            area_thresh=10000.,
            )
m.drawcoastlines()
m.drawcountries()
m.drawstates()
m.drawmapboundary(fill_color='#46bcec')
m.fillcontinents(color='white', lake_color='#46bcec')
# convert lat and lon to map projection coordinates
lonss, latss = m(lon, lat)
# plot points as red dots
m.scatter(lonss, latss, marker='.', color='r', zorder=5, s=1)

plt.savefig(fname, format='pdf', bbox_inches='tight')
