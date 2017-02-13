Data copied from [COGP repo](https://github.com/trungngv/cogp/tree/master/data/weather).

This data contains weather information collected by sensors at 4 different locations in England.
It is retrieved for the period from 01/07/2013 - 15/07/2013 from the websites

* http://www.cambermet.co.uk/
* http://www.bramblemet.co.uk/
* http://www.sotonmet.co.uk/
* http://www.chimet.co.uk/

`*x.csv` contains the time for a sensor. `*y.csv` contains four columns, for `WSPD,WD,GST,ATMP` or wind speed in knots, wind direction in degrees, max gust in knots, and air temperature in celcius, repsectively. Missing data is flagged as -1.

Nguyen and Bonilla 2014 only use the fourth column ATMP over the time period `[10, 15]`. Test imputation is performed on `[10.2, 10.8]` in `cam` and `[13.5, 14.2]` in `chi`.

