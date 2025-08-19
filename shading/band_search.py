"""
*** INCOMPLETE SCRIPT ***
Band search for shading model

This script is used to distribute the model over 
a glacier by calculating the median of shading effects
(sky-view factor and a shading timeseries that includes
which hours the point is shaded) for several points in
each elevation band.

1. Split the glacier into elevation bands approximately X meters high
2. Determines N sample points from the area within that elevation band
3. Calculates the sky-view factor and shaded timeseries for each point
4. Finds the median sky-view factor and shaded timeseries for the band
5. Stores the shading file and band information

@author: clairevwilson
"""
def get_bands():
    """
    Creates the elevation bands on which to find
    mean sky view factor and shaded array 
    """
    bands = [1000,1500,2500,3000]
    return bands

def get_points(bands):
    """
    Finds random cells within that elevation band
    """
    for band in bands:
        print('filter the dem by band[0] - band[-2]')
        # returns lat/lon points within the elevation band
    return

def search_band(band):
    # loop through each point in the elevation band
    # run the shading model
    # store the results as temporary files
    # find the median result
    # save under the median elevation of the result
    return