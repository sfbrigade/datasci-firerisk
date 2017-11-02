import pandas as pd
import shapefile

def pip(x, y, poly):
# function that determines if x,y coordinates are within a polygon
# function from here:   http://geospatialpython.com/2011/01/point-in-polygon.html

    n = len(poly)
    inside = False
    p1x, p1y = poly[0]
    for i in range(n + 1):
        p2x, p2y = poly[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xints = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


def tract_id(x, y, r):
    #with each geocode, loops through each of the tract shapes to determine to which tract the location belongs
    for i in range(len(r.shapes())):
        if pip(x, y, r.shape(i).points):
            #item 3 contains the tract info.
            return r.record(i)[3]
    #if not found return none
    return None

r = shapefile.Reader("geo_export_ba2a7c3c-9fd2-4383-947f-399256d6ad60.shp")

k = pd.read_csv('masterdf_20170920.csv', low_memory=False, )
#convert text coordinates to x,y floats
k['yx'] = k.Location_y.apply(lambda x: x[1:-1].split(','))
k['x'] = k.yx.apply(lambda x: float(x[1]))
k['y'] = k.yx.apply(lambda x: float(x[0]))

k['tract'] = k.apply(lambda cols: tract_id(cols['x'], cols['y'], r), axis=1)
#save data to new file with additional tract information
k.to_csv('masterdf_inc_census_tract.csv')
