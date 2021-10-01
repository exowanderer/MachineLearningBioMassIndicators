import json
from satsearch import Search
from satsearch.search import SatSearchError
from utils import info_message, warning_message, debug_message

url_earth_search = 'https://earth-search.aws.element84.com/v0'

doberitz_geojson = 'doberitz_multipolygon.geojson'

with open(doberitz_geojson, 'r') as json_in:
    doberitz_feats = json.load(json_in)

eo_bbox = [-110, 39.5, -105, 40.5]
eo_datetime = '2018-02-12T00:00:00Z/2018-03-18T12:31:12Z'
eo_query = {
    'eo:cloud_cover': {'lt': 10}
}

search = Search(
    url=url_earth_search,
    bbox=eo_bbox
)
print(f'bbox search: {search.found()} items')

search = Search(
    url=url_earth_search,
    datetime=eo_datetime
)
print(f'datetime search: {search.found()} items')

search = Search(
    url=url_earth_search,
    query=eo_query
)
print(f'eo:cloud_cover search: {search.found()} items')

search = Search(
    url=url_earth_search,
    datetime=eo_datetime,
    bbox=eo_bbox,
    query=eo_query,
    collections=['sentinel-s2-l2a']
)
print(f'combined search: {search.found()} items')

for feat_ in doberitz_feats['features']:
    try:
        search = Search(
            url=url_earth_search,
            intersects=feat_['geometry'],
            # datetime=eo_datetime,
            # bbox=eo_bbox,
            # query=eo_query,
            # collections=['sentinel-s2-l2a']
        )
        print(f'combined search: {search.found()} items')
    except SatSearchError as err:
        warning_message(err)
