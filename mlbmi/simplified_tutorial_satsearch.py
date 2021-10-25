"""Tutorial runthrough for full SatSearch operations"""
import os
from satsearch import Search

url_earth_search = 'https://earth-search.aws.element84.com/v0'

eo_bbox = [-110, 39.5, -105, 40.5]
eo_datetime = '2018-02-12T00:00:00Z/2018-03-18T12:31:12Z'
eo_query = {
    'eo:cloud_cover': {'lt': 10}
}

search = Search(
    url=url_earth_search,
    datetime=eo_datetime,
    bbox=eo_bbox,
    query=eo_query,
    collections=['sentinel-s2-l2a-cogs']
)
print(f'combined search: {search.found()} items')

items = search.items()
print(items.summary())

filenames = items.download_assets(requester_pays=True)
