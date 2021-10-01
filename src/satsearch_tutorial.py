
from satsearch import Search

url_earth_search = 'https://earth-search.aws.element84.com/v0'

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
    query=eo_query
)
print(f'eo:cloud_cover search: {search.found()} items')
