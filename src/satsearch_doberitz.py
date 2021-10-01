import json
import os

from dotenv import load_dotenv

from satsearch import Search
from satsearch.search import SatSearchError
from utils import info_message, warning_message, debug_message

load_dotenv('.env')

url_earth_search = os.environ.get('STAC_API_URL')

doberitz_geojson = 'doberitz_multipolygon.geojson'

with open(doberitz_geojson, 'r') as json_in:
    doberitz_feats = json.load(json_in)

eo_bbox = [-110, 39.5, -105, 40.5]
eo_datetime = '2020-01-01/2021-01-01'
eo_query = {
    'eo:cloud_cover': {'lt': 10}
}

for feat_ in doberitz_feats['features'][:1]:
    try:
        search = Search(
            url=url_earth_search,
            # intersects=feat_['geometry'],
            bbox=eo_bbox,
            datetime=eo_datetime,
            query=eo_query,
            collections=['sentinel-s2-l2a'],
            limit=2
        )
        print(f'combined search: {search.found()} items')

        items = search.items()
        print(items.summary())

        # filenames = items.download_assets(
        #     filename_template='assets/${date}/${id}',
        #     requester_pays=True
        # )
        # print(filenames)

    except SatSearchError as err:
        warning_message(err)
