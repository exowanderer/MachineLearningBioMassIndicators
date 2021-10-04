import json
import os

from argparse import ArgumentParser
from dotenv import load_dotenv

from satsearch import Search
from satsearch.search import SatSearchError

# TODO: change from utils to .utils when modularizing
from utils import info_message, warning_message, debug_message

load_dotenv('.env')

args = ArgumentParser()
args.add_argument('--download', action='store_true')
clargs = args.parse_args()

flag_download = clargs.download

url_earth_search = os.environ.get('STAC_API_URL')
doberitz_geojson = 'doberitz_multipolygon.geojson'

with open(doberitz_geojson, 'r') as json_in:
    doberitz_feats = json.load(json_in)

eo_datetime = '2021-09-01/2021-10-01'
eo_query = {
    'eo:cloud_cover': {'lt': 10}
}

for feat_ in doberitz_feats['features']:
    try:
        search = Search(
            url=url_earth_search,
            intersects=feat_['geometry'],
            # bbox=eo_bbox,
            datetime=eo_datetime,
            query=eo_query,
            collections=['sentinel-s2-l2a']
            # limit=2
        )
        print(f'Combined search: {search.found()} items')

        items = search.items()
        print(items.summary())

        if flag_download:
            # NIR Band
            # b08_filenames = items.download_assets(requester_pays=True)
            b08_filenames = items.download(
                'B08',
                # filename_template='assets/${date}/${id}',
                requester_pays=True
            )
            print(b08_filenames)

            # RED Band
            b04_filenames = items.download(
                'B04',
                # filename_template='assets/${date}/${id}',
                requester_pays=True
            )
            print(b04_filenames)

    except SatSearchError as err:
        warning_message(err)
