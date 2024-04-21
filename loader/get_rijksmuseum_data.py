import requests
import pandas as pd
from tqdm import tqdm
import re
from urllib.parse import quote

# Define API key and base URL
_API_KEY =  ' '
_BASE_URL = 'https://www.rijksmuseum.nl/api/nl/collection'



def download_img(url, fn):
    r = requests.get(url, allow_redirects=True)
    if r.status_code == 200:
        open(f'im/{fn}', 'wb').write(r.content)
    else:
        print(f"Failed to download {url}. Status code: {r.status_code}")
    open(f'im/{fn}', 'wb').write(r.content)


def extract_date(longTitle):
    years = re.findall(r'\d{4}', longTitle)
    if len(years) == 1:
        return years[0], years[0]
    elif len(years) == 2:
        return years[0], years[1]
    else:
        # raise Exception
        return 0, 0


def collection_query(query_string):
    query_url = f'{_BASE_URL}?key={_API_KEY}&{query_string}'
    r = requests.get(query_url)
    return r.json()


def process_results(results_json):
    entries = [e for e in results_json['artObjects']]
    d = []


    for e in tqdm(entries):
        if not e['hasImage'] or not e['permitDownload']:
            continue
        dl_url = e['webImage']['url']
        fn = e['webImage']['guid'] + '.jpg'

        try:
            download_img(dl_url, fn)
        except Exception as ex:
            print(f"Error downloading image: {ex}")
            continue

        earliest, latest = extract_date(e['longTitle'])

        d.append({
            'File Name': fn,
            'Artist': e['principalOrFirstMaker'],
            'Title': e['title'],
            'Earliest Date': int(earliest),
            'Latest Date': int(latest),
            'Photo Archive': 'Rijksmuseum',
            'Details URL': e['links']['web'],
            'Image Credits': dl_url
        })

    return pd.DataFrame(d)


# process a single query
def process_query(query_string):
    results = collection_query(query_string)
    return process_results(results)


# process multiple queries
def process_queries(label_index, technique_labels, max):
    results_dfs = []
    rijksmuseum = [] # List to store image filenames and their corresponding labels
    for l in label_index:
        print(l['label'])

        query = f'q={l["label"]}&imgonly=True&material=paneel&ps={max}'
        results = process_query(query)
        results_dfs.append(results)
        if not results.empty:
            rijksmuseum.extend([(fn, l["index"]) for fn in results['File Name']])

        query = f'q={l["label"]}&imgonly=True&material=olieverf&ps={max - len(results_dfs)}'
        results = process_query(query)
        results_dfs.append(results)
        if not results.empty:
            rijksmuseum.extend([(fn, l["index"]) for fn in results['File Name']])

        query = f'q={l["label"]}&imgonly=True&type=miniatuur&ps={max - len(results_dfs)}'
        results = process_query(query)
        results_dfs.append(results)
        if not results.empty:
            rijksmuseum.extend([(fn, l["index"]) for fn in results['File Name']])

    return pd.concat(results_dfs), pd.DataFrame(rijksmuseum, columns=['Image Filename', 'Label'])


type = ['miniatuur','prent']#'schilderij'
material = ['paneel','olieverf']#'doek'
technique = ['brush', 'etching', 'engraving', 'drypointdelete','color+woodcut'] #'pen',
Para = '&imgonly=True&ondisplay=True&ps=100'
max_images = 100

# Load category
with open('./Places365/categories_places365.txt', 'r') as file:

    label_index = [{'label': ' '.join(line.split()[0].split('/')[2:]).replace('_', ' '), 'index': int(line.split()[-1])} for line in file.readlines()]


results_df, image_info_df = process_queries(label_index, technique, max_images)
results_df.to_csv('rijks_meta.csv', index=False)
image_info_df.to_csv('rijksmuseum.csv', index=False)

