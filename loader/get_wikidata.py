import requests
import csv
import json
import os
import io
from PIL import Image


# load categories file
categories_file = './Places365/categories_places365.txt'

# files to save
dataframe_file_path = 'results/wikidata_artwork.csv'
metadata_file_path = 'results/wikidata_metadata.csv'

# image download directory
output_directory = './Artwork/wikidata'


Image.MAX_IMAGE_PIXELS = 2000000000

def get_qid_for_category(category_name):

    singleqid = get_qid_for_words(category_name)
    if singleqid:
        return singleqid
    else:
        words = category_name.split()
        for i in range(len(words), 0, -1):
            # Try to find QID for the last i words and the rest substring
            qid_last_i_words = get_qid_for_words(words[-i:])
            qid_rest_substring = get_qid_for_words(words[:-i])

            if qid_last_i_words and qid_rest_substring:
                return [qid_last_i_words, qid_rest_substring]

        # If no QID is found for any combination, try finding QID for each individual word
        qids_for_individual_words = [get_qid_for_words([word]) for word in words]
        return qids_for_individual_words


def get_qid_for_words(category_name):
    endpoint_url = "https://www.wikidata.org/w/api.php"
    params = {
        'action': 'wbsearchentities',
        'format': 'json',
        'language': 'en',
        'type': 'item',
        'search': category_name,
    }

    response = requests.get(endpoint_url, params=params)
    data = response.json()

    if 'search' in data and data['search']:
        return data['search'][0]['id']
    else:
        return None


def process_queries(label_index, max_images_per_category = 10):
    endpoint_url = "https://query.wikidata.org/sparql"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'application/json',
    }
    qids = []
    wikidata = []
    metadata = []
    id = 0  # Used to create unique filenames

    for l in label_index:
        category_name = l['label']
        qid = get_qid_for_category(category_name)

        print(f"The QID for {category_name} is {qid}")
        qids.append({'label': category_name, 'index': l['index'], 'qid': qid})

        if isinstance(qid, list):
            # qid_str = ' && '.join([f'?category = wd:{qid_val}' for qid_val in qid])
            # category_filter = qid_str
            category_vars = [f'?category{i+1}' for i in range(len(qid))]
            category_patterns = [f'{category_var} = wd:{qid_val}' for category_var, qid_val in zip(category_vars, qid)]
            category_filter = ' && '.join(category_patterns)
            query_patterns = [f'?item wdt:P180 {category_var};' for category_var in category_vars]
            query_patterns_str = '\n'.join(query_patterns)
            
        else:
            category_filter = f"?category = wd:{qid}"
            query_patterns_str = f"?item wdt:P180 ?category;"

        sparql_query = f"""
                SELECT ?item ?itemLabel ?image
                WHERE {{
                    {query_patterns_str}
                    ?item wdt:P31 wd:Q3305213;
                          wdt:P18 ?image.
                    FILTER ({category_filter})
                    SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
                }}
                LIMIT {max_images_per_category}
                """

        params = {'query': sparql_query, 'format': 'json'}
        response = requests.get(endpoint_url, params=params, headers=headers)
        try:
            data = response.json()
        except json.decoder.JSONDecodeError:
            print(f"Failed to decode JSON response for QID(s) {qid}. Response content: {response.text}")
            continue


        for result in data['results']['bindings']:
            image_url = result['image']['value']
            item_label = result['itemLabel']['value']
            filename = f"Artwork_wikidata_{id:08d}.jpg"
            download_image(image_url, filename)
            id += 1

            print(image_url, filename)
            wikidata.append({'Image Filename': filename, 'Label': l['index']})
            metadata.append({'Image Filename': filename, 'Item Label':item_label,'URL': image_url, 'Label': l['label']})


    # dataframe_file_path = 'results/wikidata_artwork.csv'
    with open(dataframe_file_path, 'w', newline='') as csvfile:
        fieldnames = ['Image Filename', 'Label']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for row in wikidata:
            writer.writerow(row)

    # metadata_file_path = 'results/wikidata_metadata.csv'
    with open(metadata_file_path, 'w', newline='') as csvfile_metadata:
        fieldnames_metadata = ['Image Filename', 'Item Label', 'URL', 'Label']
        writer_metadata = csv.DictWriter(csvfile_metadata, fieldnames=fieldnames_metadata)

        writer_metadata.writeheader()
        for row in metadata:
            writer_metadata.writerow(row)


def download_image(image_url, filename):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    cookies = {'cookie_name': 'cookie_value'}

    response = requests.get(image_url, headers=headers, cookies=cookies, verify=True)
    # output_directory = './Artwork/wikidata'   # './Artwork/wikidata'
    os.makedirs(output_directory, exist_ok=True)

    if response.status_code == 200:
        image_filename = filename
        image_path = os.path.join(output_directory, image_filename)
        # with open(image_path, 'wb') as f:
        #     f.write(response.content)

        img = Image.open(io.BytesIO(response.content)).convert('RGB')
        max_size = (800, 600)
        img.thumbnail(max_size)
        img.save(image_path)

        # print(f"Downloaded image: {image_filename}")
    else:
        print(f"Failed to download image for {filename}")


if __name__ == '__main__':

    # label_file = './Places365/categories_places365.txt'
    with open(categories_file, 'r') as file:
        label_index = [{'label': ' '.join(line.split()[0].split('/')[2:]).replace('_', ' '),
                        'index': int(line.split()[-1])} for line in file.readlines()]

    process_queries(label_index, max_images_per_category = 100)
