# This function will take a list of lens_ids and return bibliographic data, family_id, lens_is in the family,
# date_published, and claims
# It ensures that scroll is implemented to get the full dataset

import json
import requests
import time
import os

def lens_id_extract(lens_token:str,path_input:str,path_output_json:str):      #I have used here the same structure for the API and the same function of len_init_extract_missing_ids, but probably not the best way
    """
    This function takes a list of lens_ids from a file and retrieves their details from the Lens API.
    It processes the data in chunks to avoid hitting API limits and saves the results to a specified JSON file.

    Parameters:
        lens_token (str): The Lens API token for authentication.
        path_input (str): Path to the input file containing lens_ids, one per line.
        path_output_json (str): Path to the output JSON file where results will be saved.
    """

    with open(path_input, 'r') as f:
        missing_ids = f.read().splitlines()

    max_chunk_size = 10000
    for i in range(0,len(missing_ids),max_chunk_size):
        process_chunk = missing_ids[i:i+max_chunk_size]

        base_path_output, ext = os.path.splitext(path_output_json)
        i_output_path = f"{base_path_output}_{i}{ext}"

        lens_id_extract_subset(lens_token, process_chunk, i_output_path)
    return


def  lens_id_extract_subset(lens_token, process_chunk, path_output_json):
    print('Requesting API response for missing ids')
    headers = {
        'Authorization': 'Bearer ' + lens_token,
        'Content-Type': 'application/json'
    }
    include = ["lens_id", "jurisdiction", "doc_number", "kind", "date_published","publication_type", "biblio", "families", "claims", "abstract"]
    request_body = {
        "query": {
            "terms": {
                "lens_id": process_chunk
            }
        },
        "scroll": "3m",
        "include": include,
        "size": 100
    }
    data_json = json.dumps(request_body)
    URL = 'https://api.lens.org/patent/search'
    response = requests.post(URL, data=data_json, headers=headers)

    if response.status_code != requests.codes.ok:
        print(f'Error: {response.status_code}')
        print(response.text)
        return None
    else:
        response_json = json.loads(response.text)
        print(f'Total number of patents found: {response_json["total"]}. Ready to process {response_json["results"]} patents at a time.')
        response_data_init = response_json['data']

        scroll_id = response.json()['scroll_id']
        response_data_fin = scroll(lens_token, scroll_id,path_output_json, response_data_init)

        with open(path_output_json, "w") as f:
            json.dump(response_data_fin, f, indent=4)
        print(f"API response from patents_ids has been successfully saved to {path_output_json}.")
        return

def scroll(lens_token, scroll_id,path_output, response_data_init):
    URL = 'https://api.lens.org/patent/search'
    headers = {
    'Authorization': 'Bearer ' + lens_token,
    'Content-Type': 'application/json'
    }

    include = ["lens_id", "jurisdiction", "doc_number", "kind", "date_published", "publication_type", "biblio",
               "families", "claims"]

    if scroll_id is not None:
        request_body = {
            "scroll_id": scroll_id,
            "include": include,
            "scroll":"3m"
        }
    data_json = json.dumps(request_body)
    response = requests.post(URL, data=data_json, headers=headers)

    if response.status_code == requests.codes.too_many_requests:
        print('Rate limit reached, pausing for 30 seconds.')
        time.sleep(30)
        return scroll(lens_token, scroll_id,path_output, response_data_init)

    elif response.status_code == 204:
        print('Finished scrolling')
        scroll_id = None
        return response_data_init

    elif response.status_code != requests.codes.ok:
        print(f'Error: {response.status_code}')
        scroll_id = None
        return response_data_init

    else:
        scroll_id = response.json()['scroll_id']  # Extract the new scroll id from response
        response_data_fin = response_data_init + json.loads(response.text)['data']
        print('Executed scroll, appended data')
        with open(path_output, "w") as f:
            json.dump(response_data_fin, f, indent=4)
            print(f"Intermediate API response has been successfully saved to {path_output}.")
        return scroll(lens_token, scroll_id,path_output, response_data_fin,)