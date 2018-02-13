import json
import sys
import requests
from urllib import quote

def request(host, path, api_key, url_params=None):
    if url_params == None:
        url_params = {}

    url = '{}{}'.format(host, quote(path.encode('utf8')))
    headers = { 'Authorization': 'Bearer ' + api_key }
    response = requests.request('GET', url, headers=headers, params=url_params)
    return response.json()

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Exactly one argument, the API key file, should be provided')
        exit()
    with open(sys.argv[1]) as f:
        api_key = f.read()
    req = request('https://api.yelp.com/', 'v3/businesses/search', api_key, {'location': 84112})
    print(req)
