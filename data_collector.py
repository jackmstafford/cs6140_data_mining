import json
import sys
import requests
from urllib import quote

debug = 0 == 0

MAX_LIMIT = 50
default_host = 'https://api.yelp.com/'
default_path = 'v3/businesses/search'
default_api_key_file = 'api_key'

def request(host, path, api_key, url_params=None):
    if url_params == None:
        url_params = {}

    url = '{}{}'.format(host, quote(path.encode('utf8')))
    headers = { 'Authorization': 'Bearer ' + api_key }
    response = requests.request('GET', url, headers=headers, params=url_params)
    return response.json()

def makeABunch(zips=[84111, 84108, 84117]):
    for z in zips:
        businessesToFile(zip_code=z)

def businessesToFile(num=1000, zip_code=84112):
    api_key = getApiKey()
    busses = []
    for offset in range(0, num, MAX_LIMIT):
        limit = min(MAX_LIMIT, num - offset)
        params = {
                'location': zip_code,
                'limit': limit,
                'offset': offset,
                }
        if debug:
            print('Sending request {}'.format(params))
        response = request(default_host, default_path, api_key, params)
        b = response['businesses']
        if len(b) > 0:
            busses.extend(b)
        else:
            break
    print('Got {} businesses'.format(len(busses)))
    with open(str(zip_code), 'a') as f:
        #f.write(busses)
        json.dump(busses, f)
    return busses


def getApiKey(api_key_filename=default_api_key_file):
    with open(api_key_filename) as f:
        api_key = f.read()
    return api_key


def main(api_key_filename=default_api_key_file, host=default_host, path=default_path):
    params = {
            'location': 84112,
            'limit': 3,
            }
    api_key = getApiKey(api_key_filename)
    req = request(host, path, api_key, params)
    return req

if __name__ == '__main__':
    if len(sys.argv) > 2:
        print('At most one argument, the API key file, should be provided')
        exit()
    if len(sys.argv) == 2:
        print(main(sys.argv[1]))
    else:
        print(main())
