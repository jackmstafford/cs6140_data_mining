import json
import sys
import requests
from urllib import quote

debug = 0 == 0

MAX_LIMIT = 50
default_host = 'https://api.yelp.com/'
default_path = 'v3/businesses/search'
default_api_key_file = 'api_key'
data_dir = 'data/'

def request(host, path, api_key, url_params=None):
    if url_params == None:
        url_params = {}

    url = '{}{}'.format(host, quote(path.encode('utf8')))
    headers = { 'Authorization': 'Bearer ' + api_key }
    response = requests.request('GET', url, headers=headers, params=url_params)
    return response.json()

def getAllZips(zips=[84111, 84108, 84117], num=1000):
    for z in zips:
        businessesToFile(str(z), {'location': z}, num)

def getAllLatLong(radius=1610, # in meters, about a mile
        ll=[
            [40.763392, -111.896730, 'downtown slc'], 
            [40.721480, -111.855617, 'sugar house'], 
            [40.668214, -111.824219, 'holladay'], 
            [40.646612, -111.497251, 'park city'], 
        ]):
    for a in ll:
        businessesToFile(a[2], {'radius': radius, 'latitude': a[0], 'longitude': a[1]})

def businessesToFile(filename, params, num=1000):
    api_key = getApiKey()
    busses = []
    for offset in range(0, num, MAX_LIMIT):
        limit = min(MAX_LIMIT, num - offset)
        params['limit'] = limit
        params['offset'] = offset
        if debug:
            print('Requesting {}'.format(params))
        response = request(default_host, default_path, api_key, params)
        b = response['businesses']
        if len(b) > 0:
            busses.extend(b)
        else:
            break
    print('Got {} businesses'.format(len(busses)))
    with open(data_dir + filename, 'a') as f:
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
