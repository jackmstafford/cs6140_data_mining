import json
import os
import sys
import requests
from urllib import quote

debug = 0 == 0

MAX_LIMIT = 50
default_host = 'https://api.yelp.com/'
default_path = 'v3/businesses/search'
default_api_key_file = 'api_key'
data_dir = 'data/'
min_data_fields = ['rating', 'review_count', 'price']

def mileToMeter(miles):
    return int(round(miles * 1609.34))

def readDataFromFile(filename):
    return json.load(open(filename, 'r'))

def writeDataToFile(filename, data):
    with open(filename, 'w') as f:
        json.dump(data, f)

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

def getAllLatLong(radius=mileToMeter(2), 
        ll=[
            [41.751795, -111.834216, 'logan'],
            [41.221812, -111.973482, 'downtown ogden'],
            [40.887974, -111.888002, 'bountiful'],
            [40.736785, -114.051312, 'west wendover'],
            [40.452865, -109.534870, 'vernal'],
            [40.763392, -111.896730, 'downtown slc'], 
            [40.790655, -111.990564, 'slc airport'],
            [40.721480, -111.855617, 'sugar house'], 
            [40.668214, -111.824219, 'holladay'], 
            [40.667611, -111.938796, 'taylorsville'],
            [40.617057, -111.856020, 'fort union'],
            [40.526882, -111.888008, 'draper'],
            [40.543908, -111.983608, 'daybreak'],
            [40.723076, -111.541442, 'kimball junction'],
            [40.646612, -111.497251, 'park city'], 
            [40.297056, -111.694971, 'orem'],
            [40.233678, -111.658673, 'downtown provo'],
            [38.573346, -109.550759, 'moab'],
            [37.680549, -113.061966, 'cedar city'],
            [37.123037, -113.533595, 'st george'],
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
    writeDataToFile(data_dir + filename, busses)
    print('Saved {} businesses to {}'.format(len(busses), filename))
    return busses

def getDataFilenames(dataDirectory=data_dir, getMin=False):
    filenames = []
    for filename in os.listdir(data_dir):
        sp = filename.split('.')
        if not getMin and len(sp) == 1:
                filenames.append('{}/{}'.format(data_dir, filename))
        elif getMin and len(sp) == 2:
                filenames.append('{}/{}'.format(data_dir, filename))
    return filenames

def getDataNums(getMin=False):
    numBus = 0
    filesize = 0 # in bytes
    for filename in getDataFilenames(getMin=getMin):
        numBus += len(readDataFromFile(filename))
        filesize += os.stat(filename).st_size
    return numBus, filesize

def minimizeData(dataDirectory=data_dir, minDataFields=min_data_fields, save=False):
    data = []
    for filename in getDataFilenames():
        bizList = readDataFromFile(filename)
        minBizList = []
        for biz in bizList:
            minBiz = {}
            try:
                for f in minDataFields:
                    if f == 'price': 
                        if f not in biz:
                            minBiz[f] = 0
                        else:
                            minBiz[f] = len(biz[f])
                    else:
                        minBiz[f] = biz[f]
            except KeyError:
                print('KeyError')
                for f in biz:
                    print('\t{:20}{}'.format(f, biz[f]))
                continue
            minBizList.append(minBiz)
        if save:
            writeDataToFile(filename + '.min', minBizList)
        data.append(minBizList)
    return data


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
