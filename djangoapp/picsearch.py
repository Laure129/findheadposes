#!/usr/bin/env python
import requests
import urllib.request
import random

def image_bing_search(query, count):
    url = 'https://api.cognitive.microsoft.com/bing/v5.0/images/search'
    # query string parameters
    payload = {'q': query, 'count': count}
    # custom headers
    headers = {'Ocp-Apim-Subscription-Key': '77f4c31c2d0944249bea98e3610ff853'}
    # make GET request
    r = requests.get(url, params=payload, headers=headers)
    # get JSON response
    a_list = []
    for i in r.json().get('value'):
        a_list.append(i.get('contentUrl'))
    return a_list

def save_image(image_list, directory=None):
    image_list2 = []
    for i in image_list:
        try:
            image = str(random.randint(100000,1000000))+ '.jpg'
            image_list2.append(image)
            urllib.request.urlretrieve(i, r'C:\Users\Laure\repos\findheadposes\findheadposes\media' + '\\' + image)
        except:
            continue
    return image_list2
