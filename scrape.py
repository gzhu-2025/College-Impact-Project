import os, request

key = 'AIzaSyC2CxKExB2dVr9ISnz4sFe3mq379DV7lEo'
cx = '33a40a6ede1304964'
uri = 'https://www.googleapis.com/customsearch/'

requests = request.get(uri, params={'key': key, 'cx': cx, 'searchType': 'image', 'q': 'Car'})

print(requests.json())