import os, requests

key = 'AIzaSyC2CxKExB2dVr9ISnz4sFe3mq379DV7lEo'
cx = '33a40a6ede1304964'
uri = 'https://www.googleapis.com/customsearch/v1'
# uri = 'https://www.googleapis.com/customsearch/v1/siterestrict'


folders = ['Bicycle', 
            'Bridge', 
            'Bus', 
            'Car', 
            'Chimney', 
            'Crosswalk', 
            'Hydrant', 
            'Motorcycle', 
            'Palm', 
            'Stair', 
            'Traffic Light', 
            ]

searchTerms = ['cyclist', 
               'suspension bridge', 
               'bus', 
               'car', 
               'roof with chimney', 
               'crosswalk', 
               'hydrant', 
               'motorcycle', 
               'palm tree', 
               'flight of stairs', 
               'traffic light', 
               ]


for i in range(len(folders)):
    params = {'key': key, 
              'cx': cx, 
              'searchType': 'image', 
            #   'filter': '1', 
            #   'imgColorType': 'trans', 
              'fileType': 'png', 
              'num': 10, 
              'q': searchTerms[i]
             }

    directory = os.path.join('data', folders[i])
    
    if not os.path.isdir(directory):
        os.makedirs(directory)
    
    results = requests.get(uri, params=params)
    # print(results.json())
    # assert results.json()['error']['code'] != 429, 'daily request limit exceeded'
    j = 1

    for result in results.json()['items']:
        # print(f"{result}:", results.json()[result])
        # print(f"{result}:", results.json()['items'][0][result])
        # print(result['link'])
        filename = f'{folders[i]}_{j}.png'
        filepath = os.path.join(directory, filename)

        image = requests.get(result['link'])
        print(image.ok, image.status_code)
        print(result['link'])
        if image.ok and int(image.status_code) == 200:
            with open(filepath, mode='wb') as file:
                file.write(image.content)
            j += 1
        