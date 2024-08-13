import os, requests

key = 'AIzaSyC2CxKExB2dVr9ISnz4sFe3mq379DV7lEo'
cx = '33a40a6ede1304964'
uri = 'https://www.googleapis.com/customsearch/v1'

searchTerms = ['Bicycle', 
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
for searchTerm in searchTerms:
    directory = os.path.join('data', searchTerm)
    
    if not os.path.isdir(directory):
        os.makedirs(directory)
    
    results = requests.get(uri, params={'key': key, 'cx': cx, 'searchType': 'image', 'fileType': 'png', 'q': searchTerm})

    for result in results.json()['items']:
        i = 1
        # print(f"{result}:", results.json()[result])
        # print(f"{result}:", results.json()['items'][0][result])
        # print(result['link'])
        filename = f'{searchTerm}_{i}.png'
        filepath = os.path.join(directory, filename)

        image = requests.get(result['link'])
        print(image.ok, image.status_code)
        print(result['link'])
        if image.ok or int(image.status_code) == 200:
            with open(filepath, mode='wb') as file:
                file.write(image.content)