from resources.APIKeys import get_my_foursquare_client_id, get_my_foursquare_client_secret
import json
import requests
import time
import pandas as pd

MAP_GRID_PATH = 'data/map_grid.csv'


class Places:
    def __init__(self):
        self.client_id = get_my_foursquare_client_id()
        self.client_secret = get_my_foursquare_client_secret()
        self.url = 'https://api.foursquare.com/v2/venues/search'
        self.max_attempts = 5

    def count_places_in_box(self, sw, ne, category_id, res_limit):
        # Bounding box is up to 10,000 square km
        # se and ne format: '51.571141, -0.196196'
        params = dict(
            client_id=self.client_id,
            client_secret=self.client_secret,
            v='20190425',
            intent='browse',
            sw=sw,
            ne=ne,
            limit=res_limit,
            categoryId=category_id
        )
        attempts = 0
        status = 200
        places_num = 0
        while attempts < self.max_attempts:
            resp = requests.get(url=self.url, params=params)
            data = json.loads(resp.text)
            status = data['meta']['code']
            if status != 200:
                print('Place request error: ', status)
                if status == 429:
                    return 0, 429
                attempts += 1
                time.sleep(1)
                continue
            places_num = len(data['response']['venues'])
            break
        if attempts >= self.max_attempts:
            print('Max attempts!!')
        return places_num, status

    def count_places_in_radius(self, coordinates, radius, category_id, res_limit):
        # The maximum radius is 100,000 meters
        # Radius in meters
        params = dict(
            client_id=self.client_id,
            client_secret=self.client_secret,
            v='20190425',
            ll=coordinates,
            intent='browse',
            radius=radius,
            limit=res_limit,
            categoryId=category_id
        )
        attempts = 0
        status = 200
        places_num = 0
        while attempts < self.max_attempts:
            resp = requests.get(url=self.url, params=params)
            data = json.loads(resp.text)
            status = data['meta']['code']
            if status != 200:
                print('Place request error: ', status)
                attempts += 1
                time.sleep(1)
                continue
            places_num = len(data['response']['venues'])
            break
        if attempts >= self.max_attempts:
            print('Max attempts!!')
        return places_num, status


class SquaresLoader:
    def __init__(self):
        self.squares = pd.read_csv(MAP_GRID_PATH, index_col=0)



def main():
    start_time = time.time()
    f_places = Places()
    # sw = '51.558146, -0.228093'
    # ne = '51.571141, -0.196196'
    # category = '4bf58dd8d48988d116941735'
    # for i in range(0, 2):
    #     restaurants_num, status = f_places.count_places_in_box(sw, ne, category, 50)
    #     if status == 429:
    #         print('Limit error!!!')
    #         break
    #     print(restaurants_num)
    #     time.sleep(0.6)

if __name__ == "__main__":
    main()
