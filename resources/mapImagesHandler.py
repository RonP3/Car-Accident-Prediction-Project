from resources.APIKeys import get_here_app_id, get_here_app_code
import requests
import time
import pandas as pd

MAP_GRID_FILE = 'data/map_grid.csv'
RELATIVE_PATH_TO_DEST_FOLDER = 'D:\\MapImages\\'


class Images:
    def __init__(self):
        self.app_id = get_here_app_id()
        self.app_code = get_here_app_code()
        self.url = 'https://image.maps.api.here.com/mia/1.6/mapview'
        self.max_attempts = 5

    def get_image(self, box_index, top_right_latitude, top_right_longitude,
                  bottom_left_latitude, bottom_left_longitude):
        params = dict(
            bbox=str(top_right_latitude) + ',' + str(top_right_longitude) + ',' +
                 str(bottom_left_latitude) + ',' + str(bottom_left_longitude),
            app_id=self.app_id,
            app_code=self.app_code
        )
        attempts = 0
        status = 200
        while attempts < self.max_attempts:
            resp = requests.get(url=self.url, params=params)
            if resp.status_code == 200:
                path = RELATIVE_PATH_TO_DEST_FOLDER + str(box_index) + ".jpg"
                with open(file=path, mode='wb') as f:
                    f.write(resp.content)
            else:
                print('Image request error: ', status)
                attempts += 1
                time.sleep(1)
                continue
            break
        if attempts >= self.max_attempts:
            print('Max attempts!!')
            return status
        return status


def main():
    images = Images()
    map_grid = pd.read_csv(MAP_GRID_FILE, index_col=0)
    for grid_index, grid_row in map_grid.iterrows():
        top_right_latitude = grid_row['top_right_latitude']
        top_right_longitude = grid_row['top_right_longitude']
        bottom_left_latitude = grid_row['bottom_left_latitude']
        bottom_left_longitude = grid_row['bottom_left_longitude']
        status = images.get_image(grid_index, top_right_latitude, top_right_longitude,
                                  bottom_left_latitude, bottom_left_longitude)
        if status != 200:
            print('HERE api error: ', status)
            print('index error: ', grid_index)
            print('coordinates error: ', top_right_latitude, top_right_longitude,
                  bottom_left_latitude, bottom_left_longitude)
            break
        time.sleep(0.5)


if __name__ == "__main__":
    main()
