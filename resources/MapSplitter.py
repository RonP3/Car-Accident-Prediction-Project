from enum import Enum
from numpy import arange
import csv


class LongitudeCoordinateOffset(Enum):
    BLOCK_OFFSET_1KM = 0.014362403810407
    BLOCK_OFFSET_2KM = 0.028724806521931
    BLOCK_OFFSET_5KM = 0.071811997074957
    BLOCK_OFFSET_10KM = 0.143623856793856


class LatitudeCoordinateOffset(Enum):
    BLOCK_OFFSET_1KM = 0.0089831310685
    BLOCK_OFFSET_2KM = 0.0179662621371
    BLOCK_OFFSET_5KM = 0.0449156553427
    BLOCK_OFFSET_10KM = 0.0898313106854


def split_map(bottom_left_x, bottom_left_y, top_right_x, top_right_y, lng_offset: LongitudeCoordinateOffset,
              lat_offset: LatitudeCoordinateOffset):
    x_values = list(arange(bottom_left_x, top_right_x, lng_offset.value))
    y_values = list(reversed(arange(bottom_left_y, top_right_y, lat_offset.value)))

    with open('data/map_grid.csv', mode='w', newline='') as csv_file:
        fieldnames = ['id', 'bottom_left_longitude', 'bottom_left_latitude', 'top_right_longitude',
                      'top_right_latitude']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        curr_id = 0
        for i in range(len(x_values)-1):
            for j in range(len(y_values)-1):
                row = {
                    'id': curr_id,
                    'bottom_left_longitude': x_values[i],
                    'bottom_left_latitude': y_values[j + 1],
                    'top_right_longitude': x_values[i + 1],
                    'top_right_latitude': y_values[j]
                }
                writer.writerow(row)
                curr_id += 1


def main():
    split_map(-2.528671, 50.959878, 0.489857, 52.862297, LongitudeCoordinateOffset.BLOCK_OFFSET_1KM,
              LatitudeCoordinateOffset.BLOCK_OFFSET_1KM)
    return


if __name__ == '__main__':
    main()
