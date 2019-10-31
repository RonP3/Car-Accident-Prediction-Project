import os
import pandas as pd


def move_to_folder(set_name):
    df = pd.read_csv("data/" + set_name + "_processed.csv", index_col=0)
    for index, row in df.iterrows():
        print(index)
        os.rename("data/map_screenshots/MapImages/" + str(index) + ".jpg",
                  "data/map_screenshots/" + set_name + "/" + str(index) + ".jpg")


def move_all_to_folders():
    sets = ['test', 'train', 'validate']
    for set in sets:
        move_to_folder(set)


def main():
    move_all_to_folders()


if __name__ == '__main__':
    main()
