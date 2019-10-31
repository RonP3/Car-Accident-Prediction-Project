from resources.lsoaDataHandler import CoordinateTranslator, PopulationDensityLoader, DeprevationLoader
from resources.foursquareDataHandler import Places
import pandas as pd
import numpy as np
import time

MAP_GRID_FILE = "map_grid.csv"
LSOA_DATA_FILE = "lsoa_data.csv"
ROAD_ACCIDENTS_ORIGINAL_DATA_FILE = "dftRoadSafetyData_Accidents_2018.csv"
ACCIDENTS_PER_SMALL_BOX_DATA_FILE = "accidents_per_small_box_data.csv"
FOURSQUARE_DATA_FILE = "foursquare_data"
ACCIDENTS_IN_BIG_BOX_DATA_FILE = 'accidents_in_big_box.csv'
RELATIVE_PATH_TO_DATA = "data/"
DF_CHUNK_SIZE = 500

ID = 'id'
BOTTOM_LEFT_LNG = 'bottom_left_longitude'
BOTTOM_LEFT_LAT = 'bottom_left_latitude'
TOP_RIGHT_LNG = 'top_right_longitude'
TOP_RIGHT_LAT = 'top_right_latitude'
CENTER_LSOA_NAME = 'center_lsoa_name'
CENTER_LSOA_CODE = 'center_lsoa_code'
POPULATION = 'population'
POPULATION_DENSITY = 'population_density'
CRIME_DOM = 'crime_dom'
EDUCATION_DEP = 'education_dep'
EMPLOYMENT_DEP = 'employment_dep'
HEALTH_DEP = 'health_dep'
INCOME_DEP = 'income_dep'
LIVING_ENVIRONMENT_DEP = 'living_environment_dep'


class Categories:
    school = '4bf58dd8d48988d13b941735'
    bars = '4bf58dd8d48988d116941735'
    arts_and_entertainment = '4d4b7104d754a06370d81259'


class DataGatherer:
    def __init__(self):
        self.coordinate_translator = CoordinateTranslator()
        self.population_density_loader = PopulationDensityLoader()
        self.deprevation_loader = DeprevationLoader()
        self.places_loader = Places()

    def __split_df(self, df, chunks):
        return [df[i::chunks] for i in range(chunks)]

    def gather_lsoa_data(self):
        ct = CoordinateTranslator()
        pdl = PopulationDensityLoader()
        dl = DeprevationLoader()

        map_grid = pd.read_csv(RELATIVE_PATH_TO_DATA + MAP_GRID_FILE, index_col=0)
        block_chunks = self.__split_df(map_grid, np.math.ceil(map_grid.shape[0] / DF_CHUNK_SIZE))
        map_grid[CENTER_LSOA_NAME] = np.nan
        map_grid[CENTER_LSOA_CODE] = np.nan
        map_grid[POPULATION] = np.nan
        map_grid[POPULATION_DENSITY] = np.nan
        map_grid[CRIME_DOM] = np.nan
        map_grid[EDUCATION_DEP] = np.nan
        map_grid[EMPLOYMENT_DEP] = np.nan
        map_grid[HEALTH_DEP] = np.nan
        map_grid[INCOME_DEP] = np.nan
        map_grid[LIVING_ENVIRONMENT_DEP] = np.nan

        for chunk in block_chunks:
            center_lng = (chunk[BOTTOM_LEFT_LNG] + chunk[TOP_RIGHT_LNG])/2.0
            center_lat = (chunk[BOTTOM_LEFT_LAT] + chunk[TOP_RIGHT_LAT])/2.0
            lsoa_name = ct.translate(center_lng.values, center_lat.values)
            map_grid.loc[chunk.index, CENTER_LSOA_NAME] = lsoa_name
            lsoa_code, pop, pop_density = pdl.read_by_lsoa_name(lsoa_name)
            map_grid.loc[chunk.index, CENTER_LSOA_CODE] = lsoa_code
            map_grid.loc[chunk.index, POPULATION] = pop
            map_grid.loc[chunk.index, POPULATION_DENSITY] = pop_density

            map_grid.loc[chunk.index, CRIME_DOM] = dl.find_crime_domain(lsoa_code)
            map_grid.loc[chunk.index, EDUCATION_DEP] = dl.find_education_deprivation(lsoa_code)
            map_grid.loc[chunk.index, EMPLOYMENT_DEP] = dl.find_employment_deprivation(lsoa_code)
            map_grid.loc[chunk.index, HEALTH_DEP] = dl.find_health_deprivation(lsoa_code)
            map_grid.loc[chunk.index, INCOME_DEP] = dl.find_income_deprivation(lsoa_code)
            map_grid.loc[chunk.index, LIVING_ENVIRONMENT_DEP] = dl.find_living_environment_deprivation(lsoa_code)
            map_grid.to_csv(RELATIVE_PATH_TO_DATA + LSOA_DATA_FILE)

    def gather_foursquare_data(self, category_id, category_name, from_index=0):
        start_time = time.time()
        map_grid = pd.read_csv(RELATIVE_PATH_TO_DATA + MAP_GRID_FILE, index_col=0)
        results_limit = 50
        squares_num = len(map_grid.index)
        map_grid = map_grid[from_index:]
        map_grid[category_name] = np.nan
        chunk_size = 0
        total_requests = 0
        for index, row in map_grid.iterrows():
            if index < from_index:
                index += 1
                continue
            sw = str(row['bottom_left_latitude']) + ', ' + str(row['bottom_left_longitude'])
            ne = str(row['top_right_latitude']) + ', ' + str(row['top_right_longitude'])
            places_num, status = self.places_loader.count_places_in_box(sw, ne, category_id, results_limit)
            total_requests += 1
            if status != 200:
                print('foursquare error: ', status)
                break
            map_grid.loc[index, category_name] = places_num
            time.sleep(0.6)
            if chunk_size == 1000 or index == squares_num - 1:
                map_grid.to_csv(RELATIVE_PATH_TO_DATA + FOURSQUARE_DATA_FILE + '_' + category_name + '.csv')
                print("--- %s seconds (foursquare) ---" % (time.time() - start_time))
                print('until now: ', total_requests)
                chunk_size = 0
            chunk_size += 1

    def get_accidents_in_big_box(self, bottom_left_latitude, bottom_left_longitude, top_right_latitude, top_right_longitude):
        start_time = time.time()
        relevant_fields = ['Accident_Index', 'Longitude', 'Latitude', 'Accident_Severity', 'Number_of_Vehicles',
                           'Number_of_Casualties', 'Date', 'Day_of_Week', 'Time']
        accidents_df = pd.read_csv(RELATIVE_PATH_TO_DATA + ROAD_ACCIDENTS_ORIGINAL_DATA_FILE,
                                   usecols=relevant_fields)
        filtered_df = pd.DataFrame(columns=relevant_fields)
        for accident_index, accident_row in accidents_df.iterrows():
            if accident_index % 10000 == 0:
                print('accidents until now:', accident_index)
            accident_longitude = accident_row['Latitude']
            accident_latitude = accident_row['Longitude']
            if bottom_left_latitude <= accident_longitude <= top_right_latitude and \
                    bottom_left_longitude <= accident_latitude <= top_right_longitude:
                accident_index = accident_row['Accident_Index']
                severity = accident_row['Accident_Severity']
                vehicles = accident_row['Number_of_Vehicles']
                casualties = accident_row['Number_of_Casualties']
                date = accident_row['Date']
                day = accident_row['Day_of_Week']
                accident_time = accident_row['Time']
                accident_data = {
                    'Accident_Index': accident_index,
                    'Longitude': accident_latitude,
                    'Latitude': accident_longitude,
                    'Accident_Severity': severity,
                    'Number_of_Vehicles': vehicles,
                    'Number_of_Casualties': casualties,
                    'Date': date,
                    'Day_of_Week': day,
                    'Time': accident_time
                }
                filtered_df = filtered_df.append(accident_data, ignore_index=True)
        filtered_df.to_csv(RELATIVE_PATH_TO_DATA + ACCIDENTS_IN_BIG_BOX_DATA_FILE)
        print("--- %s seconds ---" % (time.time() - start_time))

    def gather_road_safety_data(self):
        start_time = time.time()
        map_grid = pd.read_csv(RELATIVE_PATH_TO_DATA + MAP_GRID_FILE, index_col=0)
        accidents_df = pd.read_csv(RELATIVE_PATH_TO_DATA + ACCIDENTS_IN_BIG_BOX_DATA_FILE, index_col=0)
        map_grid['accidents_num'] = 0
        map_grid['accidents_severity_avg'] = 0
        map_grid['accidents_severity_sum'] = 0
        map_grid['number_of_vehicles'] = 0
        map_grid['number_of_casualties'] = 0
        assigned_accidents = 0
        for grid_index, grid_row in map_grid.iterrows():
            is_accident_latitude_in_box = accidents_df['Latitude'].between(grid_row['bottom_left_latitude'],
                                                                           grid_row['top_right_latitude'],
                                                                           inclusive=True)
            is_accident_longitude_in_box = accidents_df['Longitude'].between(grid_row['bottom_left_longitude'],
                                                                             grid_row['top_right_longitude'],
                                                                             inclusive=True)
            is_accident_in_box = is_accident_latitude_in_box & is_accident_longitude_in_box
            accident_in_box_indexes = is_accident_in_box[is_accident_in_box].index.values.tolist()
            for accident_index in accident_in_box_indexes:
                accident_row = accidents_df.iloc[[accident_index]]
                map_grid.loc[grid_index, 'accidents_num'] += 1
                map_grid.loc[grid_index, 'number_of_vehicles'] += accident_row['Number_of_Vehicles'].values[0]
                map_grid.loc[grid_index, 'number_of_casualties'] += accident_row['Number_of_Casualties'].values[0]
                map_grid.loc[grid_index, 'accidents_severity_sum'] += accident_row['Accident_Severity'].values[0]
                accidents_num = map_grid.iloc[grid_index]['accidents_num']
                severity_sum = map_grid.iloc[grid_index]['accidents_severity_sum']
                map_grid.loc[grid_index, 'accidents_severity_avg'] = severity_sum / accidents_num
                assigned_accidents += 1
                # accidents_df.drop(accident_index, inplace=True)
                if assigned_accidents % 10000 == 0:
                    print('assigned accidents: ', assigned_accidents)
                    print("--- %s seconds ---" % (time.time() - start_time))
        print("--- %s seconds - FINISH_TIME ---" % (time.time() - start_time))
        map_grid.to_csv(RELATIVE_PATH_TO_DATA + ACCIDENTS_PER_SMALL_BOX_DATA_FILE)


def main():
    dg = DataGatherer()
    dg.gather_lsoa_data()
    dg.gather_foursquare_data(category_id=Categories.arts_and_entertainment, category_name='arts_and_entertainment')
    dg.get_accidents_in_big_box(50.959878, -2.528671, 52.862297, 0.489857)
    dg.gather_road_safety_data()


if __name__ == '__main__':
    main()
