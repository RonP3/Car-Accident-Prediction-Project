import pandas as pd
import requests
import json
import time

POPULATION_DENSITY_FILE_NAME = "lsoa_population_density.csv"
EDUCATION_DEPRIVATION_FILE_NAME = "education_dep.csv"
CRIME_DOMAIN_FILE_NAME = "crime_dom.csv"
EMPLOYMENT_DEPRIVATION_FILE_NAME = "employment_dep.csv"
HEALTH_DEPRIVATION_FILE_NAME = "health_dep.csv"
INCOME_DEPRIVATION_FILE_NAME = "income_dep.csv"
LIVING_ENVIRONMENT_DEPRIVATION_FILE_NAME = "living_environment_dep.csv"

RELATIVE_PATH_TO_DATA = "data/"
# api-endpoint
REVERSE_GEOCODE_URL = "https://api.postcodes.io/postcodes"


class CoordinateTranslator:
    def __init__(self, url=REVERSE_GEOCODE_URL, max_attempts=5, max_coordinates_per_request=100):
        self.url = url
        self.max_attempts = max_attempts
        self.max_coordinates_per_request = max_coordinates_per_request

    def _chunks(self, l, n):
        for i in range(0, len(l), n):
            yield l[i:i + n]

    def translate(self, longitudes, latitudes):
        assert len(longitudes) == len(latitudes)
        longitude_chunks = self._chunks(longitudes, self.max_coordinates_per_request)
        latitude_chunks = self._chunks(latitudes, self.max_coordinates_per_request)

        lsoa_list = []

        for longitude_chunk, latitude_chunk in zip(longitude_chunks, latitude_chunks):
            attempts = 0
            while attempts < self.max_attempts:
                geolocations = []
                for lng, lat in zip(longitude_chunk, latitude_chunk):
                    geolocation = {
                        "longitude": lng,
                        "latitude": lat,
                        "limit": 1,
                        "wideSearch": 1
                    }
                    geolocations.append(geolocation)
                data = {
                    'geolocations': geolocations
                }
                headers = {'content-type': 'application/json'}
                r = requests.post(self.url, json.dumps(data), headers=headers)
                post_return = json.loads(r.content)
                status = post_return["status"]

                if status != 200:
                    attempts += 1
                    time.sleep(1)
                    continue

                for parital_result in post_return["result"]:
                    lsoa_list.append(parital_result["result"][0]["lsoa"])
                break
            if attempts >= self.max_attempts:
                print("Too Many Attempts!")
        return lsoa_list


class PopulationDensityLoader:
    def __init__(self):
        self.data_by_code = pd.read_csv(RELATIVE_PATH_TO_DATA + POPULATION_DENSITY_FILE_NAME, index_col=0)
        self.data_by_name = pd.read_csv(RELATIVE_PATH_TO_DATA + POPULATION_DENSITY_FILE_NAME, index_col=1)

    def read_by_lsoa_code(self, lsoa_code):
        val = self.data_by_code.loc[lsoa_code]
        return val["Name"].values, val["Population"].values, val["Density"].values

    def read_by_lsoa_name(self, lsoa_name):
        val = self.data_by_name.loc[lsoa_name]
        return val["Code"].values, val["Population"].values, val["Density"].values


class DeprevationLoader:
    def __init__(self):
        self.crime_dom = pd.read_csv(RELATIVE_PATH_TO_DATA + CRIME_DOMAIN_FILE_NAME, index_col=0)
        self.education_dep = pd.read_csv(RELATIVE_PATH_TO_DATA + EDUCATION_DEPRIVATION_FILE_NAME, index_col=0)
        self.employment_dep = pd.read_csv(RELATIVE_PATH_TO_DATA + EMPLOYMENT_DEPRIVATION_FILE_NAME, index_col=0)
        self.health_dep = pd.read_csv(RELATIVE_PATH_TO_DATA + HEALTH_DEPRIVATION_FILE_NAME, index_col=0)
        self.income_dep = pd.read_csv(RELATIVE_PATH_TO_DATA + INCOME_DEPRIVATION_FILE_NAME, index_col=0)
        self.living_environment_dep = pd.read_csv(RELATIVE_PATH_TO_DATA + LIVING_ENVIRONMENT_DEPRIVATION_FILE_NAME, index_col=0)

    def find_crime_domain(self, lsoa_code):
        return self.crime_dom.loc[lsoa_code]["Score"].values

    def find_education_deprivation(self, lsoa_code):
        return self.education_dep.loc[lsoa_code]["Score"].values

    def find_employment_deprivation(self, lsoa_code):
        return self.employment_dep.loc[lsoa_code]["Score"].values

    def find_health_deprivation(self, lsoa_code):
        return self.health_dep.loc[lsoa_code]["Score"].values

    def find_income_deprivation(self, lsoa_code):
        return self.income_dep.loc[lsoa_code]["Score"].values

    def find_living_environment_deprivation(self, lsoa_code):
        return self.living_environment_dep.loc[lsoa_code]["Score"].values


def main():
    ct = CoordinateTranslator()
    pdl = PopulationDensityLoader()
    dl = DeprevationLoader()

    lsoa_list = ct.translate([0.629834723775309, -0.134096], [51.7923246977375, 51.516701])
    code, pop, density = pdl.read_by_lsoa_name(lsoa_list)
    print(dl.find_crime_domain(code))
    print(dl.find_education_deprivation(code))
    return


if __name__ == '__main__':
    main()
