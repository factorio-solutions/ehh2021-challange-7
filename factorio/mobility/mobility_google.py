import glob
import io
import os
import zipfile

import pandas as pd
from datetime import datetime, timedelta

import requests as requests
from bs4 import BeautifulSoup


class MobilityGoogle:
    def __init__(self,
                 zip_file_url=r'https://www.gstatic.com/covid19/mobility/Region_Mobility_Report_CSVs.zip',
                 extract_folder=''):
        self.zip_file_url = zip_file_url
        self.download_file()
        self.extract_folder = extract_folder
        df = None

        reports = glob.glob('mnt/*_CZ_Region_Mobility_Report.csv')
        for count, report in enumerate(reports):
            if count == 0:
                df = pd.read_csv(reports[count])
            else:
                df_2 = pd.read_csv(reports[count])
                data = [df, df_2]
                df = pd.concat(data)

        self.__reports_df = df

    def get_df(self):
        df = self.__reports_df.loc[self.__reports_df['sub_region_1'] == "Prague"]

        df1 = df[['date',
                  'retail_and_recreation_percent_change_from_baseline',
                  'grocery_and_pharmacy_percent_change_from_baseline',
                  'parks_percent_change_from_baseline',
                  'transit_stations_percent_change_from_baseline',
                  'workplaces_percent_change_from_baseline',
                  'residential_percent_change_from_baseline']]

        return df1.sort_index()

    def get_mobility(self,
                     start_date=datetime(2020, 8, 31),
                     end_date=datetime.now()):

        mobility = self.get_df()

        hourly_mobility = {}
        for index, mob in mobility.iterrows():
            time = mob['date'].split("-")
            year = int(time[0])
            month = int(time[1])
            day = int(time[2])

            for hour in range(0, 24):
                date = datetime(year, month, day, hour, 0)
                if start_date < date < end_date:
                    hourly_mobility[date] = mob.iloc[1:]

        return hourly_mobility

    def download_file(self, save_dir='mnt/'):
        r = requests.get(self.zip_file_url)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        cz_data = [f for f in z.filelist if 'CZ_Region_Mobility_Report' in f.filename]
        [z.extract(csv, save_dir) for csv in cz_data]
        return True


if __name__ == '__main__':
    mobility_google = MobilityGoogle()
    mobility_google.download_file()
    df_ = mobility_google.get_df()
    mobility_ = mobility_google.get_mobility()

    for date, data in mobility_.items():
        print(str(date) + " | " + str(data))
