import json
import urllib.request

import pandas as pd
from datetime import datetime


class MobilityApple:
    def __init__(self):
        source = self.get_link()
        self.__reports_df = pd.read_csv(source)

    @staticmethod
    def get_link():
        json_link = "https://covid19-static.cdn-apple.com/covid19-mobility-data/current/v3/index.json"
        with urllib.request.urlopen(json_link) as url:
            json_data = json.loads(url.read().decode())
        link = (
                "https://covid19-static.cdn-apple.com"
                + json_data["basePath"]
                + json_data["regions"]["en-us"]["csvPath"]
        )
        return link

    def get_mobility(self,
                     start_date=datetime(2020, 8, 31),
                     end_date=datetime(2021, 11, 18, 23, 59)):
        mobility = self.__reports_df.loc[self.__reports_df['region'] == "Prague"]
        dates = mobility.drop(['geo_type',
                               'region',
                               'transportation_type',
                               'alternative_name',
                               'sub-region',
                               'country'], axis=1)

        hourly_mobility = {}
        for index, record in dates.iterrows():
            for i, v in record.items():
                time = i.split("-")
                year = int(time[0])
                month = int(time[1])
                day = int(time[2])

                for hour in range(0, 24):
                    date = datetime(year, month, day, hour, 0)
                    if start_date < date < end_date:
                        hourly_mobility[date] = v

        return hourly_mobility


if __name__ == '__main__':
    c_date = datetime.now()
    mobility_apple = MobilityApple()
    mobility_ = mobility_apple.get_mobility(end_date=c_date)

    for date, data in mobility_.items():
        print(str(date) + " | " + str(data))
