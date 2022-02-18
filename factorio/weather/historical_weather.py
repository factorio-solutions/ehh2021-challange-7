from datetime import datetime
import matplotlib.pyplot as plt
from meteostat import Point, Hourly


class HistoricalWeather:
    def __init__(self,
                 location=Point(50.101, 14.26)):
        self.__location = location

    def get_temperature(self,
                        start_date=datetime(2020, 8, 31),
                        end_date=datetime(2021, 11, 18, 23, 59)):
        start_date = start_date.replace(tzinfo=None)
        end_date = end_date.replace(tzinfo=None)
        data = Hourly(self.__location, start_date, end_date)
        data = data.fetch()

        return data


# 50.101, 14.26, 380 (location of Prague)
if __name__ == '__main__':
    historical_weather = HistoricalWeather()

    start_date_ = datetime(2020, 8, 31)
    end_date_ = datetime(2021, 11, 18, 23, 59)
    data_ = historical_weather.get_temperature(start_date_, end_date_)

    # temperature in celsius
    # pressure in hPa

    data_.plot(y=['temp'])
    data_.plot(y=['pres'])
    plt.show()
