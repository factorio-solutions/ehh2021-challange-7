import datetime
import glob
from pathlib import Path
import warnings
from factorio.mobility.mobility_apple import MobilityApple
from factorio.mobility.mobility_google import MobilityGoogle
from factorio.mobility.mobility_waze import MobilityWaze
from factorio.utils.hack_config import HackConfig
from factorio.web_scraping.football import Football

warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset

import os

from factorio.weather import ActualWeather, HistoricalWeather

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class DataFactory:
    def __init__(self, data_frequency, hospital, data_folder, dtype=torch.float):
        self.hospital = hospital
        self.data_frequency = data_frequency
        self.covid_source = r'https://onemocneni-aktualne.mzcr.cz/api/v2/covid-19/'
        self.data_folder = data_folder
        self.scaler = MinMaxScaler()
        self.end_date = datetime.datetime(2021, 11, 15, 23, 55, 0)
        self.start_date = datetime.datetime(2017, 1, 1, 3)
        self.weather_columns = ['temp', 'pres']  # ['temp', 'rhum', 'pres']

        self.google_m = MobilityGoogle()
        self.apple_m = MobilityApple()
        self.waze_m = MobilityWaze()

        self.dset = self.create_timestamp(dtype=dtype)

    def create_timestamp(self, dtype=torch.float):
        time_data = self.__load_ikem_data()

        tmp_array = np.full((time_data.shape[0], 1), 0)
        time_data.insert(0, 'cases', tmp_array)
        hour_rate = time_data.resample(f'{self.data_frequency}min').count().loc[
                    self.start_date:self.end_date]
        # end_date = pd.to_datetime(hour_rate.index.values[-1])
        x = self.load_weather(start_date=self.start_date,
                              end_date=self.end_date).to(dtype=dtype)
        self.cases = hour_rate
        y = torch.as_tensor(hour_rate['cases'].values).to(dtype=dtype)
        return TensorDataset(x, y)

    def load_weather(self, start_date, end_date):
        historical_weather = HistoricalWeather()

        data = historical_weather.get_temperature(start_date, end_date)
        data.fillna(0, inplace=True)
        data = data.resample(f'{self.data_frequency}min').ffill()
        selected_data = data[self.weather_columns]
        selected_data.insert(0, 'hour', selected_data.index.hour)
        selected_data.insert(1, 'day in week', selected_data.index.weekday)
        selected_data.insert(2, 'month', selected_data.index.month)

        google, apple, waze = self.__load_mobility(start_date, end_date)

        selected_data = selected_data.join(google[['retail_and_recreation_percent_change_from_baseline',
                                                   'residential_percent_change_from_baseline']])

        selected_data = selected_data.join(waze['waze'])
        selected_data = selected_data.join(apple['apple'])
        selected_data.fillna(0, inplace=True)
        self.data = selected_data
        self.scaler.fit(selected_data.values)
        transformed_values = self.scaler.transform(selected_data.values)
        return torch.as_tensor(transformed_values)

    def __load_mobility(self, start_date, end_date):
        google = pd.DataFrame.from_dict(self.google_m.get_mobility(), orient='index').sort_index()
        google = google[start_date:end_date]

        apple = pd.DataFrame.from_dict(self.apple_m.get_mobility(), orient='index').sort_index()
        apple = apple[start_date:end_date]

        waze_source = pd.DataFrame.from_dict(self.waze_m.get_mobility(), orient='index').sort_index()
        waze = waze_source[start_date:end_date]
        if waze.empty:
            waze = waze_source[-1 - (end_date - start_date).days * 24:]
        apple.fillna(0, inplace=True)
        waze = waze.resample(f'{self.data_frequency}min').ffill()
        apple = apple.resample(f'{self.data_frequency}min').ffill()
        google = google.resample(f'{self.data_frequency}min').ffill()
        waze.columns = ['waze']
        apple.columns = ['apple']
        return google, apple, waze

    def __load_ikem_data(self):
        all_files = glob.glob(str(self.data_folder / 'ikem' / "*.xlsx"))
        li = []

        for filename in all_files:
            df = pd.read_excel(filename)
            df['datum a čas'] = pd.to_datetime(df['datum a čas'])
            df.set_index('datum a čas', inplace=True)
            li.append(df)
        this_year = pd.read_html(str(self.data_folder / 'vypis_9074.xls'))[0]
        this_year['datum a čas'] = pd.to_datetime(this_year['datum a čas'])
        this_year.set_index('datum a čas', inplace=True)
        this_year = this_year.drop('Unnamed: 8', axis=1)
        this_year = this_year[this_year['důvod'] != 'kardioverze']
        li.append(this_year['2021-01-01':])
        return pd.concat(li, axis=0)

    def get_min_max(self):
        return self.dset[:][0].min(dim=0)[0].tolist(), self.dset[:][0].max(dim=0)[0].tolist()

    def inverse_transform(self, X: torch.Tensor):
        return self.scaler.inverse_transform(X.numpy())

    def get_future_data(self, hour: int = 2, dtype=torch.float):
        c_date = datetime.datetime.now()
        h_weather = HistoricalWeather()
        to_past = 24 - hour
        index = pd.date_range(start=c_date - datetime.timedelta(hours=to_past),
                              end=c_date + datetime.timedelta(hours=hour),
                              freq=f"{self.data_frequency}min")
        index = [pd.to_datetime(date) for date in index]
        google, apple, waze = self.__load_mobility(index[0] - datetime.timedelta(days=7),
                                                   index[-1] - datetime.timedelta(days=7))
        data = h_weather.get_temperature(c_date - datetime.timedelta(hours=to_past),
                                         c_date + datetime.timedelta(hours=hour))
        df = data[self.weather_columns]
        df.insert(0, 'hour', df.index.hour)
        df.insert(1, 'day in week', df.index.weekday)
        df.insert(2, 'month', df.index.month)
        df.reset_index(drop=True, inplace=True)
        google.reset_index(drop=True, inplace=True)
        apple.reset_index(drop=True, inplace=True)
        waze.reset_index(drop=True, inplace=True)
        df = df.join(google[['retail_and_recreation_percent_change_from_baseline',
                             'residential_percent_change_from_baseline']])

        df = df.join(waze['waze'])
        df = df.join(apple['apple'])
        return torch.as_tensor(self.scaler.transform(df.values)).to(dtype)


class OnlineFactory:
    def __init__(self, data_frequency, hospital, data_folder, dtype=torch.float):
        self.hospital = hospital
        self.data_frequency = data_frequency
        self.covid_source = r'https://onemocneni-aktualne.mzcr.cz/api/v2/covid-19/'
        self.data_folder = data_folder
        self.scaler = MinMaxScaler()
        self.end_date = datetime.datetime(2021, 11, 15, 23, 55, 0)
        self.start_date = datetime.datetime(2017, 1, 1, 3)
        self.weather_columns = ['temp', 'pres']  # ['temp', 'rhum', 'pres']

        self.google_m = MobilityGoogle()
        self.apple_m = MobilityApple()
        self.waze_m = MobilityWaze()

    def __load_mobility(self, start_date, end_date):
        google = pd.DataFrame.from_dict(self.google_m.get_mobility(), orient='index').sort_index()
        google = google[start_date:end_date]

        apple = pd.DataFrame.from_dict(self.apple_m.get_mobility(), orient='index').sort_index()
        apple = apple[start_date:end_date]

        waze_source = pd.DataFrame.from_dict(self.waze_m.get_mobility(), orient='index').sort_index()
        waze = waze_source[start_date:end_date]
        if waze.empty:
            waze = waze_source[-1 - (end_date - start_date).days * 24:]
        apple.fillna(0, inplace=True)
        waze = waze.resample(f'{self.data_frequency}min').ffill()
        apple = apple.resample(f'{self.data_frequency}min').ffill()
        google = google.resample(f'{self.data_frequency}min').ffill()
        waze.columns = ['waze']
        apple.columns = ['apple']
        return google, apple, waze

    def get_prediction_data(self, c_date, to_future: int = 2, to_past: int = 24, dtype=torch.float):
        h_weather = HistoricalWeather()
        index = pd.date_range(start=c_date - datetime.timedelta(hours=to_past),
                              end=c_date + datetime.timedelta(hours=to_future),
                              freq=f"{self.data_frequency}min")
        index = [pd.to_datetime(date) for date in index]

        google, apple, waze = self.__load_mobility(index[0] - datetime.timedelta(days=7),
                                                   index[-1] - datetime.timedelta(days=7))
        data = h_weather.get_temperature(c_date - datetime.timedelta(hours=to_past),
                                         c_date + datetime.timedelta(hours=to_future))
        df = data[self.weather_columns]
        df.insert(0, 'hour', df.index.hour)
        df.insert(1, 'day in week', df.index.weekday)
        df.insert(2, 'month', df.index.month)
        df.reset_index(drop=True, inplace=True)
        google.reset_index(drop=True, inplace=True)
        apple.reset_index(drop=True, inplace=True)
        waze.reset_index(drop=True, inplace=True)
        df = df.join(google[['retail_and_recreation_percent_change_from_baseline',
                             'residential_percent_change_from_baseline']])

        df = df.join(waze['waze'])
        df = df.join(apple['apple'])
        return torch.as_tensor(np.nan_to_num(self.scaler.transform(df.values))).to(dtype)

    def create_timestamp(self, dtype=torch.float):
        time_data = self.__load_ikem_data()

        tmp_array = np.full((time_data.shape[0], 1), 0)
        time_data.insert(0, 'cases', tmp_array)
        return time_data

    def __load_ikem_data(self):
        all_files = glob.glob(str(self.data_folder / 'ikem' / "*.xlsx"))
        li = []

        for filename in all_files:
            df = pd.read_excel(filename)
            df['datum a čas'] = pd.to_datetime(df['datum a čas'])
            df.set_index('datum a čas', inplace=True)
            li.append(df)
        this_year = pd.read_html(str(self.data_folder / 'vypis_9074.xls'))[0]
        this_year['datum a čas'] = pd.to_datetime(this_year['datum a čas'])
        this_year.set_index('datum a čas', inplace=True)
        this_year = this_year.drop('Unnamed: 8', axis=1)
        li.append(this_year['2021-01-01':])
        return pd.concat(li, axis=0)


def load_data(data_path):
    data = pd.read_json(data_path, lines=True)
    return data


if __name__ == '__main__':
    import argparse

    time_now = datetime.datetime.utcnow()
    parser = argparse.ArgumentParser()

    path_parser = parser.add_argument('-c', '--config', type=Path, default='config.ini',
                                      help='Set path to your config.ini file.')

    args = parser.parse_args()
    if not args.config.exists():
        raise argparse.ArgumentError(path_parser, f"Config file doesn't exist! Invalid path: {args.config} "
                                                  f"to config.ini file, please check it!")

    hack_config = HackConfig.from_config(args.config)
    data_loader = DataFactory(hack_config.data_frequency,
                              hospital=hack_config.hospital,
                              data_folder=hack_config.data_folder)
    print(data_loader.get_min_max())
    future = data_loader.get_future_data()
    print(future.size())
    print(future)
