import pandas as pd
import requests as requests


class MobilityTomTom:
    def __init__(self, api_key='prague'):
        self._api_key = api_key
        self._base_api_url = "https://api.midway.tomtom.com/ranking/dailyStats/"
        self.__reports_df = self.get_report()

    def get_report(self, ):
        api_url = self._base_api_url + self._api_key
        response = requests.get(api_url)
        return pd.DataFrame(response.json())
