import datetime
from pathlib import Path

import pandas as pd
import torch
import pickle

from factorio.gpmodels.gplognormpl import LogNormGPpl
from factorio.utils import data_loader
from factorio.utils.helpers import percentiles_from_samples
import plotly.express as px


class Oracle:
    def __init__(self,
                 model_path,
                 dsfactory: data_loader.OnlineFactory) -> None:
        self.dsfactory = dsfactory
        self.model = LogNormGPpl.load_model(model_path)

        with open('mnt/scaler.pkl', 'rb') as fid:
            self.dsfactory.scaler = pickle.load(fid)
        self.model.eval()


def get_current_prediction(model, dsfactory, horizont: int = 2):
    c_date = datetime.datetime.now()
    current_data = dsfactory.get_future_data(c_date, horizont)
    to_past = 23 - horizont
    index = pd.date_range(start=c_date - datetime.timedelta(hours=to_past),
                          end=c_date + datetime.timedelta(hours=horizont),
                          freq=f"{60}min")

    with torch.no_grad():
        output = model(current_data)
    rate_samples = output.rsample(torch.Size([1000])).exp()

    percentile, = percentiles_from_samples(rate_samples, [0.8])
    return pd.DataFrame(percentile,  # np.abs(np.random.randn(24, 1)),
                        columns=['Arrivals Hourly Rate'],
                        index=[pd.to_datetime(date) for date in index]
                        )


def get_past_prediction(model, dsfactory, past_date, horizont: int = 2, to_past: int = 168):
    current_data = dsfactory.get_past_data(past_date, horizont, to_past=to_past)
    index = pd.date_range(start=past_date - datetime.timedelta(hours=to_past - 1),
                          end=past_date + datetime.timedelta(hours=horizont),
                          freq=f"{60}min")

    with torch.no_grad():
        output = model(current_data)
    rate_samples = output.rsample(torch.Size([1000])).exp()

    percentile, = percentiles_from_samples(rate_samples, [0.8])
    tmp_df = pd.DataFrame(percentile,
                          columns=['Arrivals Hourly Rate'],
                          index=[pd.to_datetime(date).strftime("%Y-%m-%d %H-%M-%S") for date in index]
                          )
    return tmp_df


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import seaborn as sns
    import argparse

    # Move to config at some point
    dtype = torch.float

    time_now = datetime.datetime.utcnow()
    parser = argparse.ArgumentParser()

    path_parser = parser.add_argument('-c', '--config', type=Path, default='config.ini',
                                      help='Set path to your config.ini file.')

    args = parser.parse_args()
    if not args.config.exists():
        raise argparse.ArgumentError(path_parser, f"Config file doesn't exist! Invalid path: {args.config} "
                                                  f"to config.ini file, please check it!")

    hack_config = data_loader.HackConfig.from_config(args.config)
    load_path = hack_config.model_path
    # dfactory_online = data_loader.OnlineFactory(data_frequency=hack_config.data_frequency,
    #                                      teams=hack_config.teams,
    #                                      hospital=hack_config.hospital,
    #                                      data_folder=hack_config.data_folder,
    #                                      dtype=dtype)

    dfactory = data_loader.DataFactory(data_frequency=hack_config.data_frequency,
                                         teams=hack_config.teams,
                                         hospital=hack_config.hospital,
                                         data_folder=hack_config.data_folder,
                                         dtype=dtype)

    model = LogNormGPpl.load_model(hack_config.model_path)

    show = 2500
    test_x = dfactory.dset[-show:][0]
    real_x = dfactory.inverse_transform(test_x)
    Y = dfactory.dset[-show:][1]
    x_plt = dfactory.data[-show:].index.values

    pressure = dfactory.data[-show:]['pres']
    pressure_zeroed = pressure-min(pressure)
    pressure_norm = pressure_zeroed/max(pressure_zeroed)

    model.eval()
    with torch.no_grad():
        output = model(test_x)

    # Similarly get the 5th and 95th percentiles
    lat_samples = output.rsample(torch.Size([30])).exp()
    samples_expanded = model.gp.likelihood(lat_samples).sample(torch.Size([30]))
    samples = samples_expanded.view(samples_expanded.size(0) * samples_expanded.size(1), -1)

    # Similarly get the 5th and 95th percentiles
    lower, fn_mean, upper = percentiles_from_samples(lat_samples, [.001, 0.5, 0.8])

    y_sim_lower, y_sim_mean, y_sim_upper = percentiles_from_samples(samples, [.001, 0.5, 0.8])

    # visualize the result
    fig, (ax_func, ax_samp) = plt.subplots(1, 2, figsize=(12, 3))
    line = ax_func.plot(
        x_plt, fn_mean.detach().cpu(), label='Arrivals rate')
    ax_func.fill_between(
        x_plt, lower.detach().cpu().numpy(),
        upper.detach().cpu().numpy(), color=line[0].get_color(), alpha=0.1
    )
    ax_func.plot(x_plt, pressure_norm, label='Pressure MinMax transformed')
    ax_func.legend()

    ax_samp.scatter(x_plt, Y, alpha=0.5,
                    label='Arrivals in hour - actual', color='orange')
    y_sim_plt = ax_samp.plot(x_plt, y_sim_mean.cpu(
    ).detach(), alpha=0.5, label='Arrivals in hour - most likely')
    ax_samp.fill_between(
        x_plt, y_sim_lower.detach().cpu(),
        y_sim_upper.detach().cpu(), color=y_sim_plt[0].get_color(), alpha=0.1
    )
    
    ax_samp.plot(x_plt, pressure_norm, label='Pressure MinMax transformed')
    ax_samp.legend()
    fig.tight_layout()
    plt.show()

    print(f'Done')
