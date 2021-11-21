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
    num_inducing = 32
    num_iter = 1000
    num_particles = 32
    loader_batch_size = 512
    slow_mode = True  # enables checkpointing and logging

    time_now = datetime.datetime.utcnow()
    parser = argparse.ArgumentParser()

    path_parser = parser.add_argument('-c', '--config', type=Path, default='config.ini',
                                      help='Set path to your config.ini file.')
    path_parser = parser.add_argument('-i', '--input', type=Path, default='mnt/model_state.pth',
                                      help='Set path to save trained model.')

    args = parser.parse_args()
    if not args.config.exists():
        raise argparse.ArgumentError(path_parser, f"Config file doesn't exist! Invalid path: {args.config} "
                                                  f"to config.ini file, please check it!")
    load_path = args.input

    hack_config = data_loader.HackConfig.from_config(args.config)
    dfactory = data_loader.OnlineFactory(data_frequency=hack_config.data_frequency,
                                         teams=hack_config.teams,
                                         hospital=hack_config.hospital,
                                         data_folder=hack_config.data_folder,
                                         dtype=dtype)

    dfactory_d = data_loader.DataFactory(data_frequency=hack_config.data_frequency,
                                         teams=hack_config.teams,
                                         hospital=hack_config.hospital,
                                         data_folder=hack_config.data_folder,
                                         dtype=dtype)

    ora = Oracle(load_path, dfactory)
    c_date = datetime.datetime.now()
    df = get_past_prediction(ora.model, ora.dsfactory, c_date, to_past=168 * 2)
    # data = dfactory.create_timestamp()
    fig, ax = plt.subplots(figsize=(20, 16))
    n = 10
    # data.loc[datetime.datetime.strptime(df.index[0], "%Y-%m-%d %H-%M-%S"): datetime.datetime.strptime(df.index[-3], "%Y-%m-%d %H-%M-%S")]
    # sns.barplot(x=df.index, y=df['Arrivals Hourly Rate'], ax=ax)
    df['Arrivals Hourly Rate'].plot(ax=ax, kind='bar')
    # kde_data = data.loc[datetime.datetime.strptime(df.index[0], "%Y-%m-%d %H-%M-%S"):]
    # ax.plot(kde_data['cases'].rolling(window=10).mean(), 'r', linewidth=3)

    ticks = ax.xaxis.get_ticklocs()
    ticklabels = [l.get_text() for l in ax.xaxis.get_ticklabels()]
    ax.xaxis.set_ticks(ticks[::n])
    ax.xaxis.set_ticklabels(ticklabels[::n])
    plt.xticks(rotation=45, ha='right')
    plt.show()
    #
    # model = LogNormGPpl.load_model(load_path)
    #
    # test_x = dfactory.dset[-200:][0]
    # Y = dfactory.dset[-200:][1]
    # x_plt = torch.arange(Y.size(0)).detach().cpu()
    # model.eval()
    # with torch.no_grad():
    #     output = model(test_x)
    #
    # # Similarly get the 5th and 95th percentiles
    # lat_samples = output.rsample(torch.Size([30])).exp()
    # samples_expanded = model.gp.likelihood(lat_samples).sample(torch.Size([30]))
    # samples = samples_expanded.view(samples_expanded.size(0) * samples_expanded.size(1), -1)
    #
    # # Similarly get the 5th and 95th percentiles
    # # samples = model.gp.likelihood(output.mean).rsample(torch.Size([1000]))
    # lower, fn_mean, upper = percentiles_from_samples(lat_samples, [.001, 0.5, 0.8])
    # # lower, upper = output.confidence_region()
    # # fn_mean = output.mean.exp()
    #
    # y_sim_lower, y_sim_mean, y_sim_upper = percentiles_from_samples(samples, [.001, 0.5, 0.8])
    #
    # # visualize the result
    # fig, (ax_func, ax_samp) = plt.subplots(1, 2, figsize=(12, 3))
    # line = ax_func.plot(
    #     x_plt, fn_mean.detach().cpu(), label='GP prediction')
    # ax_func.fill_between(
    #     x_plt, lower.detach().cpu().numpy(),
    #     upper.detach().cpu().numpy(), color=line[0].get_color(), alpha=0.5
    # )
    # ax_func.legend()
    #
    # ax_samp.scatter(x_plt, Y, alpha=0.5,
    #                 label='True train data', color='orange')
    # y_sim_plt = ax_samp.plot(x_plt, y_sim_mean.cpu(
    # ).detach(), alpha=0.5, label='Sample mean from the model')
    # ax_samp.fill_between(
    #     x_plt, y_sim_lower.detach().cpu(),
    #     y_sim_upper.detach().cpu(), color=y_sim_plt[0].get_color(), alpha=0.5
    # )
    # ax_samp.legend()
    # plt.show()
    #
    # print(f'Done')
