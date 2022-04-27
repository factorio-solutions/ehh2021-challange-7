import datetime
from pathlib import Path
from typing import Tuple
from gpytorch.distributions.multivariate_normal import MultivariateNormal

import pandas as pd
from pandas.core.indexes.datetimes import DatetimeIndex
import torch
import pickle
import logging
import pytz

from torch import distributions

from factorio.gpmodels.gplognormpl import LogNormGPpl
from factorio.utils import data_loader
from factorio.utils.helpers import percentiles_from_samples

logger = logging.getLogger('run_load_model.py')


class Oracle:
    def __init__(self,
                 model_path,
                 dsfactory: data_loader.OnlineFactory) -> None:
        self.dsfactory = dsfactory
        self.model = LogNormGPpl.load_model(model_path)

        with open('learnedmodels/scaler.pkl', 'rb') as fid:
            self.dsfactory.scaler = pickle.load(fid)
        self.model.eval()

    def get_current_prediction(self, to_future: int = 2):
        return get_current_prediction(self.model, self.dsfactory, to_future)

    def get_past_prediction(self, past_date, to_future: int = 2, to_past: int = 168):
        return get_past_prediction(self.model, self.dsfactory, past_date, to_future, to_past)

    def get_arrival_prob_mass(self,
                              n_arrivals=5,
                              to_future: int = 2,
                              to_past: int = 21,
                              now: datetime.datetime = None):
        if now is None:
            now = datetime.datetime.now()
        now = now.replace(second=0, microsecond=0, minute=0)
        current_data = self.dsfactory.get_prediction_data(
            now, to_future=to_future, to_past=to_past)
        datalen = current_data.size(0)
        index = pd.date_range(start=now - datetime.timedelta(hours=to_past),
                              end=now + datetime.timedelta(hours=to_future),
                              freq=f"{60}min")

        with torch.no_grad():
            posterior = self.model.predict(current_data)

        query = torch.arange(n_arrivals).expand(
            datalen, n_arrivals).permute(1, 0)
        probs = posterior.log_prob(query).detach().exp().T.numpy()

        return pd.DataFrame(probs,
                            columns=[str(arrivals.item())
                                     for arrivals in torch.arange(n_arrivals)],
                            index=index
                            )

    def get_arrival_rates(self,
                          to_future: int = 2,
                          to_past: int = 21,
                          now: datetime.datetime = None) -> pd.DataFrame:
        latent, index = self.get_rates_distribution(to_future=to_future,
                                                    to_past=to_past,
                                                    now=now)
        mu = latent.mean.exp()
        stddev = latent.stddev.exp()
        result = torch.stack([mu, stddev]).detach().T.numpy()

        return pd.DataFrame(result,
                            columns=['mean', 'stddev'],
                            index=[pd.to_datetime(date) for date in index]
                            )

    def get_rates_distribution(self,
                               to_future: int = 2,
                               to_past: int = 21,
                               now: datetime.datetime = None) -> Tuple[MultivariateNormal, DatetimeIndex]:
        if now is None:
            now = datetime.datetime.now()
        now = now.replace(second=0, microsecond=0, minute=0)
        current_data = self.dsfactory.get_prediction_data(
            now,
            to_future=to_future,
            to_past=to_past
        )
        index = pd.date_range(start=now - datetime.timedelta(hours=to_past),
                              end=now + datetime.timedelta(hours=to_future),
                              freq=f"{60}min")

        with torch.no_grad():
            latent_normal = self.model(current_data)
        rate_dist = distributions.LogNormal(loc=latent_normal.mean,
                                            scale=latent_normal.stddev)
        return rate_dist, index


def get_current_prediction(model, dsfactory, to_future: int = 2):
    tz = pytz.timezone('Europe/Prague')
    c_date = datetime.datetime.now(tz)
    current_data, column_names = dsfactory.get_prediction_data(c_date, to_future=to_future, to_past=to_future + 1)
    to_past = to_future
    index = pd.date_range(start=c_date - datetime.timedelta(hours=to_past),
                          end=c_date + datetime.timedelta(hours=to_future),
                          freq=f"{60}min")

    with torch.no_grad():
        output = model(current_data)
    rate_samples = output.rsample(torch.Size([1000])).exp()

    percentile, = percentiles_from_samples(rate_samples, [0.8])
    df = pd.DataFrame(percentile,
                      columns=['Arrivals Hourly Rate'],
                      index=[pd.to_datetime(date) for date in index]
                      )
    for col, i in zip(column_names, range(current_data.shape[1])):
        df[col] = current_data[:, i].detach().cpu().numpy()
    return df


def get_past_prediction(model, dsfactory, past_date, to_future: int = 2, to_past: int = 168):
    current_data = dsfactory.get_prediction_data(
        past_date, to_future, to_past=to_past)
    index = pd.date_range(start=past_date - datetime.timedelta(hours=to_past - 1),
                          end=past_date + datetime.timedelta(hours=to_future),
                          freq=f"{60}min")

    with torch.no_grad():
        output = model(current_data)
    rate_samples = output.rsample(torch.Size([1000])).exp()

    percentile, = percentiles_from_samples(rate_samples, [0.8])
    tmp_df = pd.DataFrame(percentile,
                          columns=['Arrivals Hourly Rate'],
                          index=[pd.to_datetime(date).strftime(
                              "%Y-%m-%d %H-%M-%S") for date in index]
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
    pressure_zeroed = pressure - min(pressure)
    pressure_norm = pressure_zeroed / max(pressure_zeroed)

    model.eval()
    with torch.no_grad():
        output = model(test_x)

    # Similarly get the 5th and 95th percentiles
    lat_samples = output.rsample(torch.Size([30])).exp()
    samples_expanded = model.gp.likelihood(
        lat_samples).sample(torch.Size([30]))
    samples = samples_expanded.view(
        samples_expanded.size(0) * samples_expanded.size(1), -1)

    # Similarly get the 5th and 95th percentiles
    lower, fn_mean, upper = percentiles_from_samples(
        lat_samples, [.001, 0.5, 0.8])

    y_sim_lower, y_sim_mean, y_sim_upper = percentiles_from_samples(
        samples, [.001, 0.5, 0.8])

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

    dfactory_online = data_loader.OnlineFactory(data_frequency=hack_config.data_frequency,
                                                teams=hack_config.teams,
                                                hospital=hack_config.hospital,
                                                data_folder=hack_config.data_folder,
                                                dtype=dtype)
    ora = Oracle(hack_config.model_path, dsfactory=dfactory_online)
    anchor_time = datetime.datetime.fromisoformat('2021-11-19-16:00')
    preds = ora.get_arrival_prob_mass(n_arrivals=8,
                                      to_future=10,
                                      to_past=10,
                                      now=anchor_time)
    ax = sns.heatmap(preds.T,
                     vmin=0,
                     vmax=1,
                     annot=True,
                     fmt='.2f',
                     cmap=sns.color_palette("light:darkred", as_cmap=True),
                     cbar_kws={'label': 'Probability'})
    ax.set_xticklabels(preds.index.strftime('%d. %m. %H:%M'))
    ax.invert_yaxis()
    ax.set_xlabel('Time')
    ax.set_ylabel('Number of Arrivals')
    ax.set_title(f'Probability of hourly arrivals {str(anchor_time.date())}')
    plt.show()

    rates = ora.get_arrival_rates(to_future=10,
                                  to_past=10,
                                  now=anchor_time)
    rates.plot()
    plt.show()

    rates_dist, index = ora.get_rates_distribution(to_future=10,
                                                   to_past=10,
                                                   now=anchor_time)
    lower = rates_dist.icdf(torch.tensor(0.1))
    mu = rates_dist.icdf(torch.tensor(0.5))
    upper = rates_dist.icdf(torch.tensor(0.9))
    fig, ax = plt.subplots(1, 1, figsize=(12, 3))
    lineplot = ax.plot(index, mu)
    ax.fill_between(
        index,
        lower,
        upper,
        color=lineplot[0].get_color(), alpha=0.1
    )
    ax.grid()
    ax.set_xlabel('Time')
    ax.set_ylabel('Arrival Rate')
    fig.tight_layout()

    print(f'Done')
