import configparser
import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.distributions.poisson import Poisson
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset, Subset
# from factorio.gpmodels.gppoissonpl import RateGPpl, fit
import pickle

from factorio.gpmodels.gplognormpl import LogNormGPpl, fit
from factorio.utils import data_loader
from factorio.utils.helpers import percentiles_from_samples

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import argparse

    # Move to config at some point
    dtype = torch.float
    num_inducing = 64
    num_iter = 5000
    num_particles = 32
    loader_batch_size = 15000
    learn_inducing_locations = True
    slow_mode = False  # enables checkpointing and logging
    learning_rate = 0.01
    hold_out_last_n = 500  # last hold_out_last_n samples are held out for validation

    time_now = datetime.datetime.utcnow()
    parser = argparse.ArgumentParser()

    path_parser = parser.add_argument('-c', '--config', type=Path, default='config.ini',
                                      help='Set path to your config.ini file.')
    args = parser.parse_args()
    if not args.config.exists():
        raise argparse.ArgumentError(path_parser, f"Config file doesn't exist! Invalid path: {args.config} "
                                                  f"to config.ini file, please check it!")

    hack_config = data_loader.HackConfig.from_config(args.config)

    output_path = f'mnt/model_state_{time_now.strftime("%Y%m%d-%H%M%S")}.pth'

    dfactory = data_loader.DataFactory(data_frequency=hack_config.data_frequency,
                                       hospital=hack_config.hospital,
                                       data_folder=hack_config.data_folder,
                                       weather_columns=hack_config.weather_columns,
                                       dtype=dtype)

    X_mins, X_maxs = dfactory.get_min_max()

    dlen = len(dfactory.dset)
    loader = DataLoader(
        # dfactory.dset,
        Subset(dfactory.dset, torch.arange(0, dlen - hold_out_last_n) - 1),
        batch_size=loader_batch_size,
        shuffle=True
    )
    model = LogNormGPpl(num_inducing=num_inducing,
                        X_mins=X_mins,
                        X_maxs=X_maxs,
                        learn_inducing_locations=learn_inducing_locations,
                        lr=learning_rate,
                        num_particles=num_particles,
                        num_data=dlen)

    fit(model,
        train_dataloader=loader,
        max_epochs=num_iter,
        patience=10,
        verbose=False,
        enable_checkpointing=slow_mode,
        enable_logger=True,
        use_gpu=hack_config.use_gpu)

    model.save_model(output_path)
    with open('mnt/scaler.pkl', 'wb') as fid:
        pickle.dump(dfactory.scaler, fid)
    show = 200
    test_x = dfactory.dset[-show:][0]
    real_x = dfactory.inverse_transform(test_x)
    Y = dfactory.dset[-show:][1]
    x_plt = dfactory.data[-show:].index.values
    model.eval()
    with torch.no_grad():
        output = model(test_x)

    # Similarly get the 5th and 95th percentiles
    lat_samples = output.rsample(torch.Size([100])).exp()
    samples_expanded = model.gp.likelihood(lat_samples).sample(torch.Size([100]))
    samples = samples_expanded.view(samples_expanded.size(0) * samples_expanded.size(1), -1)

    # Similarly get the 5th and 95th percentiles
    lower, fn_mean, upper = percentiles_from_samples(lat_samples)

    y_sim_lower, y_sim_mean, y_sim_upper = percentiles_from_samples(samples)

    # visualize the result
    fig, (ax_func, ax_samp) = plt.subplots(1, 2, figsize=(12, 3))
    line = ax_func.plot(
        x_plt, fn_mean.detach().cpu(), label='GP prediction')
    ax_func.fill_between(
        x_plt, lower.detach().cpu().numpy(),
        upper.detach().cpu().numpy(), color=line[0].get_color(), alpha=0.5
    )
    ax_func.legend()

    ax_samp.scatter(x_plt, Y, alpha=0.5,
                    label='True train data', color='orange')
    y_sim_plt = ax_samp.plot(x_plt, y_sim_mean.cpu(
    ).detach(), alpha=0.5, label='Sample mean from the model')
    ax_samp.fill_between(
        x_plt, y_sim_lower.detach().cpu(),
        y_sim_upper.detach().cpu(), color=y_sim_plt[0].get_color(), alpha=0.5
    )
    ax_samp.legend()
    fig.tight_layout()
    plt.show()

    print(f'Done')
