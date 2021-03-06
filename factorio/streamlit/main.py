import datetime
from pathlib import Path

import pytz
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import plotly.express as px
import argparse
import torch
import sys
import os

sys.path.extend([os.getcwd()])
from factorio.core.run_load_model import Oracle, get_current_prediction
from factorio.utils import data_loader, init_logger

parser = argparse.ArgumentParser()
path_parser = parser.add_argument('-c', '--config', type=Path, default='config.ini',
                                  help='Set path to your config.ini file.')

args = parser.parse_args()
if not args.config.exists():
    raise argparse.ArgumentError(path_parser, f"Config file doesn't exist! Invalid path: {args.config} "
                                              f"to config.ini file, please check it!")

hack_config = data_loader.HackConfig.from_config(args.config)
logger = init_logger('zCase_plus', hack_config.log_debug, hack_config.log_path)

st.set_page_config(
    page_title="Očekávané příchody pacientů",
    page_icon=":hearth:",
    layout="centered",
)

count = st_autorefresh(interval=5 * 60 * 1000, limit=1000, key="fizzbuzzcounter")

dtype = torch.float


@st.cache(hash_funcs={torch.nn.parameter.Parameter: lambda parameter: parameter.data.numpy()},
          allow_output_mutation=True)
def get_factory():
    return data_loader.OnlineFactory(data_frequency=hack_config.data_frequency,
                                     hospital=hack_config.hospital,
                                     data_folder=hack_config.data_folder,
                                     weather_columns=hack_config.weather_columns)


dfactory = get_factory()


@st.cache(hash_funcs={torch.nn.parameter.Parameter: lambda parameter: parameter.data.numpy()},
          allow_output_mutation=True)
def create_ora():
    return Oracle(hack_config.model_path, dfactory)


ora = create_ora()

tz = pytz.timezone('Europe/Prague')
c_date = datetime.datetime.now(tz)
st.subheader('Očekávané příchody pacientů')
# hour = st.slider('Prediction Window', 0, 23, 2)
hour = 4

df = get_current_prediction(ora.model, ora.dsfactory, hour)
df.index.name = 'Čas'
logger.info(f'Current prediction: {df.to_dict()}')
fig = px.bar(df['Arrivals Hourly Rate'])
fig.add_vrect(x0=c_date, x1=c_date + datetime.timedelta(minutes=5),
              annotation_text="Aktuální čas", annotation_position="top left",
              fillcolor="black", opacity=0.5, line_width=0)

fig.update_yaxes(title='y', visible=False, showticklabels=False)
# fig.update_xaxes(title='', visible=True, showticklabels=False)
fig.update_xaxes(tickformat="%H:%M")
fig.update_layout(showlegend=False,
                  margin=dict(l=0, r=0, t=0, b=0, pad=4))
st.plotly_chart(fig, use_container_width=True)
