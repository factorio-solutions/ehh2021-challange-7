import datetime
from pathlib import Path

import streamlit as st
from streamlit_autorefresh import st_autorefresh
import plotly.express as px
import argparse
import torch
import sys

sys.path.extend([r'C:\Projects\ehh2021-challange-7'])
from factorio.core.run_load_model import Oracle, get_current_prediction
from factorio.utils import data_loader

st.set_page_config(
    page_title="Patient Arrival Prediction",
    page_icon=":hearth:",
    layout="centered",
    # menu_items={
    #     'Get Help': 'https://www.extremelycoolapp.com/help',
    #     'Report a bug': "https://www.extremelycoolapp.com/bug",
    #     'About': "# This is a header. This is an *extremely* cool app!"
    # }
)

count = st_autorefresh(interval=900000, limit=1000, key="fizzbuzzcounter")
dtype = torch.float
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


@st.cache(hash_funcs={torch.nn.parameter.Parameter: lambda parameter: parameter.data.numpy()},
          allow_output_mutation=True)
def get_factory():
    hack_config = data_loader.HackConfig.from_config(args.config)
    return data_loader.OnlineFactory(data_frequency=hack_config.data_frequency,
                                     teams=hack_config.teams,
                                     hospital=hack_config.hospital,
                                     data_folder=hack_config.data_folder,
                                     dtype=dtype)


dfactory = get_factory()


@st.cache(hash_funcs={torch.nn.parameter.Parameter: lambda parameter: parameter.data.numpy()},
          allow_output_mutation=True)
def create_ora():
    return Oracle(load_path, dfactory)


ora = create_ora()

c_date = datetime.datetime.now()
st.subheader('Predict future hour!')
hour = st.slider('Prediction Window', 0, 23, 2)

df = get_current_prediction(ora.model, ora.dsfactory, hour)
df.index.name = 'Datetime'
fig = px.bar(df)
fig.add_vrect(x0=c_date, x1=c_date + datetime.timedelta(minutes=5),
              annotation_text="Current time", annotation_position="top left",
              fillcolor="black", opacity=0.5, line_width=0)

fig.update_yaxes(title='y', visible=False, showticklabels=False)
fig.update_xaxes(title='', visible=True, showticklabels=False)
fig.update_layout(showlegend=False,
                  margin=dict(l=0, r=0, t=0, b=0, pad=4))
st.plotly_chart(fig, use_container_width=True)
