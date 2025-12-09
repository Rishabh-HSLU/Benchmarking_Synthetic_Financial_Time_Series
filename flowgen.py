import pandas as pd
import numpy as np
from yahooquery import Ticker

from ydata.synthesizers.timeseries.model import TimeSeriesSynthesizer
from ydata.dataset import Dataset
from ydata.metadata import Metadata
from ydata.utils.data_types import VariableType

from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
import plotly.express as px

def stock_5y(ticker: str, period: str = "5y", interval : str= "1d" ) -> pd.DataFrame:
    stock = Ticker(ticker)
    df = stock.history(period= period, interval = interval)
    df.reset_index(level=0, inplace=True)
    df.drop('symbol', axis= 'columns', inplace=True)
    df = df[['close']].reset_index()
    return df

def synthesis(dataframe: pd.DataFrame) -> pd.DataFrame:
    import os
    os.environ['YDATA_LICENSE_KEY'] = 'b6580aad-809b-4ff4-be31-af0159c28b99'

    # wrap dataset
    dataset = Dataset(dataframe)

    # tell YData the time column type
    dataset.astype('date', VariableType.DATE)

    # metadata for the synthesizer
    metadata = Metadata(dataset, dataset_attrs={"sortbykey": 'date'})

    # synthesize
    synth = TimeSeriesSynthesizer()
    synth.fit(dataset, metadata=metadata)

    return synth.sample(n_entities=1).to_pandas()

def regime_analysis(df: pd.DataFrame, k_regimes = 2) -> list:
    # create excess price returns dataframe with one day as interval
    df_returns = pd.DataFrame({'date': df['date'], 'returns': df['close'].diff()})
    # regime switching using markov regression
    mod_kns = MarkovRegression(df_returns['returns'].dropna(), k_regimes = k_regimes, trend='n', switching_variance=True)
    res_kns = mod_kns.fit()

    regime_list = []
    if k_regimes == 2:
        low_var = list(res_kns.smoothed_marginal_probabilities[0])
        high_var = list(res_kns.smoothed_marginal_probabilities[1])

        for i in range(0, len(low_var)):
            if low_var[i] > high_var[i]:
                regime_list.append(0)
            else:
                regime_list.append(1)

    return regime_list


def plot_regime_switching(df: pd.DataFrame, regime_list: list, k_regimes = 2) -> None:
    df_trim = df.iloc[len(df) - len(regime_list):]

    # build regimes dataframe
    regimes = (
        pd.DataFrame({"regimes": regime_list}, index=df_trim.index)
        .join(df_trim)
        .reset_index()
        .rename(columns={"index": "Date"})
    )

    fig = px.scatter(
        regimes,
        x="Date",
        y="close",
        color="regimes",
        color_discrete_map={0: "green", 1: "red", 2: "yellow"},
        title="Historical SP500 Regimes",
        opacity=0.8,
    )

    fig.update_layout(height=700, title_font=dict(size=26))
    fig.show()

def spectral_entropy(time_series: pd.Series):
    spectrum = np.fft.fft(time_series)
    ps = np.abs(spectrum) ** 2
    ps_sum = np.sum(ps)
    if ps_sum == 0.0:
        return 0.0
    p = ps / ps_sum
    p = p[p != 0]
    return -np.sum(p * np.log2(p)) / np.log2(len(ps))