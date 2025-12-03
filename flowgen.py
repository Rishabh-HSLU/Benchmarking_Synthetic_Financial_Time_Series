import pandas as pd
from yahooquery import Ticker

from ydata.synthesizers.timeseries.model import TimeSeriesSynthesizer
from ydata.dataset import Dataset
from ydata.metadata import Metadata
from ydata.utils.data_types import VariableType

def stock_5y(ticker: str) -> pd.DataFrame:
    stock = Ticker(ticker)
    df = stock.history(period="5y")
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