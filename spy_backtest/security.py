from enum import Enum
import pandas
from datetime import datetime
import os

DATA_DIR = os.environ.get('DATA_DIR', './')

class SecType(str, Enum):
    STOCK = 'STOCK'
    ETF = 'ETF'
    FUTURE = 'FUTURE'
    OPTION = 'OPTION'


class Security():
    def __init__(self, json_dict=None):
        self.symbol = None
        self.sec_type = None 
        self.margin_req = None
        self.tick_size = None
        self.tick_value = None
        self._df = None
        self._use_raw = False

        assert(isinstance(json_dict, dict))

        if json_dict:
            self.symbol = json_dict.get('symbol')
            self.sec_type = json_dict.get('sec_type')
            assert(any(self.sec_type == item.value for item in SecType))
            if self.sec_type == SecType.FUTURE:
                margin_req = json_dict.get('margin_req')
                assert(margin_req > 0)
                self.margin_req = margin_req
            tick_size = json_dict.get('tick_size')
            assert(tick_size > 0)
            tick_value = json_dict.get('tick_value')
            assert(tick_value > 0)
            self.tick_size = tick_size
            self.tick_value = tick_value

        self.load_data()
        self._bar_generator = self._create_bar_generator()

    ## defaults to false in initialization 
    def use_raw(self, v=True):
        self._use_raw = v 
   
    def load_data(self):
        fn = f'{DATA_DIR}/{self.symbol}.csv'
        self._df = pandas.read_csv(fn)

    def _adjust_prices(self, df_row):
        ## adjust the entire price df_row to adjusted prices
        ## using the ratio of of Adj_Close/Close as multiplier
        r = df_row['Adj Close']/df_row['Close']
        ah = df_row['High'] * r
        al = df_row['Low'] * r
        ao = df_row['Open'] * r
        dd = dict(Date=df_row['Date'], Open=ao, High=ah, Low=al, Close=df_row['Adj Close'], Volume=df_row['Volume'])
        return pandas.Series(dd)


    def _create_bar_generator(self):

        for index, row in self._df.iterrows():
            if self._use_raw == False:
                row = self._adjust_prices(row)
            dt_index = row['Date']
            cur_dt = datetime.strptime(dt_index,"%Y-%m-%d").date()

            yield index, cur_dt, row 

    def next_bar(self):
        return next(self._bar_generator)

    def fetch_bar(self, str_date):
        filtered_df = self._df[self._df['Date'] == str_date]
        if filtered_df.empty:
            return None
        ## grab the first row in the filtered series
        index = filtered_df.index[0]
        row = filtered_df.loc[index]
        if self._use_raw == False:
            row = self._adjust_prices(row)
        dt_index = row['Date']
        cur_dt = datetime.strptime(dt_index,"%Y-%m-%d").date()

        return index, cur_dt, row 
            


if __name__ == '__main__':
    etf_def = {
            "symbol": "SPY", 
            "sec_type": "ETF",
            "tick_size": 0.01,
            "tick_value": 0.01
    }

    """
    future_def = {
            "symbol": "ES1", 
            "sec_type": "FUTURE",
            "tick_size": 0.25,
            "tick_value": 12.50,
            "margin_req": 12000
    }
    """

    spy = Security(etf_def)

    """
    ## NOTE: you can't use generators in a for loop
    i = 0
    while i < 3:
        i, dtt, row  = spy.next_bar()
        print(i, dtt, row)
    """

    while True:
        try:
            i, dtt, row = spy.next_bar()
        except StopIteration:
            break
    ## print the last row
    print(i, dtt, row)



    v = spy.fetch_bar("2008-10-14")
    ## just prints out the unpacked tuple
    print(v)

    ## test None value
    v = spy.fetch_bar("2008-10-18")  # saturday
    print(v)



