import pandas
import json
import math
from datetime import date, datetime
from enum import Enum
from prettytable import PrettyTable
from df_html_fancy import basic_table_to_html


class DumpFormat(str, Enum):
    STDOUT = 'STDOUT'
    HTML = 'HTML'
    JSON = 'JSON'
    CSV = 'CSV'

class TradeType(str, Enum):
    SELL = 'SELL'
    BUY = 'BUY'


class BuyAndHold():
    def __init__(self, security, start_dt=None):

        self.security = security
        self.trades = list()
        self.test_ledger = list()
        self.start_dt = self.start_from(start_dt)
        self.prev_close = None

        ## backtest metrics dict
        self.metrics = None

        ## controls display precision of prices and any calculations
        self.prettytable_precision = ".2" 

    def generate_metrics(self):

        trades_df = pandas.DataFrame(self.trades)
        pnl_series = pandas.DataFrame(self.test_ledger)

        wins = trades_df[ trades_df['Value'] > 0]

        trade_count = len(trades_df)
        win_pct = len(wins)/trade_count

        returns = (pnl_series['Equity'] / pnl_series['Equity'].shift(1)) - 1
        returns.dropna(inplace=True)

        trade_returns = trades_df['TradeRtn'] 

        trade_wins = trade_returns[trade_returns >= 0]
        trade_losses = trade_returns[trade_returns < 0]

        profit_factor = trade_wins.sum()/(-1 * trade_losses.sum())
        avg_win = trade_wins.mean()
        avg_loss = trade_losses.mean()

        ## vectorized calc of drawdown
        rolling_max = pnl_series['Equity'].cummax()
        jj = (pnl_series['Equity']/rolling_max) - 1
        dd = jj.min()

        years = float(pnl_series.shape[0]/252.0)

        #total return
        totalRtn = (pnl_series.iloc[-1]['Equity']/pnl_series.iloc[0]['Equity']) - 1

        #compounded annualize growth rate
        cagr = ((pnl_series.iloc[-1]['Equity']/pnl_series.iloc[0]['Equity']) ** (1.0/years)) - 1
        sharpe = cagr/(returns.std() * math.sqrt(252))

        self.metrics = dict(Sharpe=sharpe,
                        CAGR=cagr,
                        MaxDD=dd,
                        Trades=trade_count,
                        WinPct=win_pct,
                        PFactor=profit_factor,
                        AvgWin=avg_win,
                        AvgLoss=avg_loss,
                        Years=years,
                        TotalRtn=totalRtn)


    def format_df(self, lst_dicts):
        def _fstr(value):
            if value is None:
                return ""

            v = round(value, 3)
            q = round(value, 2)
            if v == q:
                return str(q)
            return str(v)

        def _istr(value):
            if value is None:
                return ""

            return str(value)

        def _format(dikt):
            float_fields = 'Entry Exit StopLevel MTM Equity DollarBase Value TradeRtn PNL'.split()
            int_fields = 'Position Duration'.split()
            for k in dikt.keys():
                if k in float_fields: dikt[k] = _fstr(dikt[k])
                if k in int_fields: dikt[k] = _istr(dikt[k])
            return dikt

        formatted_list = []
        for dikt in lst_dicts:
            formatted_list.append(_format(dikt))

        df = pandas.DataFrame(formatted_list)
        df = df.fillna("")
        return df



    def format_table(self, pretty_table):
        COLUMNS_TO_CENTER = 'Date InDate ExDate InSignal ExSignal'.split()
        MONEY_COLUMNS = 'MTM PNL Equity'.split()
        #float_fields = 'Close Entry Exit StopLevel MTM Equity'.split()
        float_fields = 'Close Entry Exit StopLevel'.split()

        new_table = pretty_table.copy()
        for col in pretty_table.field_names:
            new_table.float_format[col] = self.prettytable_precision
            if col in MONEY_COLUMNS:
                new_table.float_format[col] = ".2"
            new_table.align[col] = "r"
            if col in COLUMNS_TO_CENTER:
                new_table.align[col] = "c"
            if col == 'Position':
                new_table.float_format[col] = ".0" 

        return new_table 


    def dump_trades(self, formats=[DumpFormat.CSV]):
        ## stdout, csv, html
        trades_df = pandas.DataFrame(self.trades)
        if DumpFormat.CSV in formats:
            trades_df = trades_df.round(4)
            trades_df.to_csv('trades.csv', index=False)

        if DumpFormat.STDOUT in formats:
            trades_df = trades_df.fillna("")
            col_list = trades_df.columns.tolist()
            daily_table = PrettyTable(col_list)
            daily_table = self.format_table(daily_table)
            for i, row in trades_df.iterrows():
                daily_table.add_row(row.tolist())
            print(daily_table)

        if DumpFormat.HTML in formats:
            trades_df = self.format_df(self.trades)
            html = basic_table_to_html(trades_df, 'Buy+Hold Trades')
            with open('trades.html', 'w') as f:
                f.write(html + '\n')

    def dump_test_ledger(self, formats=[DumpFormat.STDOUT]):
        ## stdout, csv, html
        test_ledger_df = pandas.DataFrame(self.test_ledger)
        test_ledger_df = test_ledger_df.round(4)
        pnl_series_df = test_ledger_df[['Date','Equity']]
        if DumpFormat.CSV in formats:
            test_ledger_df.to_csv('test_ledger.csv', index=False)
            pnl_series_df.to_csv('pnl_series.csv', index=False)

        if DumpFormat.STDOUT in formats:
            test_ledger_df = test_ledger_df.fillna("")
            #test_ledger_df['MTM'] = test_ledger_df['MTM'].map(lambda x:f'{x:,.2f}')
            #test_ledger_df['Equity'] = test_ledger_df['Equity'].map(lambda x:f'{x:,.2f}')
            col_list = test_ledger_df.columns.tolist()
            daily_table = PrettyTable(col_list)
            daily_table = self.format_table(daily_table)
            for i, row in test_ledger_df.iterrows():
                daily_table.add_row(row.tolist())
            print(daily_table)

        if DumpFormat.HTML in formats:
            trades_series_df = self.format_df(self.test_ledger)
            html = basic_table_to_html(trades_series_df, 'Buy+Hold Series')
            with open('trades_series.html', 'w') as f:
                f.write(html + '\n')

    def dump_metrics(self, formats=[DumpFormat.STDOUT], transpose=False, value_list=None, title=None):
        ## stdout, html, and json
        metrics_df = pandas.DataFrame([self.metrics])
        if value_list:
            metrics_df = metrics_df[value_list]
        metrics_df = metrics_df.round(3)

        ttl = ' '
        if title: ttl = title

        if transpose:
            #metrics_df = self.format_df(metrics_df)
            metrics_df = metrics_df.T
            metrics_df.reset_index(inplace = True)
            metrics_df.rename(columns={'index':'Metric', 0:'Value'}, inplace=True)
           
        if DumpFormat.STDOUT in formats:
            daily_table = PrettyTable(metrics_df.columns.tolist())
            daily_table.align['Metric'] = "l"
            daily_table.align['Value'] = "r"
            daily_table.float_format['Value'] = ".3"
            for i, row in metrics_df.iterrows():
                daily_table.add_row(row.tolist())
            print(daily_table)

        if DumpFormat.HTML in formats:
            html = basic_table_to_html(metrics_df, ttl)
            with open('metrics.html', 'w') as f:
                f.write(html + '\n')

        if DumpFormat.JSON in formats:
            formatted_metrics = { k: round(v,4) for k,v in self.metrics.items() }
            metrics_json = json.dumps(formatted_metrics, indent=4)
            with open('metrics.json', 'w') as f:
                f.write(metrics_json + '\n')



    def results(self):
        trades_df = pandas.DataFrame(self.trades)
        test_ledger_df = pandas.DataFrame(self.test_ledger)
        metrics_df = pandas.DataFrame([self.metrics])
        metrics_df = metrics_df.T
        metrics_df.reset_index(inplace = True)
        metrics_df.rename(columns={'index':'Metric', 0:'Value'}, inplace=True)

        return dict(Trades=trades_df,
                    TestLedger=test_ledger_df,
                    Metrics=metrics_df)
   
  
    def start_from(self, start_from_dt):
        s = None
        if start_from_dt is not None:
            if isinstance(start_from_dt, date):
                s = start_from_dt
            else:
                s = datetime.strptime(start_from_dt,"%Y-%m-%d").date()
        return s

    ## loads the next price bar 
    def load(self):
        try:
            return self.security.next_bar()
        except StopIteration:
            return None  


    ## runs throught the strategy on ONE bar
    def execute(self, bar_data):

        i, cur_dt, bar = bar_data

        if self.start_dt and cur_dt < self.start_dt:
            return

        if self.prev_close is not None:
            trade = bar['Close'] - self.prev_close
            trade_rtn = bar['Close']/self.prev_close - 1
            self.trades.append( dict(Date=cur_dt, Value=trade, TradeRtn=trade_rtn) ) 
            self.test_ledger.append( dict(Date=cur_dt, Close=bar['Close'], Equity=bar['Close']) ) 

        self.prev_close = bar['Close']


    def run(self):

        self.metrics = None

        # bar_data = (i, cur_dt, bar)
        # i = integer index
        # cur_dt = bar datetime = datetime.strptime(dt)
        # bar = OHLC, etc data.
        while True:

            bar_data = self.load()
            if bar_data is None:
                break

            self.execute(bar_data)


    def run_and_report(self):

        self.run()

        self.generate_metrics()

        self.dump_test_ledger(formats= [DumpFormat.STDOUT, DumpFormat.CSV])
        self.dump_metrics(transpose=True)
        self.dump_trades()

