from enum import Enum
from datetime import datetime
import pandas
import math
from prettytable import PrettyTable
from df_html_fancy import basic_table_to_html
import json

class DumpFormat(str, Enum):
    STDOUT = 'STDOUT'
    HTML = 'HTML'
    JSON = 'JSON'
    CSV = 'CSV'


class Portfolio():
    def __init__(self):
        self.strategies = dict() 
        self.trades = list()
        self.test_ledger = dict()
        self.allocation_map = dict()
        self.capital = 0
        self.gauntlet = dict()

        self.show_detail = False

        ## backtest metrics dict
        self.metrics = None

    def set_capital(self, money):
        assert(money > 0)
        self.capital = money

    def add(self, strategy):
        ## attach the portfolio and strategy to each other
        strategy.portfolio = self

        self.strategies[strategy.name] = strategy
        self.allocation_map[strategy.name] = None
        self.gauntlet[strategy.name] = None

    def get_allocation(self, strategy_id):
        alloc = round(self.allocation_map[strategy_id] * self.capital, 2)
        return alloc 

    def set_allocations(self, custom_allocations=None):
        ## custom_allocations = dict(Strategy_Id:pctAmount)
        keys = list(self.allocation_map.keys())
        count = len(keys)

        allocated = 0
        alloc_sum = 0
        if custom_allocations:
            values = sum(list(custom_allocations.values()))
            assert(values <= 1)
            for k, v in custom_allocations.items():
                if k in keys:
                    self.allocation_map[k] = v
                    alloc_sum += v
                    allocated += 1

        ## distribute the remaining unallocated evenly
        if allocated < count:
            p = (1 - alloc_sum ) / (count-allocated)
            for k in keys:
                if self.allocation_map[k] is None:
                    self.allocation_map[k] = p

        ## set up inital allocations for each strategy
        ## before they start trading
        for name in keys:
            strategy = self.strategies[name]
            strategy.wallet = self.get_allocation(name)


    def _tag_dict(self, tag, index, dikt):
        ## place 'tag' at the 'index' column in the record
        keys, values = list(dikt.keys()), list(dikt.values())
        keys.insert(index,'Name')
        values.insert(index,tag)
        return dict(zip(keys,values)) 

    def post_trade(self, strategy_name, trade_dict):
        self.capital += trade_dict['Value']
        self.trades.append( self._tag_dict(strategy_name, 0, trade_dict))

    def post_record(self, strategy_name, backtest_dict):
        record = self._tag_dict(strategy_name, 1, backtest_dict)
        dt = record['Date']
        try:
            self.test_ledger[dt].append(record)
        except KeyError:
            self.test_ledger[dt] = [record]


    def run(self):

        self.set_allocations()

        while True:

            for name in self.strategies.keys():
                if self.gauntlet[name] is None:
                    ## load the strategy's next bar
                    ## if no bar to load - load() returns None
                    ## otherwise returns tuple(index, cur_dt, price_bar) 
                    bar_data = self.strategies[name].load()
                    if bar_data:
                        self.gauntlet[name] = (bar_data, name)

            loaded = self.gauntlet.values()
           
            ## no new data across all strategies -- you're done
            if not any(loaded): break 

            ## find the smallest date of the current bar across
            ## all strategies
            pending = []
            min_date = None
            for bar_data, name in loaded:
                i, cur_dt, price_bar = bar_data
                if min_date is None: min_date = cur_dt
                min_date = min(min_date, cur_dt)
                pending.append(dict(Date=cur_dt, Name=name, Data=bar_data))

            ## run all strategies marked with the current earliest date 
            for strategy_pending in pending:
                if strategy_pending['Date'] == min_date:
                    name, bar_data = strategy_pending['Name'], strategy_pending['Data']
                    self.strategies[name].execute(bar_data)
                    ## remove processed data from the gauntlet
                    self.gauntlet[name] = None


    ## post-simulation functions 

    def _collapse_test_ledger(self):

        detail = []
        combined = []
        ## just to make sure things are done cronologically
        for dt in sorted(list(self.test_ledger.keys())):
            v = self.test_ledger[dt]

            ## aggregate across all strategies and collapse
            ## total MTM for the day
            total_mtm = 0
            total_equity = 0
            for record in v:
                total_mtm += record['MTM']
                total_equity += record['Equity']
                new_record = dict(record)
                new_record.update(dict(TotalMTM=total_mtm, TotalEquity=total_equity))
                detail.append(new_record)

            ## label aggregate long and short strategies
            longs = " ".join([ x['Name'][1:] for x in v if x['Name'].startswith('+') ]) 
            shorts = " ".join([ x['Name'][1:] for x in v if x['Name'].startswith('-') ])

            combined_record = dict(Date=dt, Long=longs, Shorts=shorts)
            combined_record.update( dict(MTM=total_mtm, Equity=total_equity) ) 
            combined.append(combined_record)

        return detail, combined


    def generate_metrics(self):

        trades_df = pandas.DataFrame(self.trades)
        _, combined_ledger = self._collapse_test_ledger() 
        pnl_series = pandas.DataFrame( combined_ledger )

        #import pdb; pdb.set_trace()

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

        ##t vectorized calc of drawdown
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
        COLUMNS_TO_CENTER = 'Date InDate ExDate InSignal ExSignal Long Short Name'.split()
        MONEY_COLUMNS = 'MTM PNL Equity'.split()
        #float_fields = 'Close Entry Exit StopLevel MTM Equity'.split()
        float_fields = 'Open Close Entry Exit StopLevel'.split()

        new_table = pretty_table.copy()
        for col in pretty_table.field_names:
            new_table.float_format[col] = ".5" 
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
            html = basic_table_to_html(trades_df, 'BackTest Trades')
            with open('trades.html', 'w') as f:
                f.write(html + '\n')
            
    def dump_test_ledger(self, formats=[DumpFormat.STDOUT]):
        ## collapse ledger returns a flattened detailed and combined 
        ## version of self.test_ledger that is table/anaylsis ready
        ledgers = zip(('detailed','combined'), self._collapse_test_ledger())
        for name, ledger in ledgers:
            if name == 'detailed' and not self.show_detail: continue
            self._dump_ledger(name, ledger, formats)

    def _dump_ledger(self, name, ledger, formats):
        ## stdout, csv, html
        test_ledger_df = pandas.DataFrame(ledger)
        test_ledger_df = test_ledger_df.round(4)
        pnl_series_df = test_ledger_df[['Date','Equity']]
        if DumpFormat.CSV in formats:
            test_ledger_df.to_csv(f'test_ledger.{name}.csv', index=False)
            pnl_series_df.to_csv(f'pnl_series.{name}.csv', index=False)

        if DumpFormat.STDOUT in formats:
            test_ledger_df = test_ledger_df.fillna("")
            test_ledger_df['MTM'] = test_ledger_df['MTM'].replace(0,"")
            #test_ledger_df['MTM'] = test_ledger_df['MTM'].map(lambda x:f'{x:,.2f}')
            #test_ledger_df['Equity'] = test_ledger_df['Equity'].map(lambda x:f'{x:,.2f}')
            col_list = test_ledger_df.columns.tolist()
            daily_table = PrettyTable(col_list)
            daily_table = self.format_table(daily_table)
            for i, row in test_ledger_df.iterrows():
                daily_table.add_row(row.tolist())
            print(daily_table)

        if DumpFormat.HTML in formats:
            trades_series_df = self.format_df(ledger)
            html = basic_table_to_html(trades_series_df, 'Backtest Series')
            with open(f'trades_series.{name}.html', 'w') as f:
                f.write(html + '\n')
            html = basic_table_to_html(trades_series_df, 'Backtest Ledger')
            with open(f'test_ledger.{name}.html', 'w') as f:
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
 

    def run_and_report(self):

        self.run()

        self.generate_metrics()

        self.dump_test_ledger(formats= [DumpFormat.STDOUT, DumpFormat.CSV])
        self.dump_metrics(transpose=True)
        self.dump_trades()

