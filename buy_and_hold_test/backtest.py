import pandas
import json
import math
from datetime import date, datetime
from enum import Enum
from prettytable import PrettyTable
from indicators import StDev, HighestValue, LowestValue
import calendar_calcs
from security import SecType
from df_html_fancy import basic_table_to_html 
import uuid


class DumpFormat(str, Enum):
    STDOUT = 'STDOUT'
    HTML = 'HTML'
    JSON = 'JSON'
    CSV = 'CSV'

class TradeType(str, Enum):
    SELL = 'SELL'
    BUY = 'BUY'


class BackTest():
    def __init__(self, security, json_config, ref_index=None):

        """
        - json_config sets up params for the strategy AND 
          initial cash set up
        json_config = {
                        "settings": 
                            {
                                "StDev": 50,
                                "ma200": 200,
                                "duration": 10,
                                "start_dete": "2014-01-01"
                            },
                        "wallet":
                            {
                                "cash": 10000,
                                "wallet_alloc_pct": 0.5,
                                "borrow_margin_pct": 0.5
                            },
                        "trading_limits":
                            {
                                "dollar_limit": 100000000,
                                "position_limit: 100000,
                                "leverage_target": 1
                            }
        }

        Security Class - member variables:
            self.symbol = symbol string 9
            self.sec_type = SecType enum
            self.margin_req = margin requirement for futures 
            self.leverage_target = leverage target for futures 
            self.tick_size = tick size 
            self.tick_value = tick value 
            self._df = datafram of securirty time series 
            self._use_raw = use non adjusted prices
        """

        ## strategy settings 
        self.config = json_config

        ## turn on backtest
        self.backtest_enabled = True 
        settings = self.config.get('settings', {})
        if settings:
            self.start_dt = self.start_from( settings.get("start_date") )

        ## portfolio hooks
        self.name = self.config.get('name', uuid.uuid4())
        self.portfolio = None

        ## Security objects
        self.security = security  
        self.ref_index = ref_index 

        self.wallet = 0 
        self.wallet_alloc_pct = 1
        self.borrow_margin_pct = 1
        self._initialize_wallet()

        ## limit on capital at risk
        self.dollar_limit = None
        #W limit on position size
        self.position_limit = None
        self.leverage_target = None
        self._initialize_limits()

        ## dict of { date, price, position, ex_date, ex_price, entry_label, exit_label, ... }
        self.current_trade = None  

        self.pnl = 0
        self.trades = list()
        self.test_ledger = list()

        ## backtest metrics dict
        self.metrics = None

        ## indicators and tools

        self.stdev = StDev( sample_size= settings.get('StDev', 50) )
        self.high_marker = self.low_marker = None
        self.holidays = calendar_calcs.load_holidays()

        ## controls display precision of prices and any calculations
        self.prettytable_precision = ".2" 


    def _initialize_wallet(self):
        wallet_dict = self.config.get('wallet')
        assert( wallet_dict is not None )

        self.wallet = float( wallet_dict.get('cash', 10000) )
        self.wallet_alloc_pct = float( wallet_dict.get('wallet_alloc_pct', 1) )
        self.borrow_margin_pct = float( wallet_dict.get('borrow_margin_pct', 1) )

    def _initialize_limits(self):
        limit_dict = self.config.get('trading_limits')
        assert( limit_dict is not None )

        self.dollar_limit = float( limit_dict.get('dollar_limit', 100000000) )
        self.position_limit = int( limit_dict.get('position_limit', 100000) )
        self.leverage_target = float( limit_dict.get('leverage_target', 1) )

    @property
    def LONG(self):
        if self.current_trade is not None:
            if self.current_trade['Position'] > 0:
                return True
        return False

    @property
    def SHORT(self):
        if self.current_trade is not None:
            if self.current_trade['Position'] < 0:
                return True
        return False

    @property
    def CLOSED(self):
        ## trade state BEFORE self.current_trade is reset to None
        if self.current_trade and self.current_trade.get('Exit'):
            return True
        return False

    @property
    def FLAT(self):
        if self.current_trade is None:
            return True
        return False
            
    ## trade execution functions

    def enter_trade(self, trade_type, str_dt, security, price, label=''):

        ## if a Portfolio object is managing this strategy,
        ## ask the Portfolio object for tha amount of money the 
        ## strategy can trade
        if self.portfolio:
            self.wallet = self.portfolio.get_allocation(self.name)

        if not self.wallet:
            return None 

        basis = price
        if security.sec_type == SecType.FUTURE:
            ## allocate based on margin requirment per contact.
            ## otherwise it would be share price
            basis = security.margin_req
            

        if self.wallet > 0:
            assert(basis > 0)
            ## 1. get the cash allocated to the trade
            ## 2. then borrow on that allocation if borrow margin pct is given.
            dollar_base = self.wallet_alloc_pct * self.wallet
            buying_power = dollar_base/self.borrow_margin_pct
            shares = int(buying_power/basis)

            if security.sec_type == SecType.FUTURE:
                if self.leverage_target:
                    tick_size = security.tick_size
                    tick_value = security.tick_value

                    assert(tick_size > 0)
                    assert(tick_value > 0)
                    multiplier = tick_value / tick_size

                    ## buy contracts in accordance to desired leverage target
                    shares = int((buying_power * self.leverage_target)/(price * multiplier))


            ## limit total shares/contracts that can be traded
            if self.position_limit: 
                shares = min(shares, self.position_limit)

            if shares > 0:
                if trade_type == TradeType.SELL:
                    shares = -shares

                return dict(InDate=str_dt, Entry=price, Position=shares, DollarBase=dollar_base, Duration=0, InSignal=label)


    ## formats the self.current_trade dictionary once a trade is closed
    ## this function is needed for compatability to interface with the portfolio
    def _format_closed_trade(self):
        dict_order = 'InDate ExDate Position Duration InSignal Entry ExSignal Exit DollarBase Value TradeRtn PNL'
        return {key: self.current_trade[key] for key in dict_order.split() if key in self.current_trade}


    def exit_trade(self, str_dt, security, price, label=""):

        exit_dt = str_dt
        tick_size = security.tick_size
        tick_value = security.tick_value

        assert(tick_size > 0)
        assert(tick_value > 0)

        delta = price - self.current_trade['Entry']
        trade_value = (delta/tick_size) * tick_value * self.current_trade['Position']

        self.pnl += trade_value
        rtn = trade_value/self.current_trade['DollarBase']
        exit_dict = dict(ExDate=str_dt, Exit=price, Value=trade_value, TradeRtn=rtn, PNL=self.pnl, ExSignal=label)

        self.current_trade.update(exit_dict)

        self.trades.append( self._format_closed_trade() )
    


    def initialize_stop(self, anchor):

        initial_stop = None
        if self.LONG:
            initial_stop = self.high_marker = HighestValue(anchor)
            self.low_marker = None

        if self.SHORT:
            self.high_marker = None 
            initial_stop = self.low_marker = LowestValue(anchor) 
    
        return initial_stop


    def calc_price_stop(self, anchor, multiplier=2.5, default=0.30):
        m = None
        volatility = self.stdev.valueAt(0)

        if self.LONG:
            if volatility is not None:
                m = anchor - (volatility * multiplier)
            else:
                m = anchor * (1-default)

        if self.SHORT:
            if volatility is not None:
                m = anchor + (volatility * multiplier)
            else:
                m = anchor * (1+default)

        return m 

    def calc_drawdown_stop(self):
        ## calc stop based on percentage loss in equity
        pass


    def trade_update(self, cur_dt, bar, ref_bar):
        ## update stops, duration and any other trade specific rule at EOD

        if self.LONG:
            self.high_marker.push(bar['High'])
            stop_level = self.current_trade['StopLevel']
            self.current_trade['StopLevel'] = max(stop_level, self.calc_price_stop( self.high_marker.highest))

        if self.SHORT:
            self.low_marker.push(bar['Low'])
            stop_level = self.current_trade['StopLevel']
            self.current_trade['StopLevel'] = min(stop_level, self.calc_price_stop( self.low_marker.lowest))



    def entry_OPEN(self, cur_dt, bar, ref_bar=None):
        if self.backtest_enabled == False:
            return

        if not self.FLAT:
            return
        
        # self.enter_trade( TradeType.BUY, bar['Date'], self.security, bar['Open'], label='LEX' )

    def exit_OPEN(self, cur_dt, bar, ref_bar=None):
        if self.backtest_enabled == False:
            return

        if self.FLAT:
            return

        ## if self.LONG:
        ##    self.exit_trade( bar['Date'], security, bar['Open'] )


    def entry_CLOSE(self, cur_dt, bar, ref_bar=None):
        if self.backtest_enabled == False:
            return

        if not self.FLAT:
            return

        ## ADD entry_CLOSE SIGNAL LOGIC HERE
        ## self.current_trade = self.enter_trade( TradeType.BUY, bar['Date'], self.security, bar['Close'], label = "" )


    def exit_CLOSE(self, cur_dt, bar, ref_bar=None):
        if self.backtest_enabled == False:
            return

        if self.FLAT:
            return

        ## if self.LONG:
        ##    self.exit_trade( bar['Date'], security, bar['Open'] )

    ## position, %wallet and absolute dollar amount
    ## adjustment functions
    ## called at the end fo every trade

    def update_position_limit(self):
        ## updates self.position_limit
        ## based on custom, strategy specific logic
        pass

    def update_wallet_alloc(self):
        ## updates self.wallet_alloc_pct
        ## based on custom, strategy specific logic
        pass

    def update_dollar_limit(self):
        ## updates self.dollar_limit
        ## based on custom, strategy specific logic
        pass


    ## post-simulation functions 

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
            html = basic_table_to_html(trades_df, 'BackTest Trades')
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
            trades_series_df = self.format_df(self.test_ledger)
            html = basic_table_to_html(trades_series_df, 'Backtest Series')
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

    def mark_to_market(self, mark_price):
        tick_size = self.security.tick_size
        tick_value = self.security.tick_value
        m = tick_value/tick_size
    
        return m * (mark_price - self.current_trade['Entry']) * self.current_trade['Position']
       
    def record_backtest_data(self, bar):

        row = dict( 
                Date=bar['Date'],
                Close=bar['Close'],
                InSignal="",
                Entry=None,
                ExSignal="",
                Exit=None,
                Position=None,
                StopLevel=None,
                MTM= None,
                Equity=None
            ) 

        mtm = 0 
        if not self.FLAT:
            if bar['Date'] == self.current_trade['InDate']:
                row['InSignal'] = self.current_trade['InSignal']
                row['Entry'] = self.current_trade['Entry']

            row['Position'] = self.current_trade['Position']
            row['StopLevel'] = self.current_trade['StopLevel']

            if bar['Date'] == self.current_trade.get('ExDate'):
                row['ExSignal'] = self.current_trade['ExSignal']
                row['Exit'] = self.current_trade['Exit']

            ## mark on trade value if completed 
            if self.current_trade.get('Exit'):
                mtm = self.current_trade['Value']
            else:
                ## otherwise mark on close
                mtm = self.mark_to_market( bar['Close']) 

        row['MTM'] =  mtm
        row['Equity'] = self.wallet + mtm 

        return row


    def calc_strategy_analytics(self, cur_dt, bar, ref_bar):
        pass

        ## create all derived data and indicators here 
        ## return a dictionaty of analytics you want outputted 
        ## if desired:
        ## eaxmple:
        ## analytics_dict = dict(ma5=self.ma.valueAt(0), etc...)

        return None

    def merge_analytics(self, backtest_dict, analytics_dict, insert_at=2):
        if analytics_dict:
            b_keys = list(backtest_dict.keys())
            a_keys = list(analytics_dict.keys())

            """ backtest_dict format
            row = dict( 
                Date=bar['Date'],
                Close=bar['Close'],
                InSignal="",
                Entry=None,
                ExSignal="",
                Exit=None,
                Position=None,
                StopLevel=None,
                MTM= None,
                Equity=None
            ) 
            """
            # place analytics right after Close price (insert_at=2) by default
            # note if you want to add any bar data other than the Close
            # just include in the analytics
            ordered_keys = b_keys[:insert_at] + a_keys + b_keys[insert_at:]
            new_dict = backtest_dict.copy()
            new_dict.update(analytics_dict)
            new_dict = { k:new_dict[k] for k in ordered_keys }
            return new_dict

        return backtest_dict

    ## loads the next price bar 
    def load(self):
        try:
            return self.security.next_bar()
        except StopIteration:
            return None  


    ## runs throught the strategy on ONE bar
    def execute(self, bar_data):

        i, cur_dt, bar = bar_data

        self.backtest_enabled = True 
        if self.start_dt and cur_dt < self.start_dt:
            self.backtest_enabled = False 

        ref_bar = None
        if self.ref_index is not None:
            ref_tuple = self.ref_index.fetch_bar(bar['Date'])
            ## unpack index, date, and bar from reference
            ref_i, ref_cur_dt, ref_bar = ref_tuple
        
        self.exit_OPEN(cur_dt, bar, ref_bar)
        self.entry_OPEN(cur_dt, bar, ref_bar)


        ## do analytics here
        analytics_dict = self.calc_strategy_analytics(cur_dt, bar, ref_bar )

        ## track volatility
        self.stdev.push(bar['Close'])


        # collect all analytics, but don't start trading until we
        # hit the the start_from_dt trading date

        if not self.FLAT:
            self.current_trade['Duration'] += 1
            ## self.current_trade['CurDate'] = cur_dt.strftime("%Y-%m-%d")
            ## print(json.dumps(self.current_trade, indent=4))

        self.exit_CLOSE(cur_dt, bar, ref_bar)
        self.trade_update(cur_dt, bar, ref_bar)
        self.entry_CLOSE(cur_dt, bar, ref_bar)


        ## record stats ONLY when the backtest kicks in
        if self.backtest_enabled == False:
            return

        ## record trade info for the day.
        ## merge in any analytics info you want as part of the 
        ## test_ledger output
        backtest_dict = self.record_backtest_data( bar )
        if self.portfolio:
            ## mark the current position side if an open trade
            record_tag = self.name
            ## tag current LONGS with +sign, SHORTS -sign
            if self.LONG: record_tag = f'+{self.name}'
            if self.SHORT: record_tag = f'-{self.name}'
            portfolio_dict = self.merge_analytics( backtest_dict, dict(Open=bar['Open']), insert_at=1 )
            self.portfolio.post_record( record_tag, portfolio_dict )

        backtest_dict = self.merge_analytics(backtest_dict, analytics_dict)

        if self.backtest_enabled:
            self.test_ledger.append( backtest_dict )

        ## reset trade
        if self.CLOSED:
            self.wallet += self.current_trade['Value'] 
            if self.portfolio:
                self.portfolio.post_trade( self.name, self._format_closed_trade() )
            self.current_trade = None
        
        if self.FLAT:
            self.high_marker = self.low_marker = None
            self.update_position_limit()
            self.update_wallet_alloc()
            self.update_dollar_limit()



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

