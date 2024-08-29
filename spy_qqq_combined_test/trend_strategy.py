from datetime import date, datetime
from indicators import StDev, MA
from backtest import BackTest, TradeType, DumpFormat
import calendar_calcs

## example LONG only 10/40 MA crossover trend following strategy
## !!! NOT A PRODUCTION STRATEGY - FOR EXAMPLE ONLY

class Trend_Strategy(BackTest):
    def __init__(self, security, json_config, ref_index=None):
        super().__init__(security, json_config, ref_index)

        ## needed indicators and tools

        self.settings = self.config.get('settings')

        self.stdev = StDev( sample_size= self.settings.get('StDev', 50) )

        ## 10 day moving average - average(10 closing prices)
        self.ma10 = MA(self.settings.get("short_len",10))
        ## 40 day moving average - average(40 closing prices)
        self.ma40 = MA(self.settings.get("long_len",40))

    def calc_strategy_analytics(self, cur_dt, bar, ref_bar):

        def _round(v,p):
            if v is not None:
                return round(v,p)
            return ""

        close_p = bar['Close']

        ## create all derived data and indicatirs here 
        self.stdev.push(close_p)

        w10 = _round(self.ma10.push(close_p),3)
        w40 = _round(self.ma40.push(close_p),3)

        a_dict = dict(Close=f'{round(close_p,2)}', MA10=w10, MA40=w40)

        return a_dict



    def exit_OPEN(self, cur_dt, bar, ref_bar=None):
        if self.backtest_enabled == False:
            return

        if self.FLAT:
            return

        if self.LONG:

            ## 10/40 crossover down
            if self.ma40.count() > 1 and self.ma10.count() > 1:
                x = self.ma10.valueAt(0) - self.ma40.valueAt(0)
                y = self.ma10.valueAt(1) - self.ma40.valueAt(1)
                if x < 0 and x*y <= 0 :
                    self.exit_trade( bar['Date'], self.security, bar['Open'], label='MA-' )
            elif bar['Close'] <= self.current_trade['StopLevel']:
                self.exit_trade( bar['Date'], self.security, bar['Close'], label='STOP_OUT' )


    def entry_OPEN(self, cur_dt, bar, ref_bar=None):
        if self.backtest_enabled == False:
            return

        if not self.FLAT:
            return

        ## 10/40 crossover up 
        if self.ma40.count() > 1 and self.ma10.count() > 1:
            x = self.ma10.valueAt(0) - self.ma40.valueAt(0)
            y = self.ma10.valueAt(1) - self.ma40.valueAt(1)
            if x > 0 and x*y <= 0:
                self.current_trade = self.enter_trade( TradeType.BUY, bar['Date'], self.security, bar['Open'], label='MA+' )
            if self.LONG:
                initial_stop = self.initialize_stop(anchor= bar['Open'])
                self.current_trade['StopLevel'] = self.calc_price_stop( initial_stop.highest )


    def exit_CLOSE(self, cur_dt, bar, ref_bar=None):
        if self.backtest_enabled == False:
            return

        if self.FLAT:
            return


    def entry_CLOSE(self, cur_dt, bar, ref_bar=None):
        if self.backtest_enabled == False:
            return

        if not self.FLAT:
            return

        ## ADD entry_CLOSE SIGNAL LOGIC HERE
        ## self.current_trade = self.enter_trade( TradeType.BUY, bar['Date'], self.security, bar['Close'], label = "" )


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
