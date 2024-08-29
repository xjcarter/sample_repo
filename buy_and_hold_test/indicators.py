

from collections import deque
import pandas
import numpy
import statistics
from datetime import datetime, timedelta
import math
import calendar_calcs


def to_date(date_object):
    if isinstance(date_object, str):
        return datetime.strptime(date_object, "%Y-%m-%d").date()
    return date_object

def _weekday(date_object):
    return to_date(date_object).weekday()

## consumes an entire dataframe and returns
## a parallel timeseries of the desired indicator
def indicator_to_df(stock_df, indicator, name='Value', merge=False):

    derived = []
    for i in range(stock_df.shape[0]):
        index = stock_df.index[i]
        stock_bar = stock_df.loc[index]
        v = indicator.push(stock_bar)
        if v is None: v = numpy.nan
        derived.append( {'Date':stock_bar['Date'], name:v })

    new_df = pandas.DataFrame(derived)
    if merge:
        new_df = pandas.merge(stock_df, new_df, on='Date', how='left')

    return new_df

class Indicator(object):
    def __init__(self, history_len, derived_len=None):
        self.history = deque()
        self.derived = deque()
        self.history_len = history_len
        self.derived_len = derived_len

    def push(self, data, valueAt=0):
        t_data = self.transform(data)
        self.history.append(t_data)

        old_len = len(self.derived)

        ## flag when new derived data is available
        self._calculate()
        new_len = len(self.derived)

        if self.derived_len and len(self.derived) > self.derived_len: self.derived.popleft()
        if len(self.history) > self.history_len: self.history.popleft()
        if new_len > old_len:
            ## return requested history value, default is the current value: self.valueAt(0)
            return self.valueAt(idx=valueAt)
        else:
            return None

    def count(self):
        return len(self.derived)

    ## external mechanism to attach at transform function
    ## without having to write a new indicator
    def attach_transform(self, tfunc):
        self.transform = tfunc

    def transform(self, data_point):
        #transform historical data point before feeding it to indicator calcs
        #this allows us to pick of a value in a dataframe, combine dataframe values, etc..
        #example - see EMA versus CloseEMA:
        #EMA works on a single datapoint value, where CloseEMA provides that specialized datapoint as df_bar['Close']

        return data_point

    def _calculate(self):
        return None

    def valueAt(self, idx):
        if idx >= 0 and len(self.derived) >= idx+1:
            i = -1
            if idx > 0: i = -(idx+1)
            return self.derived[i]
        else:
            return None

def cross_up(timeseries, threshold, front_index, back_index):
    x_up = False
    front = back = None
    if isinstance(threshold, Indicator):
        front = threshold.valueAt(front_index)
        back = threshold.valueAt(back_index)
    else:
        ## constant value
        front = back = threshold
        
    if front != None and back != None:
        if timeseries.valueAt(front_index) > front:
            if timeseries.valueAt(back_index) <= back: 
                x_up = True

    return x_up

def cross_dwn(timeseries, threshold, front_index, back_index):
    x_dwn = False
    front = back = None
    if isinstance(threshold, Indicator):
        front = threshold.valueAt(front_index)
        back = threshold.valueAt(back_index)
    else:
        ## constant value
        front = back = threshold
        
    if front != None and back != None:
        if timeseries.valueAt(front_index) < front:
            if timeseries.valueAt(back_index) >= back: 
                x_dwn = True

    return x_dwn

class HighestValue():
    def __init__(self, v, history=50):
        self.highest = v
        self.history = history
        self.highs = deque()
        self.highs.append(v)

    def push(self, new_value):
        self.highest = max(self.highest, new_value)
        self.highs.append(self.highest)
        if len(self.highs) > self.history:
            self.highs.popleft()

class LowestValue():
    def __init__(self, v, history=50):
        self.lowest = v
        self.history = history
        self.lows = deque()
        self.lows.append(v)

    def push(self, new_value):
        self.lowest = min(self.lowest, new_value)
        self.lows.append(self.lowest)
        if len(self.lows) > self.history:
            self.lows.popleft()

class DataSeries(Indicator):
    def __init__(self, derived_len=50):
        super().__init__(history_len=0, derived_len=derived_len)

    def _calculate(self):
        self.derived.append(self.history[-1])

    def highest(self, count):
        if len(self.derived) >= count:
            return max(list(self.derived)[-count:])
        return None

    def lowest(self, count):
        if len(self.derived) >= count:
            return min(list(self.derived)[-count:])
        return None

class Runner(Indicator):
    ## seeks out N up/down days in a row
    ## returned value = -(total of returns) (down run), 0 (no run), or +(total of returns) (up run)
    def __init__(self, run_count, derived_len=10):
        super().__init__(history_len=derived_len+1, derived_len=derived_len)
        self.run_count = run_count

    def _calculate(self):
        run = 0
        group = []
        if len(self.history) > self.run_count:
            i = 0
            while i < self.run_count:
                v = self.history[-(i+1)]/self.history[-(i+2)] - 1
                group.append(v)
                i += 1
            tot = sum(group)
            sum_of_abs = sum([abs(x) for x in group])
            if abs(tot) == sum_of_abs:
                run = tot
        self.derived.append((run, group))

class WeeklyBar(Indicator):
    def __init__(self,anchor_day=None):
        super().__init__(history_len=10)
        self.holidays = calendar_calcs.load_holidays()
        self.open = self.high = self.low = self.close = None
        self.volume = 0
        ## optional market where a week begins nad wends
        ## i.e. anchor_day = 4, end the week on Friday
        self.anchor_day = anchor_day
        self.present_week = None
        self.present_eow = None

    def _clear_week(self):
        self.open = self.high = self.low = self.close = None
        self.volume = 0

    def get_week(self, data_dt):
        ONE_WEEK = 7
        week  = 0
        prev = data_dt
        while prev.month == data_dt.month:
            prev -= timedelta(days=ONE_WEEK)
            week += 1
        return week

    def get_present_week(self, data_dt):
        ONE_WEEK = 7
        if calendar_calcs.is_end_of_week(data_dt, self.holidays):
            self.present_week = self.get_week(data_dt)
            self.present_eow = data_dt
            return self.present_week
        else:
            if self.present_week is not None:
                next_week = self.present_eow + timedelta(days=ONE_WEEK)
                if next_week.month == self.present_eow.month:
                    return self.present_week + 1
                else:
                    return 1
            return None

    def _calculate(self):
        ## expecting a OHLC bar to be pushed
        daily_bar = self.history[-1]

        if self.open is None: self.open = daily_bar['Open']
        if self.high is not None:
            self.high = max(self.high, daily_bar['High'])
        else:
            self.high = daily_bar['High']
        if self.low is not None:
            self.low = min(self.low, daily_bar['Low'])
        else:
            self.low = daily_bar['Low']
        self.close = daily_bar['Close']
        self.volume += daily_bar['Volume']

        data_dt = datetime.strptime(daily_bar['Date'],"%Y-%m-%d").date()

        bundle_bar = False
        if self.anchor_day is not None:
            bundle_bar = calendar_calcs.is_day_of_week(self.anchor_day, data_dt, self.holidays)
        else:
            bundle_bar = calendar_calcs.is_end_of_week(data_dt, self.holidays)
        if bundle_bar:
            v = {
                    'Date':daily_bar['Date'],
                    'Week':self.get_week(data_dt),
                    'Open':self.open,
                    'High':self.high,
                    'Low':self.low,
                    'Close':self.close,
                    'Volume':self.volume
            }
            self.derived.append(v)
            self._clear_week()

    def convert_daily_dataframe(self, stock_df):
        for i in range(stock_df.shape[0]):
            idate = stock_df.index[i]
            stock_bar = stock_df.loc[idate]
            self.push(stock_bar)

        return pandas.DataFrame(self.derived)



class Mo(Indicator):
    def __init__(self, history_len, derived_len=50):
        super().__init__(history_len, derived_len)

    def _calculate(self):
        if len(self.history) >= self.history_len:
            n = self.history[-1]
            p = self.history[-self.history_len]
            m = n - p
            self.derived.append(m)

class LogRtn(Indicator):
    def __init__(self, history_len, derived_len=50):
        super().__init__(history_len, derived_len)

    def _calculate(self):
        if len(self.history) >= self.history_len:
            n = self.history[-1]
            p = self.history[-self.history_len]
            m = math.log(n/p)
            self.derived.append(m)

class MA(Indicator):
    def __init__(self, history_len, derived_len=50):
        super().__init__(history_len, derived_len=derived_len)

    def _calculate(self):
        if len(self.history) >= self.history_len:
            m = list(self.history)[-self.history_len:]
            self.derived.append(statistics.mean(m))

class TrueRange(Indicator):
    ## Average True Range 
    def __init__(self, derived_len=50):
        super().__init__(history_len=2, derived_len=derived_len)

    def _calculate(self):
        if len(self.history) >= self.history_len:
            prev_bar, bar  = self.history[-2], self.history[-1]
            r1 = bar['High'] - bar['Low']
            r2 = bar['High'] - prev_bar['Close']
            r3 = prev_bar['Close'] - bar['Low']
            self.derived.append(max(r1,r2,r3))

class IBS(Indicator):
    ## internal bar strength iindicator
    ## (Close - Low)/(High - Low)
    def __init__(self, derived_len=50):
        super().__init__(history_len=1, derived_len=derived_len)

    def _calculate(self):
        if len(self.history) >= self.history_len:
            bar = self.history[-1]
            ibs = (bar['Close'] - bar['Low'])/(bar['High'] - bar['Low']) * 100
            self.derived.append(ibs)

class StochK(Indicator):
    ## price relative to range over N days
    ## (Close - Low)/(High - Low)
    def __init__(self, history_len, derived_len=50):
        super().__init__(history_len=history_len, derived_len=derived_len)

    def _calculate(self):
        if len(self.history) >= self.history_len:
            if len(self.history) >= self.history_len:
                v =list(self.history)[-self.history_len:]
                bar = self.history[-1]
                lowest = min([ x['Low'] for x in v])
                highest = max([ x['High'] for x in v])
                try:
                    stoch_k = (bar['Close'] - lowest)/(highest - lowest) * 100
                    self.derived.append(stoch_k)
                except:
                    pass


class RSI(Indicator):
    def __init__(self, history_len, derived_len=50, coeff=None):
        super().__init__(history_len, derived_len)
        self.INF = 1e10
        self.upv = None
        self.dnv = None
        self.coeff = coeff
    
    def _calculate(self):
        if len(self.history) >= self.history_len+1:
            # subtract current prices a[1:], from last prices a[:-1]
            # remember the most current price is the last price in the deque/list
            chgs = [x[0] - x[1] for x in zip(list(self.history)[1:],list(self.history)[:-1])]
            ups = sum([ x for x in chgs if x >= 0 ])
            dns = sum([ abs(x) for x in chgs if x < 0 ])

            a = 1.0/self.history_len
            ## allow for custom weighting: coeff = 1, simple MA,  coeff = 2/(N+1), standard EMA
            if self.coeff is not None: a = self.coeff

            upv = a * ups + (1-a) * self.upv if self.upv is not None else ups 
            dnv = a * dns + (1-a) * self.dnv if self.dnv is not None else dns
            self.upv, self.dnv = upv, dnv
            if dnv == 0: dnv = self.INF

            rsi = 100.0 - 100.0/(1+(upv/dnv))
            self.derived.append(rsi)


class CutlersRSI(Indicator):
    def __init__(self, history_len, derived_len=50):
        super().__init__(history_len, derived_len)

    def _calculate(self):
        if len(self.history) >= self.history_len+1:
            # subtract current prices a[1:], from last prices a[:-1]
            # remember the most current price is the last price in the deque/list
            chgs = [x[0] - x[1] for x in zip(list(self.history)[1:],list(self.history)[:-1])]
            ups = sum([ x for x in chgs if x >= 0 ])
            dns = sum([ abs(x) for x in chgs if x < 0 ])

            rsi = None
            if dns == 0:
                rsi = 100 
            else:
                rsi = 100.0 - 100.0/(1+(ups/dns))

            self.derived.append(rsi)

## synthethic VIX calculation that is highly correlated with the actual VIX
## (when applied to SPY)
class SyntheticVIX(Indicator):
    def __init__(self, history_len=22, derived_len=50):
        super().__init__(history_len, derived_len)

    def _calculate(self):
        if len(self.history) >= self.history_len:
            highest = max([x['Close'] for x in self.history])
            low_today = self.history[-1]['Low']
            vix = 100 * (highest - low_today)/highest
            self.derived.append(vix)
            

class Thanos(Indicator):
    def __init__(self, ma_len, no_of_samples, derived_len=50):
        # ma_len = benchmark moving average
        # no_of_samples = cnt of devations vs the ma needed to calculate zscore 
        super().__init__(history_len=(ma_len+no_of_samples), derived_len=derived_len)
        self.deviations = deque()
        self.ma_len = ma_len
        self.no_of_samples = no_of_samples

    def _calculate(self):
        if len(self.history) >= self.ma_len:
            ma = statistics.mean(list(self.history)[-self.ma_len:])
            dev = math.log(self.history[-1]/ma)
            self.deviations.append(dev)
            self.history.popleft()

            if len(self.deviations) >= self.no_of_samples:
                samples = list(self.deviations)[-self.no_of_samples:]
                z = (dev - statistics.mean(samples))/statistics.pstdev(samples)
                self.derived.append(z)
                self.deviations.popleft()


class StDev(Indicator):
    def __init__(self, sample_size, derived_len=50):
        super().__init__(history_len=sample_size, derived_len=derived_len)

    def _calculate(self):
        if len(self.history) >= self.history_len:
            m = list(self.history)[-self.history_len:]
            v = pandas.Series(data=m)
            self.derived.append(v.std())

class Corr(Indicator):
    ## correlation - expects pairs to be pushed (price1, price2)
    ## calculates the correlation of the returns of time series price1 vs time series price2
    def __init__(self, sample_size, derived_len=50):
        super().__init__(history_len=sample_size+1, derived_len=derived_len)

    def returns(self, price_array):
        p = price_array
        rtns = []
        for i in range(1, len(p)):
            r = math.log(p[i]/p[i-1])
            rtns.append(r)
        return rtns

    def _calculate(self):
        if len(self.history) >= self.history_len:
            pairs = list(self.history)[-self.history_len:]
            a = pandas.Series(data= self.returns([ x[0] for x in pairs ]))
            b = pandas.Series(data= self.returns([ x[1] for x in pairs ]))
            m = a.corr(b)
            self.derived.append(m)

class Beta(Indicator):
    ## beta - expects pairs to be pushed (price1, price2)
    ## calculates the beta of the returns of time series price1 vs time series price2
    def __init__(self, sample_size, derived_len=50):
        super().__init__(history_len=sample_size+1, derived_len=derived_len)

    def returns(self, price_array):
        p = price_array
        rtns = []
        for i in range(1, len(p)):
            r = math.log(p[i]/p[i-1])
            rtns.append(r)
        return rtns

    def _calculate(self):
        if len(self.history) >= self.history_len:
            pairs = list(self.history)[-self.history_len:]
            a = pandas.Series(data= self.returns([ x[0] for x in pairs ]))
            b = pandas.Series(data= self.returns([ x[1] for x in pairs ]))
            m = a.cov(b)/b.var()
            self.derived.append(m)

class Median(Indicator):
    def __init__(self, sample_size, derived_len=50):
        super().__init__(history_len=sample_size, derived_len=derived_len)
        self.sample_sz = sample_size

    def _calculate(self):
        if len(self.history) >= self.history_len:
            m = statistics.median(list(self.history)[-self.sample_sz:])
            self.derived.append(m)

class ZScore(Indicator):
    def __init__(self, sample_size, derived_len=50):
        super().__init__(history_len=sample_size+1, derived_len=derived_len)
        self.sample_sz = sample_size+1

    def _calculate(self):
        if len(self.history) >= self.history_len:
            pop = list(self.history)[-(self.sample_sz):-2]
            v = self.history[-1]
            s = statistics.pstdev(pop)
            m = statistics.mean(pop)
            self.derived.append((v-m)/s)

class EMA(Indicator):
    def __init__(self, coeff, history_len, derived_len=50):
        super().__init__(history_len, derived_len)
        self.coeff = coeff
        self.prev = None

    def _calculate(self):
        n = self.history[-1]
        if self.prev is not None:
            self.prev = (self.coeff * n) + (1 - self.coeff) * self.prev
            self.derived.append(self.prev)
        else:
            if len(self.history) >= self.history_len:
                v = list(self.history)[-self.history_len:]
                self.prev = statistics.mean(v)
                self.derived.append(self.prev)

class LastLow(Indicator):
    def __init__(self, last_len, derived_len=50):
        super().__init__(last_len, derived_len)
        self.last_len = last_len

    def _calculate(self):
        if len(self.history) >= self.history_len:
            lowest = min(list(self.history)[-self.last_len:])
            self.derived.append(lowest)
            return lowest
        else:
            return None

class LastHigh(Indicator):
    def __init__(self, last_len, derived_len=50):
        super().__init__(last_len, derived_len)
        self.last_len = last_len

    def _calculate(self):
        if len(self.history) >= self.history_len:
            highest = max(list(self.history)[-self.last_len:])
            self.derived.append(highest)
            return highest 
        else:
            return None


class MACD(Indicator):
    def __init__(self, long_len, short_len, signal_len, warmup=10, history_len=50, derived_len=50):
        super().__init__(history_len, derived_len)
        self.q, self.k, self.sig  = 2.0/long_len, 2.0/short_len, 2.0/signal_len
        self.warmup = warmup 
        self.signal_warmup = signal_len
        self.counter = 0
        self.signal_counter = 0
        self.prev_q, self.prev_k, self.prev_sig = None, None, None

    def _calculate(self):
        n = self.history[-1]
        if self.prev_q is not None:
            self.prev_q = (self.q * n) + (1 - self.q) * self.prev_q
        else:
            self.prev_q = n
        if self.prev_k is not None:
            self.prev_k = (self.k * n) + (1 - self.k) * self.prev_k
        else:
            self.prev_k = n

        self.counter += 1

        if self.counter >= self.warmup:
            diff = self.prev_k - self.prev_q
            if self.prev_sig is not None:
                self.prev_sig = (self.sig * diff) + (1 - self.sig) * self.prev_sig
            else:
                self.prev_sig = diff
            
            self.counter = self.warmup  #just to prevent counter rollover
            self.signal_counter += 1

            macd = None 
            if self.signal_counter >= self.signal_warmup:
                macd = diff - self.prev_sig
                self.derived.append(macd)
                self.signal_counter = self.signal_warmup  #just to prevent counter rollover
            return macd
        else:
            return None


class CloseEMA(EMA):
    def __init__(self, coeff, warmup, history_len, derived_len=50):
        super().__init__(coeff, warmup, history_len, derived_len)

    def transform(self, df_bar):
        return df_bar['Close']


class CloseMACD(MACD):
    def __init__(self, long_len, short_len, signal_len, warmup=10, history_len=50, derived_len=50):
        super().__init__(long_len, short_len, signal_len, warmup, history_len, derived_len)

    def transform(self, df_bar):
        return df_bar['Close']

class WeightedClose(Indicator):
    def __init__(self, history_len, derived_len=50):
        super().__init__(history_len, derived_len)

    def _calculate(self):
        n = self.history[-1]
        wp = ( n['Close'] + n['High'] + n['Low'] )/3.0
        self.derived.append(wp)
        return self.derived[-1]

## uses first day of the week range as breakout points
## requires true OHLC bar information from a DataFrame and date as a tuple -> (dt, OHLC bar)
## pushes to self.derived (current anchor, breakout amt above/below anchor range)
class MondayAnchor(Indicator):
    def __init__(self, derived_len=50):
        super().__init__(history_len=1, derived_len=derived_len)
        self.holidays = calendar_calcs.load_holidays() 
        self.anchor = None

    def _calculate(self):
        ## expecting (datetime.date, OHLC bar) tuple
        dt, bar = self.history[-1]
        if calendar_calcs.is_start_of_week(dt, self.holidays):
            self.anchor = bar
            ## record new anchor with no breakout
            self.derived.append((self.anchor, 0))
        else:
            ## record breakouts above or below current anchor
            if self.anchor is not None:
                if bar['Close'] > self.anchor['High']:
                    self.derived.append((self.anchor, bar['Close'] - self.anchor['High']))
                elif bar['Close'] < self.anchor['Low']:
                    self.derived.append((self.anchor, bar['Close'] - self.anchor['Low']))
                else:
                    self.derived.append((self.anchor, 0))

class Anchor(Indicator):
    def __init__(self, day, derived_len=50):
        super().__init__(history_len=1, derived_len=derived_len)
        self.holidays = calendar_calcs.load_holidays()
        self.anchor = None
        self.day = day

    def _calculate(self):
        ## expecting (datetime.date, OHLC bar) tuple
        dt, bar = self.history[-1]
        if calendar_calcs.is_day_of_week(self.day, dt, self.holidays):
            self.anchor = bar
            ## record new anchor with no breakout
            self.derived.append((self.anchor, 0))
        else:
            ## record breakouts above or below current anchor
            if self.anchor is not None:
                if bar['Close'] > self.anchor['High']:
                    self.derived.append((self.anchor, bar['Close'] - self.anchor['High']))
                elif bar['Close'] < self.anchor['Low']:
                    self.derived.append((self.anchor, bar['Close'] - self.anchor['Low']))
                else:
                    self.derived.append((self.anchor, 0))


## uses last day of the week range as breakout points
class LastDayAnchor(Indicator):
    def __init__(self, derived_len=50):
        super().__init__(history_len=1, derived_len=derived_len)
        self.holidays = calendar_calcs.load_holidays() 
        self.anchor = None

    def _calculate(self):
        ## expecting (datetime.date, OHLC bar) tuple
        dt, bar = self.history[-1]
        if calendar_calcs.is_end_of_week(dt, self.holidays):
            self.anchor = bar
            ## record new anchor with no breakout
            self.derived.append((self.anchor, 0))
        else:
            ## record breakouts above or below current anchor
            if self.anchor is not None:
                if bar['Close'] > self.anchor['High']:
                    self.derived.append((self.anchor, bar['Close'] - self.anchor['High']))
                elif bar['Close'] < self.anchor['Low']:
                    self.derived.append((self.anchor, bar['Close'] - self.anchor['Low']))
                else:
                    self.derived.append((self.anchor, 0))


## returns a tuple of cardinal alue of current trading day (week, month, year)
## expecting successive dt from a progressing timeseries 
class TradingDayMarker(Indicator):
    def __init__(self, derived_len=None):
        super().__init__(history_len=1, derived_len=derived_len)
        self.prev_year = None
        self.prev_month = None
        self.wk = self.mth = self.yr = None

    def _calculate(self):
        ## expecting successive dt 
        dt = self.history[-1]
        mark_date = pandas.to_datetime(dt)
        self.wk = mark_date.weekday()
        month = mark_date.month
        year = mark_date.year

        if self.prev_month is None: self.prev_month = month
        if self.prev_year is None: self.prev_year = year 

        if month != self.prev_month:
            self.mth = 1
            self.prev_month = month
        elif self.mth is not None:
            self.mth += 1

        if year != self.prev_year:
            self.yr = 1
            self.prev_year = year
        elif self.yr is not None:
            self.yr += 1

        self.derived.append((self.wk, self.mth, self.yr))


def create_simulated_timeseries(length):
    from datetime import date, timedelta
    from random import randint

    cols = 'Date Open High Low Close Day'.split()

    df = None
    opn = 100
    dt = date(2022,5,10)
    for i in range(length):
        hi = opn + randint(1,100)/100.0
        lo = opn - randint(1,100)/100.0
        r = (hi-lo)
        close = (r * randint(0,100)/100.0) + lo
        bar = pandas.DataFrame(columns=cols,data=[[dt,opn,hi,lo,close,dt.weekday()]])
        opn = close + (r * randint(-200,200)/100.0)
        dt = dt+timedelta(days=3) if dt.weekday() == 4 else dt+timedelta(days=1)
        if df is None:
            df = pandas.DataFrame(bar)
        else:
            df = pandas.concat([df,bar])

    df[cols] = df[cols].round(2) 
    df.set_index('Date', inplace=True)

    return df


def test_monday_anchor():

    df = create_simulated_timeseries(length=16)
    print(df)

    m = MondayAnchor()
    for i in range(df.shape[0]):
        cur_dt = df.index[i]
        v = m.push((cur_dt, df.loc[cur_dt]))
        print("bar = ", df.loc[cur_dt])
        print(v)
        print("")

def test_atr():

    df = create_simulated_timeseries(length=16)

    tr = TrueRange()
    atr = MA(10)
    
    results = []
    for i in range(df.shape[0]):
        v = q = None
        bar = df.iloc[i]
        v = tr.push(bar)
        if v is not None:
            q = atr.push(v)
            if q is not None: q = round(q,3)
            v = round(v,3)
        p = bar.to_dict()
        p.update(dict(TR=v, ATR=q))
        results.append(p)

    xf = pandas.DataFrame(results)
    print(xf)
    print(atr.derived)


def test_ema():

    e = EMA(coeff=0.3, warmup=10, history_len=50, derived_len=20)

    from random import randint
    changes = [ randint(-100,100)/100.0 for x in range(100) ]
    changes[0] = 100
    prices = [ sum(changes[0:i]) for i in range(1,101) ]
    for i, p in enumerate(prices):
        v = e.push(p)
        print(f'{i:03d} {p:10.4f} {v}')

    print(e.valueAt(2))
    print(e.valueAt(1))
    print(e.valueAt(0))

def test_correlation():

    e = Corr(sample_size=6)
   
    prices = [ (43, 99), (21, 65), (25,79), (42,75), (57,87), (59,81) ]
    for i, p in enumerate(prices):
        v = e.push(p)
        print(f'{i:03d} {p} {v}')

def test_beta():

    e = Beta(sample_size=6)
   
    prices = [ (43, 99), (21, 65), (25,79), (42,75), (57,87), (59,81) ]
    for i, p in enumerate(prices):
        v = e.push(p)
        print(f'{i:03d} {p} {v}')

def test_stdev():

    e = StDev(sample_size=6)
   
    prices = [ (43, 99), (21, 65), (25,79), (42,75), (57,87), (59,81) ]
    for i, p in enumerate(prices):
        v = e.push(p[0])
        print(f'{i:03d} {p} {v}')

    m = [x[0] for x in prices]
    j = pandas.Series(data=m)
    print(j.std())

def test_zscore():

    e = ZScore(sample_size=50)

    from random import randint
    changes = [ randint(-100,100)/100.0 for x in range(60) ]
    changes[0] = 100
    prices = [ sum(changes[0:i]) for i in range(1,61) ]
    for i, p in enumerate(prices):
        v = e.push(p)
        print(f'{i:03d} {p:10.4f} {v}')


def test_macd():

    e = MACD(long_len=28, short_len=12, signal_len=9, history_len=50, derived_len=20)

    from random import randint
    changes = [ randint(-100,100)/100.0 for x in range(100) ]
    changes[0] = 100
    prices = [ sum(changes[0:i]) for i in range(1,101) ]
    for i, p in enumerate(prices):
        v = e.push(p)
        #print(f'{i:03d} {p:10.4f} {v}')

def test_cross_up():

    e = MA(9)
    d = DataSeries()

    from random import randint
    changes = [ randint(-100,100)/100.0 for x in range(100) ]
    changes[0] = 100
    prices = [ sum(changes[0:i]) for i in range(1,101) ]
    for i, p in enumerate(prices):
        v = e.push(p)
        x = d.push(p)
        x_up = cross_up(d,e,0,1)
        above = ''
        if v is not None and x is not None and v < x: above = 'ABOVE'
        print(f'{i:03d} {p:10.4f} ma={v}   {x_up}  {above}')

def test_cross_dwn():

    e = MA(9)
    d = DataSeries()

    from random import randint
    changes = [ randint(-100,100)/100.0 for x in range(100) ]
    changes[0] = 100
    prices = [ sum(changes[0:i]) for i in range(1,101) ]
    for i, p in enumerate(prices):
        v = e.push(p)
        x = d.push(p)
        x_dn = cross_dwn(d,e,0,1)
        below = ''
        if v is not None and x is not None and v > x: below = 'BELOW'
        print(f'{i:03d} {p:10.4f} ma={v}   {x_dn}  {below}')


def test_last_low():

    e = LastLow(last_len=5)

    from random import randint
    changes = [ randint(-100,100)/100.0 for x in range(20) ]
    changes[0] = 100
    prices = [ sum(changes[0:i]) for i in range(1,21) ]
    for i, p in enumerate(prices):
        v = e.push(p)
        print(f'{i:03d} {p:10.4f} {v}')
    print(len(e.history))

def test_rsi():

    e = RSI(14)

    from random import randint
    samples = 500
    changes = [ randint(-300,300)/100.0 for x in range(samples) ]
    changes[0] = 100
    prices = [ sum(changes[0:i]) for i in range(1,samples+1) ]
    for i, p in enumerate(prices):
        v = e.push(p)
        print(f'{i:03d} {p:10.4f} {v}')
    print(len(e.history))

def test_cutlers_rsi():

    e = CutlersRSI(3)

    from random import randint
    samples = 20
    changes = [ randint(-300,300)/100.0 for x in range(samples) ]
    changes[0] = 100
    prices = [ sum(changes[0:i]) for i in range(1,samples+1) ]
    for i, p in enumerate(prices):
        v = e.push(p)
        print(f'{i:03d} {p:10.4f} {v}')
    print(len(e.history))


def test_dataseries():

    e = DataSeries(14)

    from random import randint
    samples = 50
    changes = [ randint(-300,300)/100.0 for x in range(samples) ]
    changes[0] = 100
    prices = [ sum(changes[0:i]) for i in range(1,samples+1) ]
    for i, p in enumerate(prices):
        v = e.push(p)
        print(f'{i:03d} {p:10.4f} {v}')
    print(e.derived)
    print(e.valueAt(3))


def test_thanos():

    e = Thanos(ma_len=50, no_of_samples=20)

    from random import randint
    samples = 500
    changes = [ randint(-300,300)/100.0 for x in range(samples) ]
    changes[0] = 100
    prices = [ sum(changes[0:i]) for i in range(1,samples+1) ]
    for i, p in enumerate(prices):
        v = e.push(p)
        print(f'{i:03d} {p:10.4f} {v}')



if __name__ == '__main__':
    #test_thanos()
    #test_rsi()
    #test_cutlers_rsi()
    #test_dataseries()
    #test_monday_anchor()
    #test_cross_dwn()
    #test_ema()
    #test_correlation()
    #test_stdev()
    #test_beta()
    #test_macd()
    #test_last_low()
    test_atr()



