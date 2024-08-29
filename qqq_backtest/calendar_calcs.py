
import pandas
import calendar
from datetime import date, datetime, timedelta

MONDAY = 0
TUESDAY = 1 
WEDNESDAY = 2 
THURSDAY = 3 
FRIDAY = 4
SATURDAY = 5
SUNDAY = 6
WEEKDAYS = ['MON','TUE','WED','THU','FRI','SAT','SUN']

def cvt_date(str):
    return datetime.strptime(str,"%Y-%m-%d").date()

def weekday(dt):
    return cvt_date(dt).weekday()

def business_days(trade_date):
    cal = calendar.Calendar()
    month_cal = cal.monthdatescalendar(trade_date.year, trade_date.month)
    busi_days = []
    for week in month_cal:
        for d in week:
            if d.weekday() < SATURDAY and d.month == trade_date.month:
                busi_days.append(d)

    return busi_days


## find the first trading day of the week 
def is_start_of_week(trade_date, holidays):
    firsts = []
    find_next_day = False
    for d in business_days(trade_date):
        if find_next_day == False:
            if d.weekday() == MONDAY:
                if d not in holidays:
                    firsts.append(d) 
                else:
                    find_next_day = True
        elif d.weekday() < SATURDAY and d not in holidays:
            firsts.append(d) 
            find_next_day = False

    return trade_date in firsts

def is_day_of_week(tgt_day, trade_date, holidays):
    firsts = []
    find_next_day = False
    for d in business_days(trade_date):
        if find_next_day == False:
            if d.weekday() == tgt_day:
                if d not in holidays:
                    firsts.append(d) 
                else:
                    find_next_day = True
        elif d.weekday() < SATURDAY and d not in holidays:
            firsts.append(d) 
            find_next_day = False

    return trade_date in firsts


## find previous trading day relative to the date given
def prev_trading_day(reference_date, holidays):
    one_day_back = timedelta(days=1)
    d = reference_date - one_day_back
    while True:
        if d.weekday() < SATURDAY and d.strftime("%Y-%m-%d") not in holidays:
            return date(d.year, d.month, d.day)
        else:
            d -= one_day_back


## find the last trading day of the week 
def is_end_of_week(trade_date, holidays):
    ends = [] 
    find_prev_day = False
    for d in business_days(trade_date)[::-1]:
        if find_prev_day == False:
            if d.weekday() == FRIDAY:
                if d not in holidays:
                    ends.append(d)
                else:
                    find_prev_day = True
        elif d.weekday() < SATURDAY and d not in holidays:
            ends.append(d)
            find_prev_day = False

    return trade_date in ends


def load_holidays():
    ## file foramt: "Date,Holiday"
    fn = "us_market_holidays.csv"
    df = pandas.read_csv(fn)
    holidays = []
    for d in df['Date'].to_list():
        holidays.append(datetime.strptime(d,"%Y-%m-%d").date())
    return holidays

def show_trading_calendar(year, month):
    holidays = load_holidays()
    cal_columns = f'Date Action Day Holiday'
    cal_table = PrettyTable(cal_columns.split())
    cal = calendar.Calendar()
    test_month = cal.monthdatescalendar(year, month)
    #print('holidays: ', holidays)
    for week in test_month:
        for d in week:
            if d.month == month:
                hh = "-----" if d not in holidays else "HOLIDAY" 
                action = "----"
                if check_buy(d,holidays):
                    action = "BUY"
                elif check_sell(d,holidays):
                    action = "SELL"
                #print(d, action, WEEKDAYS[d.weekday()], hh)
                cal_table.add_row([d, action, WEEKDAYS[d.weekday()], hh])
    
    print(" ")
    print(cal_table)
   

if __name__ == '__main__':
    g = cvt_date("2022-08-16")
    print(business_days(g))

    print(WEEKDAYS[weekday("2022-08-16")])

