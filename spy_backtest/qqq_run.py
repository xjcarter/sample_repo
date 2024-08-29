from trend_strategy import Trend_Strategy, DumpFormat
from security import Security

qqq_def = {
    "symbol": "QQQ",
    "sec_type": "ETF",
    "tick_size": 0.01,
    "tick_value": 0.01
}

qqq = Security(qqq_def)

trend_config = {
    "settings":
        {
            "StDev": 50,
            "short_len": 10,
            "long_len": 40
        },
    "wallet":
        {
            "cash": 10000,
            "wallet_alloc_pct": 1,
            "borrow_margin_pct": 1 
        },
    "trading_limits":
        {
            "dollar_limit": 100000000,
            "position_limit": 100000,
            "leverage_target": 1
        }
}

trend = Trend_Strategy(qqq, trend_config)
trend.run_and_report()
#trend.dump_trades(formats=[DumpFormat.STDOUT])
