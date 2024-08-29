from trend_strategy import Trend_Strategy, DumpFormat
from security import Security
from portfolio import Portfolio

spy_def = {
    "symbol": "SPY",
    "sec_type": "ETF",
    "tick_size": 0.01,
    "tick_value": 0.01
}

qqq_def = {
    "symbol": "QQQ",
    "sec_type": "ETF",
    "tick_size": 0.01,
    "tick_value": 0.01
}

spy = Security(spy_def)
qqq = Security(qqq_def)

spy_config = {
    "name": "SPY_Layer`",
    "settings":
        {
            "use_stop": True
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

qqq_config = {
    "name": "QQQ_Layer",
    "settings":
        {
            "use_stop": True,
            "duration": 10
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


v1 = Trend_Strategy(spy, spy_config)
v2 = Trend_Strategy(qqq, qqq_config)

portfolio = Portfolio()
portfolio.set_capital(10000)
portfolio.add(v1)
portfolio.add(v2)

portfolio.run_and_report()
portfolio.dump_trades(formats=[DumpFormat.HTML, DumpFormat.CSV])
portfolio.dump_test_ledger(formats=[DumpFormat.HTML, DumpFormat.CSV])
portfolio.dump_metrics([DumpFormat.JSON, DumpFormat.HTML], transpose=False, title=" ")
