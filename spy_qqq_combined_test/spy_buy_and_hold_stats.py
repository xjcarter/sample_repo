from security import Security
from buy_and_hold import BuyAndHold, DumpFormat 

spy_def = {
    "symbol": "SPY",
    "sec_type": "ETF",
    "tick_size": 0.01,
    "tick_value": 0.01
}

spy = Security(spy_def)

sec_stats = BuyAndHold(spy, start_dt=None)

sec_stats.run_and_report()
sec_stats.dump_trades(formats=[DumpFormat.HTML, DumpFormat.CSV])
sec_stats.dump_test_ledger(formats=[DumpFormat.HTML, DumpFormat.CSV])
sec_stats.dump_metrics([DumpFormat.JSON, DumpFormat.HTML], transpose=False, title=" ")
