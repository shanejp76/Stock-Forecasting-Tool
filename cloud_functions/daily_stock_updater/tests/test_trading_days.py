from main import TradingDayValidator
from datetime import datetime, timedelta

validator = TradingDayValidator()

# Test today (Sunday)
today = datetime.now()
print(
    f'Today ({today.strftime("%A, %Y-%m-%d")}): Trading day = {validator.is_trading_day(today)}'
)

# Test last few days
for i in range(1, 8):
    test_date = today - timedelta(days=i)
    is_trading = validator.is_trading_day(test_date)
    print(f'{test_date.strftime("%A, %Y-%m-%d")}: Trading day = {is_trading}')

# Test last trading day function
last_trading = validator.get_last_trading_day()
print(f'\nLast trading day: {last_trading.strftime("%A, %Y-%m-%d")}')
