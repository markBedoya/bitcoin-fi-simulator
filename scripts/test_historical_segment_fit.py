import numpy as np
import pandas as pd
from src.price_model import _fit_historical_segment_path

# Delayed bull: very little movement until late in the segment.
dates = pd.date_range('2015-01-01','2017-12-31',freq='D')
p = np.linspace(0,1,len(dates))
move = p**3.2
start_price, end_price = 200.0, 20000.0
prices = np.exp(np.log(start_price) + (np.log(end_price)-np.log(start_price))*move)
train = pd.DataFrame({'date':dates,'price_usd':prices})
knots = pd.DataFrame([
    {'date':dates[0],'knot_price_usd':start_price},
    {'date':dates[-1],'knot_price_usd':end_price},
])
fit = _fit_historical_segment_path(train, knots)
mid = len(fit)//2
norm_mid = (np.log(fit[mid])-np.log(start_price))/(np.log(end_price)-np.log(start_price))
assert norm_mid < 0.25, norm_mid
assert abs(fit[0]/start_price-1) < 1e-12
assert abs(fit[-1]/end_price-1) < 1e-12
print('Historical segment fit preserves delayed acceleration and exact anchors.')
