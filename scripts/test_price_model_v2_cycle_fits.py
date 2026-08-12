import numpy as np
import pandas as pd

from src.price_model_v2 import get_cycle_anchor_df, fit_cycle_combo_centerlines


def synthetic_prices():
    dates = pd.date_range('2011-01-01', '2026-08-01', freq='D')
    days = (dates - pd.Timestamp('2009-01-03')).days.to_numpy(dtype=float)
    # Positive power-law-like synthetic series with smooth cycle variation.
    base = 1.2e-16 * np.power(days, 5.75)
    phase = np.sin(np.arange(len(dates)) * 2.0 * np.pi / 1428.0)
    price = base * np.exp(0.7 * phase)
    return pd.DataFrame({'date': dates, 'price_usd': price})


def main():
    prices = synthetic_prices()
    anchors = get_cycle_anchor_df(prices)
    fits, curves = fit_cycle_combo_centerlines(prices)

    assert len(fits) == 9, f"Expected 9 fits, got {len(fits)}"
    assert not curves.empty, "Expected non-empty curve output"
    assert anchors['price_usd'].notna().sum() == len(anchors), "Expected every anchor to resolve"
    assert (fits['days_used'] > 300).all(), "Each fit should use a substantial daily window"
    assert (fits['slope'] > 0).all(), "Power-law exponents should be positive"
    assert set(fits['fit_group']) == {'Complete cycle fits', 'Live-data fits'}

    expected_ids = {
        'cycles_0_0', 'cycles_0_1', 'cycles_0_2',
        'cycles_1_1', 'cycles_1_2', 'cycles_2_2',
        'live_2011', 'live_2015', 'live_2018',
    }
    assert set(fits['fit_id']) == expected_ids, set(fits['fit_id'])

    print('PASS: Price Model v2 expanded cycle fit outputs look valid.')


if __name__ == '__main__':
    main()
