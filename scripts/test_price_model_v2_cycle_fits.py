import numpy as np
import pandas as pd

from src.price_model_v2 import (
    get_cycle_anchor_df,
    fit_cycle_combo_centerlines,
    build_common_date_comparison,
    fit_progress_weighted_backbone,
    build_model_diagnostics,
)


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
    comparison, spread = build_common_date_comparison(fits, prices)
    weighted, weighted_curve = fit_progress_weighted_backbone(prices)
    expanding, deviations, geometry, floor, maturity = build_model_diagnostics(prices, fits, weighted)

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
    assert len(comparison) == 10, f"Expected Actual BTC + 9 fit rows, got {len(comparison)}"
    assert comparison.iloc[0]['fit_group'] == 'Actual BTC'
    checkpoint_cols = [c for c in comparison.columns if ' | ' in c]
    assert len(checkpoint_cols) == 7, checkpoint_cols
    assert len(spread) == 7, f"Expected 7 checkpoint summary rows, got {len(spread)}"
    assert (spread['centerline_min_usd'] > 0).all()
    assert (spread['centerline_max_usd'] >= spread['centerline_median_usd']).all()
    assert (spread['centerline_median_usd'] >= spread['centerline_min_usd']).all()
    assert 0.0 < weighted['progress'] <= 1.0
    assert weighted['evidence_weight'] == weighted['progress']
    assert weighted['live_centerline_usd'] > 0
    assert len(weighted_curve) == len(prices)
    assert list(expanding['fit_id']) == ['cycles_0_0', 'cycles_0_1', 'cycles_0_2', 'live_2011']
    assert (expanding['centerline_at_live_usd'] > 0).all()
    assert len(deviations) == 9
    assert (deviations['actual_to_centerline'] > 0).all()
    assert 'weighted_centerline_usd' in deviations.columns
    assert deviations.iloc[-1]['type'] == 'forming_trough'
    assert deviations.iloc[-1]['anchor_status'] == 'partial'
    assert deviations.loc[deviations['label'] == '2025 peak', 'anchor_status'].iloc[0] == 'confirmed'
    assert len(geometry) == 4
    assert geometry.iloc[-1]['trough_status'] == 'partial'
    assert 'peak_compression_vs_prior' in geometry.columns
    assert len(floor) == 1
    assert floor.iloc[0]['remaining_expected_days'] >= 0
    assert floor.iloc[0]['mature_completed_trough_median'] > 0
    assert floor.iloc[0]['forming_trough_date'] <= floor.iloc[0]['latest_date']
    assert len(maturity) == 1
    assert maturity.iloc[0]['completed_trough_min'] <= maturity.iloc[0]['completed_trough_max']
    assert maturity.iloc[0]['pattern_read'] == 'stable mature trough band; two-stage upside compression'

    print('PASS: Price Model v2 expanded cycle fit outputs look valid.')


if __name__ == '__main__':
    main()
