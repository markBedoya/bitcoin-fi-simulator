from src.data_pipeline import load_coinmetrics
from src.price_model_v2 import get_cycle_anchor_df, fit_cycle_combo_centerlines


def main():
    prices, _ = load_coinmetrics(refresh=False)
    anchors = get_cycle_anchor_df(prices)
    fits, curves = fit_cycle_combo_centerlines(prices)

    # 3 individual complete cycles + 3 contiguous combined fits + 3 live-data fits
    assert len(fits) == 9, f"Expected 9 fits, got {len(fits)}"
    assert not curves.empty, "Expected non-empty curve output"
    assert anchors['price_usd'].notna().sum() >= 7, "Expected historical anchors with observed prices"
    assert (fits['days_used'] > 300).all(), "Each fit should use a substantial daily window"
    assert (fits['slope'] > 0).all(), "Power-law exponents should be positive"
    assert set(fits['fit_group']) == {'Complete cycle fits', 'Live-data fits'}

    print('PASS: Price Model v2 expanded cycle fit outputs look valid.')


if __name__ == '__main__':
    main()
