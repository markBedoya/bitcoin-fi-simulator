from pathlib import Path

page = Path('app_pages/2_Price_Model.py').read_text(encoding='utf-8')
assert 'Price Model v3.5 — Fair-Value Cycle Valuations' in page
assert 'Latest Bitcoin price data' in page
assert 'end_anchor_options = anchor_options + [latest_end_label]' in page
assert 'end_anchor_lookup[end_anchor_choice]' in page
assert 'cycle_valuation_history' in page
assert 'peak_valuation_model' in page
assert 'trough_valuation_model' in page
print('Price Model fair-value UI contract checks passed.')
