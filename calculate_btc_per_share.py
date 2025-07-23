#!/usr/bin/env python3
"""
Calculate quarterly BTC per share metrics for companies holding Bitcoin
Supports MicroStrategy (Entity ID: 1) and Metaplanet (Entity ID: 176)
"""

import requests
import json
from datetime import datetime, timezone, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import math
import argparse
import pandas as pd

def fetch_treasury_data(entity_id=1):
    """Fetch treasury data from API for specified entity"""
    url = f"https://bitcointreasuries.net/api/web/entities/{entity_id}/timeseries"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

def calculate_p_byd_ratio(mnav, btc_yield_annual):
    """
    Calculate P/BYD (Premium to Bitcoin Yield Days) ratio using logarithmic formula

    Args:
        mnav: Market-to-NAV ratio
        btc_yield_annual: Annualized Bitcoin Yield as a percentage

    Returns:
        tuple: (p_byd_ratio, premium_percentage, years_of_yield)
    """
    if btc_yield_annual <= 0 or mnav <= 0:
        return None, None, None

    # Calculate premium percentage for display
    premium_percentage = (mnav - 1) * 100

    # Convert annual yield percentage to decimal
    yield_decimal = btc_yield_annual / 100

    # Calculate P/BYD using logarithmic formula
    # P/BYD = ln(mNAV) / ln(1 + yield)
    # This gives the number of years needed at the given yield to reach the current mNAV
    try:
        p_byd_ratio = math.log(mnav) / math.log(1 + yield_decimal)
        years_of_yield = p_byd_ratio
    except (ValueError, ZeroDivisionError):
        return None, None, None

    return p_byd_ratio, premium_percentage, years_of_yield

def calculate_btc_yield(data, start_date, end_date):
    """Calculate BTC Yield between two dates"""
    # Get BTC per share for start and end dates
    start_timestamp = int(start_date.timestamp() * 1000)
    end_timestamp = int(end_date.timestamp() * 1000)

    btc_per_share_data = data.get('btcPerShare', [])


    start_btc_per_share = None
    end_btc_per_share = None

    # Find start date BTC per share
    for timestamp, per_share in btc_per_share_data:
        if timestamp <= start_timestamp:
            start_btc_per_share = per_share
        elif start_btc_per_share is not None:
            break

    # Find end date BTC per share
    for timestamp, per_share in btc_per_share_data:
        if timestamp <= end_timestamp:
            end_btc_per_share = per_share
        elif end_btc_per_share is not None:
            break

    if start_btc_per_share is None or end_btc_per_share is None:
        return None, None, None

    # Calculate yield
    btc_yield = ((end_btc_per_share - start_btc_per_share) / start_btc_per_share) * 100

    # Calculate annualized yield
    days_between = (end_date - start_date).days
    if days_between > 0:
        annualized_yield = (btc_yield / days_between) * 365
    else:
        annualized_yield = 0

    return btc_yield, annualized_yield, (start_btc_per_share, end_btc_per_share)

def get_quarter_dates(year, quarter):
    """Get start and end dates for a given quarter"""
    if quarter == 1:
        start = datetime(year, 1, 1, tzinfo=timezone.utc)
        end = datetime(year, 3, 31, tzinfo=timezone.utc)
    elif quarter == 2:
        start = datetime(year, 4, 1, tzinfo=timezone.utc)
        end = datetime(year, 6, 30, tzinfo=timezone.utc)
    elif quarter == 3:
        start = datetime(year, 7, 1, tzinfo=timezone.utc)
        end = datetime(year, 9, 30, tzinfo=timezone.utc)
    elif quarter == 4:
        start = datetime(year, 10, 1, tzinfo=timezone.utc)
        end = datetime(year, 12, 31, tzinfo=timezone.utc)
    return start, end

def get_month_dates(year, month):
    """Get start and end dates for a given month"""
    from calendar import monthrange
    start = datetime(year, month, 1, tzinfo=timezone.utc)
    last_day = monthrange(year, month)[1]
    end = datetime(year, month, last_day, tzinfo=timezone.utc)
    return start, end

def check_month_data_coverage(data, year, month, coverage_threshold=0.6):
    """
    Check if data is available for more than the specified threshold of days in a month

    Args:
        data: The entity data containing timestamps
        year: Year of the month to check
        month: Month number (1-12)
        coverage_threshold: Minimum fraction of days with data required (default: 0.6 for 60%)

    Returns:
        tuple: (has_sufficient_coverage, actual_coverage_ratio, days_with_data, total_days)
    """
    from calendar import monthrange

    # Get month boundaries
    start_date = datetime(year, month, 1, tzinfo=timezone.utc)
    total_days = monthrange(year, month)[1]
    end_date = datetime(year, month, total_days, tzinfo=timezone.utc)

    start_timestamp = int(start_date.timestamp() * 1000)
    end_timestamp = int(end_date.timestamp() * 1000)

    # Check available data points in the month
    # We'll check stock prices as they're recorded daily
    stock_prices = data.get('stockPrices', [])

    days_with_data = set()
    for timestamp, price in stock_prices:
        if start_timestamp <= timestamp <= end_timestamp:
            # Convert timestamp to date
            date = datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc)
            days_with_data.add(date.date())

    # Calculate coverage
    actual_coverage = len(days_with_data) / total_days
    has_sufficient_coverage = actual_coverage >= coverage_threshold

    return has_sufficient_coverage, actual_coverage, len(days_with_data), total_days

def find_stock_price_for_date(data, target_date):
    """Find stock price for a specific date"""
    target_timestamp = int(target_date.timestamp() * 1000)
    stock_prices = data.get('stockPrices', [])

    stock_price = None
    for timestamp, price in stock_prices:
        if timestamp == target_timestamp:
            stock_price = price
            break

    if stock_price is None:
        # Find closest before target date
        for timestamp, price in stock_prices:
            if timestamp <= target_timestamp:
                stock_price = price
            else:
                break

    return stock_price

def find_max_stock_price_for_quarter(data, start_date, end_date):
    """Find maximum stock price during a quarter"""
    start_timestamp = int(start_date.timestamp() * 1000)
    end_timestamp = int(end_date.timestamp() * 1000)
    stock_prices = data.get('stockPrices', [])

    max_price = 0
    for timestamp, price in stock_prices:
        if start_timestamp <= timestamp <= end_timestamp:
            if price > max_price:
                max_price = price

    return max_price if max_price > 0 else None

def find_btc_balance_for_date(data, target_date):
    """Find total BTC balance for a specific date"""
    target_timestamp = int(target_date.timestamp() * 1000)
    btc_balances = data.get('btcBalances', [])

    btc_balance = None
    for timestamp, balance in btc_balances:
        if timestamp == target_timestamp:
            btc_balance = balance
            break

    if btc_balance is None:
        # Find closest before target date
        for timestamp, balance in btc_balances:
            if timestamp <= target_timestamp:
                btc_balance = balance
            else:
                break

    return btc_balance

def find_btc_per_share_for_date(data, target_date):
    """Find BTC per share for a specific date"""
    target_timestamp = int(target_date.timestamp() * 1000)
    btc_per_share_data = data.get('btcPerShare', [])

    btc_per_share = None
    for timestamp, per_share in btc_per_share_data:
        if timestamp == target_timestamp:
            btc_per_share = per_share
            break

    if btc_per_share is None:
        # Find closest before target date
        for timestamp, per_share in btc_per_share_data:
            if timestamp <= target_timestamp:
                btc_per_share = per_share
            else:
                break

    return btc_per_share

def find_mnav_for_date(data, target_date):
    """Find mNAV (NAV multiplier) for a specific date"""
    # Convert target date to timestamp (milliseconds)
    target_timestamp = int(target_date.timestamp() * 1000)

    nav_multipliers = data.get('navMultipliers', [])

    # Find the mNAV for our target date
    mnav = None

    for timestamp, nav_mult in nav_multipliers:
        if timestamp == target_timestamp:
            mnav = nav_mult
            break

    if mnav is None:
        # Find closest before target date
        for timestamp, nav_mult in nav_multipliers:
            if timestamp <= target_timestamp:
                mnav = nav_mult
            else:
                break

    return mnav

def find_max_mnav_for_quarter(data, start_date, end_date):
    """Find maximum mNAV during a quarter"""
    start_timestamp = int(start_date.timestamp() * 1000)
    end_timestamp = int(end_date.timestamp() * 1000)
    nav_multipliers = data.get('navMultipliers', [])

    max_mnav = 0
    for timestamp, nav_mult in nav_multipliers:
        if start_timestamp <= timestamp <= end_timestamp:
            if nav_mult > max_mnav:
                max_mnav = nav_mult

    return max_mnav if max_mnav > 0 else None

def calculate_monthly_metrics(data, start_date, end_date):
    """
    Calculate monthly Bitcoin Yield and P/BYD ratios

    Returns:
        dict with month_ends, btc_yields, mnavs, p_byds, stock_prices, btc_balances, btc_per_shares
    """
    month_ends = []
    btc_yields = []
    annualized_yields = []
    mnavs = []
    mnavs_max = []
    p_byds = []
    month_labels = []
    stock_prices = []
    btc_balances = []
    btc_per_shares = []

    current_date = start_date
    today = datetime.now(timezone.utc)

    while current_date <= end_date and current_date <= today:
        year = current_date.year
        month = current_date.month

        # Get month boundaries
        m_start, m_end = get_month_dates(year, month)

        # Create month label
        month_name = m_start.strftime('%b')
        label = f"{month_name} {year}"

        # For current month, check if we have enough data coverage
        if m_end > today:
            # Check data coverage for partial month
            has_coverage, coverage_ratio, days_with_data, total_days = check_month_data_coverage(data, year, month)
            if not has_coverage:
                # Not enough data for this month yet
                break
            # If we have enough coverage, use today as the end date
            print(f"  Including partial month {label} with {days_with_data}/{total_days} days ({coverage_ratio:.1%} coverage)")
            m_end = today

        # Calculate BTC Yield for the month
        yield_result = calculate_btc_yield(data, m_start, m_end)
        if yield_result[0] is None:
            # Skip this month if no yield data available
            # Move to next month
            if month == 12:
                current_date = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
            else:
                current_date = datetime(year, month + 1, 1, tzinfo=timezone.utc)
            continue

        btc_yield, annual_yield, (start_bps, end_bps) = yield_result

        # Get both max and month-end mNAV
        mnav_max = find_max_mnav_for_quarter(data, m_start, m_end)
        mnav_end = find_mnav_for_date(data, m_end)

        # Get stock price at month end
        stock_price = find_stock_price_for_date(data, m_end)

        # Get BTC balance at month end
        btc_balance = find_btc_balance_for_date(data, m_end)

        # Calculate BTC per share
        btc_per_share = end_bps if end_bps is not None else None

        # Calculate P/BYD ratio if we have positive annual yield
        if annual_yield and annual_yield > 0 and mnav_end:
            p_byd, _, _ = calculate_p_byd_ratio(mnav_end, annual_yield)
        else:
            p_byd = None

        # Store results
        month_ends.append(m_end)
        btc_yields.append(btc_yield)
        annualized_yields.append(annual_yield)
        mnavs.append(mnav_end)
        mnavs_max.append(mnav_max)
        p_byds.append(p_byd)
        month_labels.append(label)
        stock_prices.append(stock_price)
        btc_balances.append(btc_balance)
        btc_per_shares.append(btc_per_share)

        # Move to next month
        if month == 12:
            current_date = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
        else:
            current_date = datetime(year, month + 1, 1, tzinfo=timezone.utc)

    return {
        'labels': month_labels,
        'month_ends': month_ends,
        'btc_yields': btc_yields,
        'annualized_yields': annualized_yields,
        'mnavs': mnavs,
        'mnavs_max': mnavs_max,
        'p_byds': p_byds,
        'stock_prices': stock_prices,
        'btc_balances': btc_balances,
        'btc_per_shares': btc_per_shares
    }

def calculate_quarterly_metrics(data, start_date, end_date):
    """
    Calculate quarterly Bitcoin Yield and P/BYD ratios

    Returns:
        dict with quarter_ends, btc_yields, mnavs, p_byds, stock_prices, btc_balances, btc_per_shares
    """
    quarter_ends = []
    btc_yields = []
    annualized_yields = []
    mnavs = []
    mnavs_max = []
    p_byds = []
    quarter_labels = []
    stock_prices = []
    btc_balances = []
    btc_per_shares = []

    current_date = start_date
    today = datetime.now(timezone.utc)

    while current_date <= end_date and current_date <= today:
        # Determine current quarter
        year = current_date.year
        month = current_date.month
        quarter = (month - 1) // 3 + 1

        # Get quarter boundaries
        q_start, q_end = get_quarter_dates(year, quarter)

        # Skip incomplete quarters
        if q_end > today:
            break

        label = f"Q{quarter} {year}"

        # Calculate BTC Yield for the quarter
        yield_result = calculate_btc_yield(data, q_start, q_end)
        if yield_result[0] is None:
            # Skip this quarter if no yield data available
            # Move to next quarter
            if quarter == 4:
                current_date = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
            else:
                current_date = datetime(year, quarter * 3 + 1, 1, tzinfo=timezone.utc)
            continue

        btc_yield, annual_yield, (start_bps, end_bps) = yield_result

        # Get both max and quarter-end mNAV
        mnav_max = find_max_mnav_for_quarter(data, q_start, q_end)
        mnav_end = find_mnav_for_date(data, q_end)

        # Get additional metrics
        stock_price = find_max_stock_price_for_quarter(data, q_start, q_end)  # Max price during quarter
        btc_balance = find_btc_balance_for_date(data, q_end)  # BTC at quarter end
        btc_per_share = find_btc_per_share_for_date(data, q_end)  # BTC/share at quarter end

        if btc_yield is not None and mnav_end is not None and annual_yield is not None:
            # Calculate P/BYD using quarter-end mNAV (only for positive yields)
            if annual_yield > 0:
                p_byd, _, _ = calculate_p_byd_ratio(mnav_end, annual_yield)
            else:
                p_byd = None  # P/BYD is not meaningful for negative yields

            quarter_ends.append(q_end)
            btc_yields.append(btc_yield)
            annualized_yields.append(annual_yield)
            mnavs.append(mnav_end)
            mnavs_max.append(mnav_max if mnav_max is not None else mnav_end)
            p_byds.append(p_byd if p_byd is not None else np.nan)
            quarter_labels.append(label)
            stock_prices.append(stock_price if stock_price is not None else 0)
            btc_balances.append(btc_balance if btc_balance is not None else 0)
            btc_per_shares.append(btc_per_share if btc_per_share is not None else 0)

        # Move to next quarter
        if quarter == 4:
            current_date = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
        else:
            current_date = datetime(year, quarter * 3 + 1, 1, tzinfo=timezone.utc)

    return {
        'quarter_ends': quarter_ends,
        'btc_yields': btc_yields,
        'annualized_yields': annualized_yields,
        'mnavs': mnavs,
        'mnavs_max': mnavs_max,
        'p_byds': p_byds,
        'labels': quarter_labels,
        'stock_prices': stock_prices,
        'btc_balances': btc_balances,
        'btc_per_shares': btc_per_shares
    }

def plot_quarterly_metrics(quarterly_data, title_prefix="MicroStrategy", data=None):
    """Backward compatibility wrapper for plot_metrics"""
    return plot_metrics(quarterly_data, title_prefix, is_monthly=False, data=data)

def plot_metrics(metrics_data, title_prefix="MicroStrategy", is_monthly=False, data=None):
    """
    Plot Bitcoin Yield, P/BYD ratios, and Stock Price (quarterly or monthly)
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    period_key = 'month_ends' if is_monthly else 'quarter_ends'
    period_label = 'Monthly' if is_monthly else 'Quarterly'

    dates = metrics_data.get(period_key, [])
    labels = metrics_data['labels']

    # Plot Bitcoin Yield
    ax1.plot(dates, metrics_data['btc_yields'], 'bo-', linewidth=2, markersize=8, label=f'{period_label} BTC Yield')
    ax1.set_ylabel('Bitcoin Yield (%)', fontsize=12)
    ax1.set_title(f'{title_prefix} - {period_label} Bitcoin Yield', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')

    # Add value labels on points
    for i, (date, yield_val, label) in enumerate(zip(dates, metrics_data['btc_yields'], labels)):
        ax1.annotate(f'{yield_val:.1f}%',
                    (date, yield_val),
                    textcoords="offset points",
                    xytext=(0,10),
                    ha='center',
                    fontsize=9)

    # Calculate and plot overall values if data is provided
    if data is not None and len(dates) > 0:
        # Get the date range for overall calculation
        if is_monthly:
            # For monthly, parse month name and year
            first_label_parts = labels[0].split()
            first_month_name = first_label_parts[0]
            first_year = int(first_label_parts[1])
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            first_month = month_names.index(first_month_name) + 1
            first_period_start, _ = get_month_dates(first_year, first_month)
        else:
            # For quarterly
            first_q_year = int(labels[0].split()[1])
            first_q_num = int(labels[0].split()[0][1])
            first_period_start, _ = get_quarter_dates(first_q_year, first_q_num)

        last_period_end = dates[-1]

        # Calculate overall BTC yield
        overall_result = calculate_btc_yield(data, first_period_start, last_period_end)
        if overall_result[0] is not None:
            overall_btc_yield, overall_annual_yield, _ = overall_result

            # Add horizontal line for overall BTC yield
            ax1.axhline(y=overall_btc_yield, color='black', linestyle='--', alpha=0.7,
                       label=f'Overall: {overall_btc_yield:.1f}%')
            ax1.legend(loc='best')

    # Plot P/BYD Ratio
    ax2.plot(dates, metrics_data['p_byds'], 'ro-', linewidth=2, markersize=8)
    ax2.set_ylabel('P/BYD Ratio', fontsize=12)
    ax2.set_title(f'{title_prefix} - {period_label} P/BYD Ratio', fontsize=14)
    ax2.grid(True, alpha=0.3)

    # Add value labels
    for i, (date, p_byd) in enumerate(zip(dates, metrics_data['p_byds'])):
        if p_byd is not None and not np.isnan(p_byd):
            ax2.annotate(f'{p_byd:.2f}',
                        (date, p_byd),
                        textcoords="offset points",
                        xytext=(0,10),
                        ha='center',
                        fontsize=9)

    # Add overall P/BYD line if we calculated overall values
    if data is not None and len(dates) > 0 and 'overall_result' in locals() and overall_result[0] is not None:
        # Calculate overall P/BYD
        last_mnav_at_end = find_mnav_for_date(data, last_period_end)
        if overall_annual_yield and overall_annual_yield > 0 and last_mnav_at_end:
            overall_p_byd, _, _ = calculate_p_byd_ratio(last_mnav_at_end, overall_annual_yield)
            if overall_p_byd is not None:
                ax2.axhline(y=overall_p_byd, color='black', linestyle='--', alpha=0.7,
                           label=f'Overall: {overall_p_byd:.2f}')
                ax2.legend(loc='best')

    # Plot Stock Price
    ax3.plot(dates, metrics_data['stock_prices'], 'go-', linewidth=2, markersize=8)
    ax3.set_ylabel('Stock Price ($)', fontsize=12)
    ax3.set_title(f'{title_prefix} - {period_label} Stock Price', fontsize=14)
    ax3.grid(True, alpha=0.3)

    # Add value labels on stock price points
    for i, (date, price, label) in enumerate(zip(dates, metrics_data['stock_prices'], labels)):
        if price is not None:
            ax3.annotate(f'${price:.0f}',
                        (date, price),
                        textcoords="offset points",
                        xytext=(0,10),
                        ha='center',
                        fontsize=9)

    # Format x-axis with period labels (now on ax3 since it's the bottom plot)
    ax3.set_xticks(dates)
    ax3.set_xticklabels(labels, rotation=45, ha='right')

    # Add mNAV values as text
    fig.text(0.02, 0.98, 'mNAV values:', transform=fig.transFigure, fontsize=10, verticalalignment='top')
    mnav_text = '\n'.join([f'{label}: {mnav:.3f}' for label, mnav in zip(labels, metrics_data['mnavs'])])
    fig.text(0.02, 0.95, mnav_text, transform=fig.transFigure, fontsize=9, verticalalignment='top')

    plt.tight_layout()
    plt.subplots_adjust(left=0.15)  # Make room for mNAV text
    plt.show()

def export_to_csv(all_metrics_data, start_date, end_date, is_monthly=False):
    """
    Export metrics data to CSV files with timestamp

    Args:
        all_metrics_data: List of tuples (metrics, company_name, data)
        start_date: Start date for the analysis
        end_date: End date for the analysis
        is_monthly: Whether the data is monthly or quarterly

    Returns:
        filenames: List of created CSV filenames
    """
    # Generate timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filenames = []

    period_key = 'month_ends' if is_monthly else 'quarter_ends'
    period_label = 'Month' if is_monthly else 'Quarter'
    period_end_label = 'M-end mNAV' if is_monthly else 'Q-end mNAV'

    for metrics, company_name, data in all_metrics_data:
        if len(metrics.get(period_key, [])) == 0:
            continue

        # Create DataFrame for this company
        df_data = {
            period_label: metrics['labels'],
            'Total BTC': metrics['btc_balances'],
            'Max mNAV': metrics['mnavs_max'],
            period_end_label: metrics['mnavs'],
            'BTC/Share': metrics['btc_per_shares'],
            'BTC Yield (%)': metrics['btc_yields'],
            'Ann. Yield (%)': metrics['annualized_yields'],
            'P/BYD': metrics['p_byds']
        }

        df = pd.DataFrame(df_data)

        # Calculate overall metrics
        if len(metrics[period_key]) > 0:
            if is_monthly:
                # For monthly, parse month name and year
                first_label_parts = metrics['labels'][0].split()
                first_month_name = first_label_parts[0]
                first_year = int(first_label_parts[1])
                # Convert month name to number
                month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                first_month = month_names.index(first_month_name) + 1
                first_period_start, _ = get_month_dates(first_year, first_month)
            else:
                # For quarterly
                first_q_year = int(metrics['labels'][0].split()[1])
                first_q_num = int(metrics['labels'][0].split()[0][1])
                first_period_start, _ = get_quarter_dates(first_q_year, first_q_num)

            last_period_end = metrics[period_key][-1]

            overall_result = calculate_btc_yield(data, first_period_start, last_period_end)
            if overall_result[0] is not None:
                overall_btc_yield, overall_annual_yield, _ = overall_result
                last_mnav_at_end = find_mnav_for_date(data, last_period_end)
                overall_p_byd, _, _ = calculate_p_byd_ratio(last_mnav_at_end, overall_annual_yield) if overall_annual_yield and overall_annual_yield > 0 and last_mnav_at_end else (None, None, None)

                # Add overall row
                overall_row = {
                    period_label: 'OVERALL',
                    'Total BTC': '',
                    'Max mNAV': '',
                    period_end_label: '',
                    'BTC/Share': '',
                    'BTC Yield (%)': overall_btc_yield,
                    'Ann. Yield (%)': overall_annual_yield,
                    'P/BYD': overall_p_byd if overall_p_byd is not None else ''
                }
                df = pd.concat([df, pd.DataFrame([overall_row])], ignore_index=True)

        # Clean up filename
        clean_name = company_name.replace(' ', '_').replace('(', '').replace(')', '').replace('.', '')
        filename = f"btc_metrics_{clean_name}_{timestamp}.csv"

        # Write to CSV
        df.to_csv(filename, index=False)
        filenames.append(filename)

    # If multiple entities, also create a combined file
    if len(all_metrics_data) > 1:
        combined_filename = f"btc_metrics_combined_{timestamp}.csv"
        with open(combined_filename, 'w') as f:
            for i, (metrics, company_name, data) in enumerate(all_metrics_data):
                if len(metrics.get(period_key, [])) == 0:
                    continue

                # Write company header
                if i > 0:
                    f.write('\n')  # Empty line between companies
                f.write(f"Company: {company_name}\n")

                # Create and write DataFrame
                df_data = {
                    period_label: metrics['labels'],
                    'Total BTC': metrics['btc_balances'],
                    'Max mNAV': metrics['mnavs_max'],
                    period_end_label: metrics['mnavs'],
                    'BTC/Share': metrics['btc_per_shares'],
                    'BTC Yield (%)': metrics['btc_yields'],
                    'Ann. Yield (%)': metrics['annualized_yields'],
                    'P/BYD': metrics['p_byds']
                }
                df = pd.DataFrame(df_data)

                # Add overall metrics
                if len(metrics[period_key]) > 0:
                    if is_monthly:
                        # For monthly, parse month name and year
                        first_label_parts = metrics['labels'][0].split()
                        first_month_name = first_label_parts[0]
                        first_year = int(first_label_parts[1])
                        # Convert month name to number
                        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                        first_month = month_names.index(first_month_name) + 1
                        first_period_start, _ = get_month_dates(first_year, first_month)
                    else:
                        # For quarterly
                        first_q_year = int(metrics['labels'][0].split()[1])
                        first_q_num = int(metrics['labels'][0].split()[0][1])
                        first_period_start, _ = get_quarter_dates(first_q_year, first_q_num)

                    last_period_end = metrics[period_key][-1]

                    overall_result = calculate_btc_yield(data, first_period_start, last_period_end)
                    if overall_result[0] is not None:
                        overall_btc_yield, overall_annual_yield, _ = overall_result
                        last_mnav_at_end = find_mnav_for_date(data, last_period_end)
                        overall_p_byd, _, _ = calculate_p_byd_ratio(last_mnav_at_end, overall_annual_yield) if overall_annual_yield and overall_annual_yield > 0 and last_mnav_at_end else (None, None, None)

                        overall_row = {
                            period_label: 'OVERALL',
                            'Total BTC': '',
                            'Max mNAV': '',
                            period_end_label: '',
                            'BTC/Share': '',
                            'BTC Yield (%)': overall_btc_yield,
                            'Ann. Yield (%)': overall_annual_yield,
                            'P/BYD': overall_p_byd if overall_p_byd is not None else ''
                        }
                        df = pd.concat([df, pd.DataFrame([overall_row])], ignore_index=True)

                # Write to file
                df.to_csv(f, index=False)

        filenames.append(combined_filename)

    return filenames

def plot_multiple_entities(all_metrics, is_monthly=False, all_data=None):
    """
    Plot metrics for multiple entities on the same charts (quarterly or monthly)

    Args:
        all_metrics: List of tuples (metrics_data, company_name)
        is_monthly: Whether the data is monthly or quarterly
        all_data: List of data DataFrames corresponding to each entity in all_metrics
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    # Define colors for different entities
    colors = ['b', 'r', 'g', 'orange', 'purple', 'brown', 'pink', 'gray']
    markers = ['o', 's', '^', 'D', 'v', '*', 'p', 'h']

    period_key = 'month_ends' if is_monthly else 'quarter_ends'
    period_label = 'Monthly' if is_monthly else 'Quarterly'

    # Plot each entity
    for idx, (metrics_data, company_name) in enumerate(all_metrics):
        if len(metrics_data.get(period_key, [])) == 0:
            continue

        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]

        dates = metrics_data[period_key]
        labels = metrics_data['labels']

        # Plot Bitcoin Yield
        ax1.plot(dates, metrics_data['btc_yields'],
                color=color, marker=marker, linestyle='-',
                linewidth=2, markersize=8, label=f'{company_name} - BTC Yield')

        # Add value labels on points (only for first and last few to avoid clutter)
        if len(dates) <= 6:  # Show all labels if 6 or fewer points
            for i, (date, yield_val) in enumerate(zip(dates, metrics_data['btc_yields'])):
                ax1.annotate(f'{yield_val:.1f}%',
                            (date, yield_val),
                            textcoords="offset points",
                            xytext=(0,10),
                            ha='center',
                            fontsize=8,
                            color=color)
        else:  # Show only first and last labels if more than 6 points
            for i in [0, len(dates)-1]:
                ax1.annotate(f'{metrics_data["btc_yields"][i]:.1f}%',
                            (dates[i], metrics_data['btc_yields'][i]),
                            textcoords="offset points",
                            xytext=(0,10),
                            ha='center',
                            fontsize=8,
                            color=color)

        # Plot P/BYD Ratio
        ax2.plot(dates, metrics_data['p_byds'],
                color=color, marker=marker, linestyle='-',
                linewidth=2, markersize=8, label=f'{company_name} - P/BYD')

        # Add value labels (selective to avoid clutter)
        if len(dates) <= 6:
            for i, (date, p_byd) in enumerate(zip(dates, metrics_data['p_byds'])):
                if p_byd is not None and not np.isnan(p_byd):
                    ax2.annotate(f'{p_byd:.2f}',
                                (date, p_byd),
                                textcoords="offset points",
                                xytext=(0,10),
                                ha='center',
                                fontsize=8,
                                color=color)
        else:
            for i in [0, len(dates)-1]:
                if metrics_data['p_byds'][i] is not None and not np.isnan(metrics_data['p_byds'][i]):
                    ax2.annotate(f'{metrics_data["p_byds"][i]:.2f}',
                                (dates[i], metrics_data['p_byds'][i]),
                                textcoords="offset points",
                                xytext=(0,10),
                                ha='center',
                                fontsize=8,
                                color=color)

        # Plot Stock Price
        ax3.plot(dates, metrics_data['stock_prices'],
                color=color, marker=marker, linestyle='-',
                linewidth=2, markersize=8, label=f'{company_name} - Stock Price')

        # Add value labels (selective to avoid clutter)
        if len(dates) <= 6:
            for i, (date, price) in enumerate(zip(dates, metrics_data['stock_prices'])):
                if price is not None:
                    ax3.annotate(f'${price:.0f}',
                                (date, price),
                                textcoords="offset points",
                                xytext=(0,10),
                                ha='center',
                                fontsize=8,
                                color=color)
        else:
            for i in [0, len(dates)-1]:
                if metrics_data['stock_prices'][i] is not None:
                    ax3.annotate(f'${metrics_data["stock_prices"][i]:.0f}',
                                (dates[i], metrics_data['stock_prices'][i]),
                                textcoords="offset points",
                                xytext=(0,10),
                                ha='center',
                                fontsize=8,
                                color=color)

    # Add overall lines for each entity if data is provided
    if all_data is not None and len(all_data) == len(all_metrics):
        overall_line_colors = ['black', 'darkred', 'darkgreen', 'darkorange', 'darkviolet', 'darkgoldenrod', 'deeppink', 'darkgray']

        for idx, ((metrics_data, company_name), data) in enumerate(zip(all_metrics, all_data)):
            if len(metrics_data.get(period_key, [])) == 0:
                continue

            dates = metrics_data[period_key]
            labels = metrics_data['labels']

            if len(dates) > 0:
                # Parse the first period start date
                if is_monthly:
                    # For monthly, parse month name and year
                    first_label_parts = labels[0].split()
                    first_month_name = first_label_parts[0]
                    first_year = int(first_label_parts[1])
                    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                    first_month = month_names.index(first_month_name) + 1
                    first_period_start, _ = get_month_dates(first_year, first_month)
                else:
                    # For quarterly
                    first_q_year = int(labels[0].split()[1])
                    first_q_num = int(labels[0].split()[0][1])
                    first_period_start, _ = get_quarter_dates(first_q_year, first_q_num)

                last_period_end = dates[-1]

                # Calculate overall BTC yield
                overall_result = calculate_btc_yield(data, first_period_start, last_period_end)
                if overall_result[0] is not None:
                    overall_btc_yield, overall_annual_yield, _ = overall_result
                    line_color = overall_line_colors[idx % len(overall_line_colors)]

                    # Add horizontal line for overall BTC yield
                    ax1.axhline(y=overall_btc_yield, color=line_color, linestyle='--', alpha=0.7,
                               label=f'{company_name} Overall: {overall_btc_yield:.1f}%')

                    # Calculate overall P/BYD
                    last_mnav_at_end = find_mnav_for_date(data, last_period_end)
                    if overall_annual_yield and overall_annual_yield > 0 and last_mnav_at_end:
                        overall_p_byd, _, _ = calculate_p_byd_ratio(last_mnav_at_end, overall_annual_yield)
                        if overall_p_byd is not None:
                            ax2.axhline(y=overall_p_byd, color=line_color, linestyle='--', alpha=0.7,
                                       label=f'{company_name} Overall: {overall_p_byd:.2f}')

    # Configure Bitcoin Yield plot
    ax1.set_ylabel('Bitcoin Yield (%)', fontsize=12)
    ax1.set_title(f'{period_label} Bitcoin Yield Comparison', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', fontsize=10)

    # Configure P/BYD plot
    ax2.set_ylabel('P/BYD Ratio', fontsize=12)
    ax2.set_title(f'{period_label} P/BYD Ratio Comparison', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best', fontsize=10)

    # Configure Stock Price plot
    ax3.set_ylabel('Stock Price ($)', fontsize=12)
    ax3.set_title(f'{period_label} Stock Price Comparison', fontsize=14)
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='best', fontsize=10)

    # Format x-axis with period labels (now on ax3 since it's the bottom plot)
    all_dates = []
    all_labels = []
    for metrics_data, _ in all_metrics:
        if len(metrics_data.get(period_key, [])) > 0:
            all_dates.extend(metrics_data[period_key])
            all_labels.extend(metrics_data['labels'])

    # Get unique dates and labels (in case of overlaps)
    unique_data = sorted(set(zip(all_dates, all_labels)))
    if unique_data:
        dates, labels = zip(*unique_data)
        ax3.set_xticks(dates)
        ax3.set_xticklabels(labels, rotation=45, ha='right')

    plt.tight_layout()
    plt.show()

def process_entity(entity_id, start_date, end_date, entity_names, is_monthly=False):
    """Process a single entity and return its metrics"""
    company_name = entity_names.get(entity_id, f"Entity {entity_id}")

    print(f"\nFetching {company_name} data...")
    try:
        data = fetch_treasury_data(entity_id)
    except Exception as e:
        print(f"Error: Failed to fetch data for entity ID {entity_id}")
        print(f"Details: {str(e)}")
        return None, None, None

    period_type = "monthly" if is_monthly else "quarterly"
    print(f"Calculating {period_type} metrics from {start_date.strftime('%B %Y')} to {end_date.strftime('%B %Y')}...")

    if is_monthly:
        metrics = calculate_monthly_metrics(data, start_date, end_date)
    else:
        metrics = calculate_quarterly_metrics(data, start_date, end_date)

    return metrics, company_name, data

def print_entity_table(metrics, company_name, data, is_monthly=False):
    """Print the metrics table for a single entity"""
    period_key = 'month_ends' if is_monthly else 'quarter_ends'
    period_label = 'Month' if is_monthly else 'Quarter'

    if len(metrics.get(period_key, [])) > 0:
        print(f"\n{company_name} - Found data for {len(metrics[period_key])} {period_label.lower()}s:")

        # Print summary table
        print("\n" + "="*140)
        period_end_label = 'M-end mNAV' if is_monthly else 'Q-end mNAV'
        print(f"{period_label:<12} {'Total BTC':<12} {'Max mNAV':<10} {period_end_label:<11} {'BTC/Share':<12} {'BTC Yield':<10} {'Ann. Yield':<11} {'P/BYD':<8}")
        print("="*140)

        for i in range(len(metrics['labels'])):
            label = metrics['labels'][i]
            btc_yield = metrics['btc_yields'][i]
            annual_yield = metrics['annualized_yields'][i]
            mnav_end = metrics['mnavs'][i]
            mnav_max = metrics['mnavs_max'][i]
            p_byd = metrics['p_byds'][i]
            stock_price = metrics['stock_prices'][i]
            btc_balance = metrics['btc_balances'][i]
            btc_per_share = metrics['btc_per_shares'][i]

            p_byd_str = f"{p_byd:>7.2f}" if p_byd is not None and not np.isnan(p_byd) else "    N/A"
            print(f"{label:<12} {btc_balance:>11,.0f} {mnav_max:>9.3f} {mnav_end:>10.3f} {btc_per_share:>11.6f} {btc_yield:>9.2f}% {annual_yield:>10.1f}% {p_byd_str}")

        print("="*140)

        # Calculate aggregate metrics for the entire period
        if len(metrics[period_key]) > 0:
            # Get overall BTC yield from first period start to last period end
            if is_monthly:
                # For monthly, parse month name and year
                first_label_parts = metrics['labels'][0].split()
                first_month_name = first_label_parts[0]
                first_year = int(first_label_parts[1])
                # Convert month name to number
                month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                first_month = month_names.index(first_month_name) + 1
                first_period_start, _ = get_month_dates(first_year, first_month)
            else:
                # For quarterly
                first_q_year = int(metrics['labels'][0].split()[1])
                first_q_num = int(metrics['labels'][0].split()[0][1])
                first_period_start, _ = get_quarter_dates(first_q_year, first_q_num)

            last_period_end = metrics[period_key][-1]

            # Calculate overall BTC yield
            overall_result = calculate_btc_yield(data, first_period_start, last_period_end)
            if overall_result[0] is not None:
                overall_btc_yield, overall_annual_yield, (start_bps, end_bps) = overall_result

                # Get mNAV at the end of the period for P/BYD calculation
                last_mnav_at_end = find_mnav_for_date(data, last_period_end)
                overall_p_byd, _, _ = calculate_p_byd_ratio(last_mnav_at_end, overall_annual_yield) if overall_annual_yield and overall_annual_yield > 0 and last_mnav_at_end else (None, None, None)

                # Print aggregate row
                if overall_p_byd is not None:
                    print(f"{'OVERALL':<12} {'':>11} {'':>9} {'':>10} {'':>11} {overall_btc_yield:>9.2f}% {overall_annual_yield:>10.1f}% {overall_p_byd:>7.2f}")
                else:
                    print(f"{'OVERALL':<12} {'':>11} {'':>9} {'':>10} {'':>11} {overall_btc_yield:>9.2f}% {overall_annual_yield:>10.1f}% {'N/A':>7}")
            print("="*140)
    else:
        period_name = 'monthly' if is_monthly else 'quarterly'
        print(f"\n{company_name} - No {period_name} data available for the specified period.")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Calculate quarterly BTC metrics for companies holding Bitcoin')
    parser.add_argument('--plot', action='store_true', help='Show plots of quarterly metrics')
    parser.add_argument('--start-date', type=str, help='Start date for calculations (YYYY-MM-DD format, default: 2024-01-01)')
    parser.add_argument('--entity-id', type=int, nargs='+', default=[1], help='Entity ID(s) from Bitcoin Treasuries API (default: 1 for MicroStrategy, 176 for Metaplanet). Can specify multiple IDs.')
    parser.add_argument('--csv', action='store_true', help='Export data to CSV file with timestamp')
    parser.add_argument('--monthly', action='store_true', help='Calculate metrics monthly instead of quarterly')
    args = parser.parse_args()

    # Map entity IDs to company names for display
    entity_names = {
        1: "MicroStrategy (MSTR)",
        5: "Block (XYZ)",
        176: "Metaplanet (3350.T)",
        194: "Semler Scientific (SMLR)",
        295: "The Blockchain Group (ALTBG)",
        440: "The Smarter Web Company (SWC)",
        644: "Sequans Communications (SQNS)",
    }

    # Parse start date from command line or use default
    if args.start_date:
        try:
            start_date = datetime.strptime(args.start_date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        except ValueError:
            print(f"Error: Invalid date format '{args.start_date}'. Please use YYYY-MM-DD format.")
            return
    else:
        start_date = datetime(2024, 1, 1, tzinfo=timezone.utc)

    end_date = datetime.now(timezone.utc)

    # Process each entity
    all_metrics = []
    all_metrics_with_data = []  # For CSV export
    for entity_id in args.entity_id:
        metrics, company_name, data = process_entity(entity_id, start_date, end_date, entity_names, args.monthly)

        if metrics is None:
            continue

        # Store data and company name for plotting
        all_metrics.append((metrics, company_name))
        all_metrics_with_data.append((metrics, company_name, data))

        # Print table for this entity
        print_entity_table(metrics, company_name, data, args.monthly)

    # Export to CSV if --csv flag is provided
    if args.csv and len(all_metrics_with_data) > 0:
        print(f"\nExporting data to CSV...")
        csv_filenames = export_to_csv(all_metrics_with_data, start_date, end_date, args.monthly)
        print(f"Data exported to:")
        for filename in csv_filenames:
            print(f"  - {filename}")

    # Plot all entities together if --plot flag is provided
    if args.plot and len(all_metrics) > 0:
        period_type = "monthly" if args.monthly else "quarterly"
        print(f"\nGenerating {period_type} plots...")
        if len(all_metrics) == 1:
            # Single entity - use original plot function with data
            plot_metrics(all_metrics[0][0], title_prefix=all_metrics[0][1], is_monthly=args.monthly, data=all_metrics_with_data[0][2])
        else:
            # Multiple entities - create combined plot with data
            all_data = [item[2] for item in all_metrics_with_data]
            plot_multiple_entities(all_metrics, is_monthly=args.monthly, all_data=all_data)

if __name__ == "__main__":
    main()