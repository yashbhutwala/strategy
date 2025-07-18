# Bitcoin Treasury Metrics Calculator

This project provides tools to analyze bitcoin holdings and calculate key financial metrics for companies holding Bitcoin as treasury assets, using data from the Bitcoin Treasuries API.

## Overview

The main script `calculate_btc_per_share.py` fetches real-time data from the Bitcoin Treasuries API and calculates important metrics for companies holding Bitcoin. It supports multiple companies through entity IDs.

## Installation

```bash
pip install requests pandas matplotlib
```

## Supported Companies

| Company | Stock Ticker | Entity ID |
|---------|-------------|-----------|
| MicroStrategy | MSTR | 1 |
| Block | XYZ | 5 |
| Metaplanet | 3350.T | 176 |
| Semler Scientific | SMLR | 194 |

## Usage

### Basic Usage (defaults to MicroStrategy)
```bash
python calculate_btc_per_share.py
```

### Analyze Metaplanet
```bash
python calculate_btc_per_share.py --entity-id 176
```

### Analyze Multiple Companies
```bash
python calculate_btc_per_share.py --entity-id 1 176
```

### Specify Start Date
```bash
python calculate_btc_per_share.py --start-date 2023-01-01
```

### Show Plots
```bash
python calculate_btc_per_share.py --plot
```

### Export to CSV
```bash
python calculate_btc_per_share.py --csv
```

### Combined Example - Multiple Companies with Plots and Export
```bash
python calculate_btc_per_share.py --entity-id 1 176 --start-date 2024-01-01 --plot --csv
```

## Command Line Options

- `--entity-id`: Specify one or more companies to analyze (default: 1 for MicroStrategy). Can specify multiple IDs separated by spaces.
- `--start-date`: Start date for analysis in YYYY-MM-DD format (default: 2024-01-01)
- `--plot`: Display graphical plots of the metrics. When multiple entities are specified, they will be shown on the same plot for comparison.
- `--csv`: Export data to CSV files with timestamps. Creates individual files for each company and a combined file when multiple entities are specified.

## Key Metrics Explained

### BTC per Share
- **Definition**: The amount of bitcoin owned per share of MicroStrategy stock
- **Formula**: BTC per Share = Total BTC Holdings / Shares Outstanding
- **Example**: 0.00213833 BTC/share means each share represents ownership of approximately 0.00214 bitcoin
- **Importance**: Core metric for tracking MicroStrategy's bitcoin accumulation strategy

### mNAV (Market-to-Net Asset Value)
- **Definition**: The ratio of MicroStrategy's market capitalization to its Net Asset Value (NAV)
- **Formula**: mNAV = Market Cap / NAV
- **Interpretation**:
  - mNAV > 1: Stock trades at a premium to bitcoin holdings
  - mNAV = 1: Stock trades at par with bitcoin holdings
  - mNAV < 1: Stock trades at a discount to bitcoin holdings
- **Example**: mNAV of 1.656 = 65.6% premium to NAV

### Bitcoin Yield
- **Definition**: The percentage increase in BTC per share over a specific time period
- **Formula**: Bitcoin Yield = (BTC per share end - BTC per share start) / BTC per share start × 100%
- **Purpose**: Measures effectiveness of capital raising and bitcoin accumulation strategy
- **Example**: If BTC/share increases from 0.00200 to 0.00214, that's a 7% Bitcoin Yield

### BYD (Bitcoin Yield Days)
- **Definition**: The number of days required to achieve a certain Bitcoin Yield at the current accumulation rate
- **Purpose**: Normalizes Bitcoin Yield to a daily rate for easier comparison
- **Calculation**: Uses annualized Bitcoin Yield divided by 365 to determine daily yield rate

### P/BYD Ratio (Premium to Bitcoin Yield Days)
- **Definition**: Valuation metric relating MicroStrategy's stock premium to its Bitcoin Yield generation capability
- **Formula**: P/BYD = (mNAV - 1) / (Bitcoin Yield Rate)
- **Interpretation**: Indicates how many days/years of Bitcoin Yield the current premium represents
- **Use Case**: Helps investors assess whether the stock premium is justified by the company's BTC accumulation rate

## API Data Structure

The Bitcoin Treasuries API provides the following key data fields:
- `btcBalances`: Historical bitcoin holdings
- `btcPerShare`: Historical BTC per share values
- `navMultipliers`: Historical mNAV values
- `stockPrices`: Historical stock prices
- `btcPrices`: Historical bitcoin prices

## Example Output

```
Fetching MicroStrategy data...

Result:
BTC per Share on June 30, 2025: 0.00213833
Total BTC Holdings: 597,325
Implied Shares Outstanding: 279,341,823
mNAV (NAV Multiplier) on June 30, 2025: 1.6560
```

## Investment Considerations

These metrics are essential for understanding MicroStrategy as a "Bitcoin Development Company" that aims to:
1. Continuously increase BTC per share through strategic capital raises
2. Leverage various financial instruments (convertible bonds, ATM equity offerings)
3. Create value for shareholders beyond simply holding bitcoin

## Data Source

All data is sourced from: https://bitcointreasuries.net/api/web/entities/1/timeseries

## Disclaimer

This tool is for informational purposes only and should not be considered investment advice. Always conduct your own research and consult with financial professionals before making investment decisions.