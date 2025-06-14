import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
from datetime import datetime
import time

###############################################################################
# Utility functions
###############################################################################

def _coerce_numeric(df: pd.DataFrame, cols: list[str]):
    """Convert given columns to numeric, ignoring errors."""
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def load_trades(csv_file: str | bytes) -> pd.DataFrame:
    """Load the CSV file exported from your broker and clean it a little."""
    df = pd.read_csv(csv_file)

    # Standardise column names: strip whitespace and lower‑case without spaces
    df.columns = df.columns.str.strip()

    # Numeric conversions
    numeric_cols = [
        "No. of shares",
        "Price / share",
        "Exchange rate",
        "Result",
        "Total",
        "Withholding tax",
        "Currency conversion fee",
    ]
    _coerce_numeric(df, numeric_cols)

    # In this new file, the date and time are in one column named "Time"
    if "Time" in df.columns:
        df["Date"] = pd.to_datetime(df["Time"], errors="coerce")
    
    # Drop rows where date parsing failed
    df.dropna(subset=["Date"], inplace=True)

    return df


def realised_pnl(df: pd.DataFrame) -> float:
    """Sum of realised P&L from sell transactions."""
    return df.loc[df["Action"].str.contains("sell", case=False, na=False), "Result"].sum()


def total_fees(df: pd.DataFrame) -> float:
    return df["Currency conversion fee"].sum(skipna=True)


def net_deposits(df: pd.DataFrame) -> float:
    deposits = df.loc[df["Action"].str.contains("deposit", case=False, na=False), "Total"].sum()
    withdrawals = df.loc[df["Action"].str.contains("withdraw", case=False, na=False), "Total"].sum()
    return deposits - withdrawals


def current_positions(df: pd.DataFrame) -> pd.DataFrame:
    """Return a dataframe of current positions based on cumulative share count."""
    # Positive for buys, negative for sells
    df = df.copy()
    df["signed_shares"] = np.where(
        df["Action"].str.contains("sell", case=False, na=False),
        -df["No. of shares"],
        df["No. of shares"],
    )
    positions = (
        df.groupby("Ticker")["signed_shares"].sum().to_frame(name="NetShares")
    )
    positions = positions[positions["NetShares"] != 0]
    return positions


def fetch_market_prices(tickers: list[str]) -> dict[str, float]:
    """Fetch latest market price using yfinance. Returns dict ticker->price."""
    if not tickers:
        return {}

    prices = {}
    try:
        # Batch download data for all tickers to avoid rate limiting
        data = yf.download(tickers, period="5d", progress=False, auto_adjust=True)
        
        if data.empty:
            raise ValueError("No data returned from yfinance")

        # Get the last 'Close' price, handling both single and multiple tickers
        if isinstance(data.columns, pd.MultiIndex):
            last_prices = data['Close'].iloc[-1]
            prices = last_prices.to_dict()
        else:
            last_price = data['Close'].iloc[-1]
            prices[tickers[0]] = last_price

    except Exception as e:
        print(f"Batch yf.download failed: {e}. Falling back to individual requests.")
        # Fallback with delay if batch request fails
        for t in tickers:
            try:
                ticker_data = yf.Ticker(t)
                price = ticker_data.info.get("currentPrice") or ticker_data.info.get("regularMarketPrice")
                prices[t] = price if price is not None else np.nan
                time.sleep(0.2)  # 200ms delay
            except Exception:
                prices[t] = np.nan

    # Ensure all tickers have an entry in the final dict
    for t in tickers:
        if t not in prices or pd.isna(prices.get(t)):
            prices[t] = np.nan

    return prices


def compute_unrealised_pnl(df: pd.DataFrame, positions: pd.DataFrame) -> pd.DataFrame:
    if positions.empty:
        return positions.assign(MarketPrice=np.nan, CostBasis=np.nan, Unrealised=np.nan)

    # Compute average cost per share for each ticker based on buys
    buys = df[df["Action"].str.contains("buy", case=False, na=False)].copy()
    cost_basis = (
        (buys["Price / share"] * buys["No. of shares"]).groupby(buys["Ticker"]).sum()
        / buys.groupby("Ticker")["No. of shares"].sum()
    )
    positions = positions.join(cost_basis.rename("CostBasisPerShare"), how="left")

    # Get live prices
    prices = fetch_market_prices(positions.index.tolist())
    positions["MarketPrice"] = positions.index.map(prices)

    positions["Unrealised"] = (
        (positions["MarketPrice"] - positions["CostBasisPerShare"]) * positions["NetShares"]
    )
    return positions


def equity_curve(df: pd.DataFrame) -> pd.Series:
    df_sorted = df.sort_values("Date").copy()
    df_sorted["net_cash_flow"] = np.where(
        df_sorted["Action"].str.contains("buy", case=False, na=False),
        -df_sorted["Total"],
        np.where(
            df_sorted["Action"].str.contains("sell", case=False, na=False),
            df_sorted["Total"],
            np.where(
                df_sorted["Action"].str.contains("deposit", case=False, na=False),
                df_sorted["Total"],
                0.0,
            ),
        ),
    )
    equity = df_sorted["net_cash_flow"].cumsum()
    equity.index = df_sorted["Date"]
    return equity


def realised_pnl_by_day(df: pd.DataFrame) -> pd.Series:
    """Calculate realised PnL for each day."""
    sells = df.loc[df["Action"].str.contains("sell", case=False, na=False)].copy()
    if sells.empty:
        return pd.Series(dtype=float)
        
    sells["Day"] = sells["Date"].dt.date
    daily_pnl = sells.groupby("Day")["Result"].sum()
    return daily_pnl


def fees_by_day(df: pd.DataFrame) -> pd.Series:
    """Calculate total fees for each day."""
    df_copy = df.copy()
    if df_copy.empty or "Currency conversion fee" not in df_copy.columns:
        return pd.Series(dtype=float)

    df_copy["Day"] = df_copy["Date"].dt.date
    daily_fees = df_copy.groupby("Day")["Currency conversion fee"].sum()
    return daily_fees


def daily_net_realised_pnl(df: pd.DataFrame) -> pd.Series:
    """Calculate daily realised PnL, net of fees."""
    daily_pnl = realised_pnl_by_day(df)
    daily_fees = fees_by_day(df)

    # Combine into a DataFrame to align by date, fill NaNs with 0
    combined = pd.DataFrame({"pnl": daily_pnl, "fees": daily_fees}).fillna(0)

    # Net PnL is pnl minus fees
    net_daily_pnl = combined["pnl"] - combined["fees"]
    return net_daily_pnl


def trades_per_day(df: pd.DataFrame) -> pd.Series:
    """Calculate the total number of trades for each day."""
    trades_only = df[~df["Action"].str.contains("deposit|withdraw", case=False, na=False)].copy()
    if trades_only.empty:
        return pd.Series(dtype=int)

    trades_only["Day"] = trades_only["Date"].dt.date
    daily_trades = trades_only.groupby("Day").size()
    daily_trades.name = "NumberOfTrades"
    return daily_trades


def average_trades_by_hour(df: pd.DataFrame) -> pd.Series:
    """Calculate the average number of trades for each hour of the day."""
    trades_only = df[~df["Action"].str.contains("deposit|withdraw", case=False, na=False)].copy()
    if trades_only.empty:
        return pd.Series(dtype=float)

    num_days = trades_only["Date"].dt.date.nunique()
    if num_days == 0:
        return pd.Series(dtype=float)

    trades_only["Hour"] = trades_only["Date"].dt.hour
    hourly_trades = trades_only.groupby("Hour").size()

    avg_hourly_trades = hourly_trades / num_days
    avg_hourly_trades.name = "AverageNumberOfTrades"
    return avg_hourly_trades


def average_trades_by_hour_outcome(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """
    Calculate the average number of winning and losing trades for each hour.
    A trade is a "sell" action. A win has Result > 0, a loss has Result <= 0.
    The average is calculated over all days with any trading activity.
    """
    trades_only = df[~df["Action"].str.contains("deposit|withdraw", case=False, na=False)].copy()
    all_hours = pd.RangeIndex(start=0, stop=24, name="Hour")
    empty_series = pd.Series(0, index=all_hours, dtype=float)

    if trades_only.empty:
        return empty_series, empty_series

    num_trading_days = trades_only["Date"].dt.date.nunique()
    if num_trading_days == 0:
        return empty_series, empty_series

    sells = trades_only[trades_only["Action"].str.contains("sell", case=False, na=False)].copy()
    sells["Hour"] = sells["Date"].dt.hour

    wins = sells[sells["Result"] > 0]
    losses = sells[sells["Result"] <= 0]

    hourly_wins = wins.groupby("Hour").size()
    hourly_losses = losses.groupby("Hour").size()

    avg_hourly_wins = (hourly_wins / num_trading_days).reindex(all_hours, fill_value=0)
    avg_hourly_losses = (hourly_losses / num_trading_days).reindex(all_hours, fill_value=0)

    avg_hourly_wins.name = "AverageWinningTrades"
    avg_hourly_losses.name = "AverageLosingTrades"

    return avg_hourly_wins, avg_hourly_losses


def average_trade_value_by_hour_outcome(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """
    Calculate the average win/loss value for each hour of the day.
    A trade is a "sell" action. A win has Result > 0, a loss has Result <= 0.
    The average is calculated over all days with any trading activity.
    """
    trades_only = df[~df["Action"].str.contains("deposit|withdraw", case=False, na=False)].copy()
    all_hours = pd.RangeIndex(start=0, stop=24, name="Hour")
    empty_series = pd.Series(0, index=all_hours, dtype=float)

    if trades_only.empty:
        return empty_series, empty_series

    num_trading_days = trades_only["Date"].dt.date.nunique()
    if num_trading_days == 0:
        return empty_series, empty_series

    sells = trades_only[trades_only["Action"].str.contains("sell", case=False, na=False)].copy()
    sells["Hour"] = sells["Date"].dt.hour

    wins = sells[sells["Result"] > 0]
    losses = sells[sells["Result"] <= 0]

    # Sum the 'Result' for wins and losses per hour
    hourly_win_value = wins.groupby("Hour")["Result"].sum()
    hourly_loss_value = losses.groupby("Hour")["Result"].sum()

    # Calculate average over the number of trading days
    avg_hourly_win_value = (hourly_win_value / num_trading_days).reindex(all_hours, fill_value=0)
    avg_hourly_loss_value = (hourly_loss_value / num_trading_days).reindex(all_hours, fill_value=0)

    avg_hourly_win_value.name = "AverageWinValue"
    avg_hourly_loss_value.name = "AverageLossValue"

    return avg_hourly_win_value, avg_hourly_loss_value


def pnl_by_period(df: pd.DataFrame, period: str) -> pd.Series:
    """Calculate realised PnL aggregated by a given period ('W' for week, 'M' for month)."""
    sells = df[df["Action"].str.contains("sell", case=False, na=False)].copy()
    if sells.empty or "Result" not in sells.columns or "Date" not in sells.columns:
        return pd.Series(dtype=float)

    sells = sells.set_index("Date")
    pnl = sells["Result"].resample(period).sum()
    pnl = pnl[pnl != 0]

    if pnl.empty:
        return pd.Series(dtype=float)

    if period == 'W':
        pnl.index = pnl.index.strftime('%Y-W%U')
    elif period == 'M':
        pnl.index = pnl.index.strftime('%Y-%m')
    
    pnl.name = "PnL"
    return pnl


def net_pnl_by_period(df: pd.DataFrame, period: str) -> pd.Series:
    """Calculate realised PnL net of fees, aggregated by a given period ('W' for week, 'M' for month)."""
    daily_net = daily_net_realised_pnl(df)
    if daily_net.empty:
        return pd.Series(dtype=float)

    # The index of daily_net is strings of dates, so convert to datetime for resampling
    daily_net.index = pd.to_datetime(daily_net.index)
    
    net_pnl = daily_net.resample(period).sum()
    net_pnl = net_pnl[net_pnl != 0]

    if net_pnl.empty:
        return pd.Series(dtype=float)
        
    if period == 'W':
        net_pnl.index = net_pnl.index.strftime('%Y-W%U')
    elif period == 'M':
        net_pnl.index = net_pnl.index.strftime('%Y-%m')

    net_pnl.name = "NetPnL"
    return net_pnl


def trading_accuracy_by_hour(df: pd.DataFrame) -> pd.Series:
    """Calculate trading accuracy for each hour of the day."""
    sells = df[df["Action"].str.contains("sell", case=False, na=False)].copy()
    if sells.empty:
        return pd.Series(dtype=float)

    sells["Hour"] = sells["Date"].dt.hour
    
    hourly_wins = sells[sells["Result"] > 0].groupby("Hour").size()
    total_hourly_trades = sells.groupby("Hour").size()

    all_hours = pd.RangeIndex(start=0, stop=24, name="Hour")
    hourly_wins = hourly_wins.reindex(all_hours, fill_value=0)
    total_hourly_trades = total_hourly_trades.reindex(all_hours, fill_value=0)
    
    accuracy = (hourly_wins / total_hourly_trades).fillna(0)
    accuracy.name = "Accuracy"
    return accuracy


def trading_accuracy_by_period(df: pd.DataFrame, period: str) -> pd.Series:
    """Calculate trading accuracy aggregated by a given period ('D' for day, 'W' for week)."""
    sells = df[df["Action"].str.contains("sell", case=False, na=False)].copy()
    if sells.empty:
        return pd.Series(dtype=float)

    sells_indexed = sells.set_index("Date")
    
    wins = (sells_indexed["Result"] > 0).astype(int)
    
    periodic_wins = wins.resample(period).sum()
    periodic_total_trades = sells_indexed["Result"].resample(period).count()
    
    accuracy = (periodic_wins / periodic_total_trades).fillna(0)
    accuracy = accuracy[periodic_total_trades > 0]

    if accuracy.empty:
        return pd.Series(dtype=float)

    if period == 'W':
        accuracy.index = accuracy.index.strftime('%Y-W%U')
    elif period == 'D':
        accuracy.index = accuracy.index.to_series().dt.date.astype(str)
    
    accuracy.name = "Accuracy"
    return accuracy


###############################################################################
# Streamlit UI
###############################################################################

def main():
    st.set_page_config(page_title="Trading Dashboard", layout="wide")
    st.title("Personal Trading Analytics Dashboard")

    st.sidebar.header("Upload your CSV")
    uploaded = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded:
        df = load_trades(uploaded)

        # --- Metrics ---
        realised = realised_pnl(df)
        fees = total_fees(df)
        deposits = net_deposits(df)

        positions = current_positions(df)
        positions_detail = compute_unrealised_pnl(df, positions)
        unrealised = positions_detail["Unrealised"].sum()
        net_pnl = realised + unrealised - fees

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Net Deposits", f"£{deposits:,.2f}")
        col2.metric("Realised PnL", f"£{realised:,.2f}")
        col3.metric("Unrealised PnL", f"£{unrealised:,.2f}")
        col4.metric("Total Fees", f"£{fees:,.2f}")

        st.subheader("Current Positions")
        st.dataframe(positions_detail, use_container_width=True)

        # --- Equity curve chart ---
        ec = equity_curve(df)
        fig = px.line(ec, labels={"value": "Equity (£)", "index": "Date"}, title="Equity Curve")
        st.plotly_chart(fig, use_container_width=True)

        # --- PnL by Ticker ---
        pnl_by_ticker = df.groupby("Ticker")["Result"].sum()
        pnl_fig = px.bar(pnl_by_ticker, text_auto=True, title="Realised PnL by Ticker")
        st.plotly_chart(pnl_fig, use_container_width=True)

        # --- PnL by Day ---
        daily_pnl = realised_pnl_by_day(df)
        if not daily_pnl.empty:
            daily_pnl_fig = px.bar(daily_pnl, text_auto=True, title="Realised PnL by Day")
            daily_pnl_fig.update_layout(showlegend=False)
            st.plotly_chart(daily_pnl_fig, use_container_width=True)

        # --- Net PnL by Day (including fees) ---
        daily_net_pnl = daily_net_realised_pnl(df)
        if not daily_net_pnl.empty:
            daily_net_pnl_fig = px.bar(
                daily_net_pnl, text_auto=True, title="Daily Net Realised PnL (including fees)"
            )
            daily_net_pnl_fig.update_layout(showlegend=False)
            st.plotly_chart(daily_net_pnl_fig, use_container_width=True)

        # --- Fees by Day ---
        daily_fees = fees_by_day(df)
        if not daily_fees.empty:
            daily_fees_fig = px.bar(daily_fees, text_auto=True, title="Fees by Day")
            daily_fees_fig.update_layout(showlegend=False)
            st.plotly_chart(daily_fees_fig, use_container_width=True)

        # --- Number of Trades per Day ---
        daily_trades = trades_per_day(df)
        if not daily_trades.empty:
            daily_trades_fig = px.bar(
                daily_trades, text_auto=True, title="Number of Trades per Day"
            )
            daily_trades_fig.update_layout(showlegend=False)
            st.plotly_chart(daily_trades_fig, use_container_width=True)

        # --- Average Trades by Hour ---
        avg_hourly_trades = average_trades_by_hour(df)
        if not avg_hourly_trades.empty:
            avg_hourly_trades_fig = px.bar(
                avg_hourly_trades, text_auto=".2f", title="Average Trades per Hour"
            )
            avg_hourly_trades_fig.update_layout(showlegend=False, xaxis_title="Hour of Day")
            st.plotly_chart(avg_hourly_trades_fig, use_container_width=True)

        # --- Average Winning/Losing Trades by Hour ---
        avg_hourly_wins, avg_hourly_losses = average_trades_by_hour_outcome(df)

        col1, col2 = st.columns(2)
        with col1:
            if not avg_hourly_wins.empty:
                win_fig = px.bar(
                    avg_hourly_wins,
                    text_auto=".2f",
                    title="Average Winning Trades per Hour",
                    color_discrete_sequence=["green"],
                )
                win_fig.update_layout(showlegend=False, xaxis_title="Hour of Day")
                st.plotly_chart(win_fig, use_container_width=True)

        with col2:
            if not avg_hourly_losses.empty:
                loss_fig = px.bar(
                    avg_hourly_losses,
                    text_auto=".2f",
                    title="Average Losing Trades per Hour",
                    color_discrete_sequence=["red"],
                )
                loss_fig.update_layout(showlegend=False, xaxis_title="Hour of Day")
                st.plotly_chart(loss_fig, use_container_width=True)

        # --- Average Win/Loss Value by Hour ---
        avg_hourly_win_value, avg_hourly_loss_value = average_trade_value_by_hour_outcome(df)

        col1, col2 = st.columns(2)
        with col1:
            if not avg_hourly_win_value.empty:
                win_val_fig = px.bar(
                    avg_hourly_win_value,
                    text_auto=".2f",
                    title="Average Win Value per Hour (£)",
                    color_discrete_sequence=["green"],
                )
                win_val_fig.update_layout(showlegend=False, xaxis_title="Hour of Day")
                st.plotly_chart(win_val_fig, use_container_width=True)

        with col2:
            if not avg_hourly_loss_value.empty:
                loss_val_fig = px.bar(
                    avg_hourly_loss_value,
                    text_auto=".2f",
                    title="Average Loss Value per Hour (£)",
                    color_discrete_sequence=["red"],
                )
                loss_val_fig.update_layout(showlegend=False, xaxis_title="Hour of Day")
                st.plotly_chart(loss_val_fig, use_container_width=True)

        # --- PnL by Week ---
        weekly_pnl = pnl_by_period(df, "W")
        if not weekly_pnl.empty:
            weekly_pnl_fig = px.bar(
                weekly_pnl, text_auto=True, title="Realised PnL by Week"
            )
            weekly_pnl_fig.update_layout(showlegend=False, yaxis_title="PnL (£)")
            st.plotly_chart(weekly_pnl_fig, use_container_width=True)

        # --- Net PnL by Week (including fees) ---
        weekly_net_pnl = net_pnl_by_period(df, "W")
        if not weekly_net_pnl.empty:
            weekly_net_pnl_fig = px.bar(
                weekly_net_pnl, text_auto=True, title="Net Realised PnL by Week (including fees)"
            )
            weekly_net_pnl_fig.update_layout(showlegend=False, yaxis_title="Net PnL (£)")
            st.plotly_chart(weekly_net_pnl_fig, use_container_width=True)

        # --- PnL by Month ---
        monthly_pnl = pnl_by_period(df, "M")
        if not monthly_pnl.empty:
            monthly_pnl_fig = px.bar(
                monthly_pnl, text_auto=True, title="Realised PnL by Month"
            )
            monthly_pnl_fig.update_layout(showlegend=False, yaxis_title="PnL (£)")
            st.plotly_chart(monthly_pnl_fig, use_container_width=True)

        # --- Trading Accuracy ---
        st.subheader("Trading Accuracy")

        # By Hour
        hourly_accuracy = trading_accuracy_by_hour(df)
        if not hourly_accuracy.empty:
            hourly_acc_fig = px.bar(
                hourly_accuracy,
                text_auto=".0%",
                title="Trading Accuracy by Hour",
            )
            hourly_acc_fig.update_layout(
                showlegend=False, yaxis_title="Accuracy", yaxis_tickformat=".0%", xaxis_title="Hour of Day"
            )
            st.plotly_chart(hourly_acc_fig, use_container_width=True)

        # By Day
        daily_accuracy = trading_accuracy_by_period(df, "D")
        if not daily_accuracy.empty:
            daily_acc_fig = px.line(
                daily_accuracy,
                text=daily_accuracy.apply(lambda x: f"{x:.0%}"),
                title="Trading Accuracy by Day",
            )
            daily_acc_fig.update_traces(mode="lines+markers+text", textposition="top center")
            daily_acc_fig.update_layout(
                showlegend=False, yaxis_title="Accuracy", yaxis_tickformat=".0%", xaxis_title="Date"
            )
            st.plotly_chart(daily_acc_fig, use_container_width=True)

        # By Week
        weekly_accuracy = trading_accuracy_by_period(df, "W")
        if not weekly_accuracy.empty:
            weekly_acc_fig = px.line(
                weekly_accuracy,
                text=weekly_accuracy.apply(lambda x: f"{x:.0%}"),
                title="Trading Accuracy by Week",
            )
            weekly_acc_fig.update_traces(mode="lines+markers+text", textposition="top center")
            weekly_acc_fig.update_layout(
                showlegend=False, yaxis_title="Accuracy", yaxis_tickformat=".0%", xaxis_title="Week"
            )
            st.plotly_chart(weekly_acc_fig, use_container_width=True)

        st.subheader("Raw Trades Table")
        st.dataframe(df, use_container_width=True)
    else:
        st.info("Please upload a CSV file using the sidebar.")


if __name__ == "__main__":
    main()
