import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="ASX Options Strategy Visualizer",
    page_icon="📈",
    layout="wide",
)

# --- APP TITLE ---
st.title("ASX Options Strategy Visualizer")
st.caption("A tool for constructing and visualizing multi-leg option strategies at maturity.")


# --- CORE FUNCTIONS ---

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_stock_price(ticker):
    """Fetches the last closing price of a given stock ticker from Yahoo Finance."""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1d")
        if not hist.empty:
            return hist['Close'].iloc[-1]
    except Exception:
        return None
    return None

def calculate_payoff(leg_type, strike, premium, quantity, s, initial_price):
    """Calculates the profit/loss for a single leg of the strategy."""
    if leg_type == "Long Share":
        return (s - initial_price) * quantity
    elif leg_type == "Long Call":
        return (np.maximum(0, s - strike) - premium) * 100 * quantity
    elif leg_type == "Short Call":
        return (premium - np.maximum(0, s - strike)) * 100 * quantity
    elif leg_type == "Long Put":
        return (np.maximum(0, strike - s) - premium) * 100 * quantity
    elif leg_type == "Short Put":
        return (premium - np.maximum(0, strike - s)) * 100 * quantity
    return 0

def find_breakeven_points(price_range, total_pnl):
    """
    Identifies the breakeven points by finding where the P&L crosses the zero line.
    """
    # Find where the sign of the PnL changes
    sign_changes = np.where(np.diff(np.sign(total_pnl)))[0]
    
    breakevens = []
    for idx in sign_changes:
        # Linear interpolation to find a more precise breakeven point
        p1, pnl1 = price_range[idx], total_pnl[idx]
        p2, pnl2 = price_range[idx + 1], total_pnl[idx + 1]
        
        if pnl1 * pnl2 < 0: # Ensure they are on opposite sides of zero
            breakeven = p1 - pnl1 * (p2 - p1) / (pnl2 - pnl1)
            breakevens.append(breakeven)
            
    return breakevens

# --- SESSION STATE INITIALIZATION ---
if 'legs' not in st.session_state:
    st.session_state.legs = []
if 'ref_price' not in st.session_state:
    st.session_state.ref_price = 100.0

# --- SIDEBAR - INPUTS & CONTROLS ---
with st.sidebar:
    st.header("1. Stock & Price")
    
    # ASX Stock Data
    stock_code = st.text_input("ASX Stock Code (e.g., BHP, CBA)", "BHP").upper()
    
    col1, col2 = st.columns([1,1])
    with col1:
        if st.button("Fetch Price", use_container_width=True):
            if not stock_code.endswith(".AX"):
                ticker = stock_code + ".AX"
            else:
                ticker = stock_code
            
            price = get_stock_price(ticker)
            if price:
                st.session_state.ref_price = round(price, 2)
                st.success(f"Fetched {stock_code}: ${st.session_state.ref_price}")
            else:
                st.error(f"Could not fetch price for {stock_code}. Check the code.")

    # Reference Share Price
    st.session_state.ref_price = st.number_input(
        "Reference Share Price ($)",
        value=st.session_state.ref_price,
        step=0.1,
        format="%.2f",
        help="Auto-populates on fetch, but can be manually overridden."
    )
    
    st.divider()

    # Inputs & Variables
    st.header("2. Chart & Fees")
    fee = st.number_input("Bank/Brokerage Fee ($)", value=10.0, min_value=0.0, step=1.0)
    
    st.subheader("Chart Range (X-Axis)")
    chart_min = st.number_input("Min Share Price ($)", value=float(st.session_state.ref_price * 0.8), step=1.0)
    chart_max = st.number_input("Max Share Price ($)", value=float(st.session_state.ref_price * 1.2), step=1.0)
    
    if chart_min >= chart_max:
        st.warning("Min price must be less than Max price.")

# --- MAIN CONTENT - STRATEGY BUILDER ---
st.header("Strategy Builder")
st.write("Construct your options strategy by adding one or more 'legs'.")

# Leg addition form
with st.form("add_leg_form", clear_on_submit=True):
    st.subheader("Add a New Leg")
    cols = st.columns([2, 1, 1, 1, 2])
    leg_type = cols[0].selectbox("Type", ["Long Share", "Long Call", "Short Call", "Long Put", "Short Put"])
    strike = cols[1].number_input("Strike", value=100.0, step=0.50, format="%.2f")
    premium = cols[2].number_input("Premium/Cost", value=2.50, step=0.01, format="%.2f")
    quantity = cols[3].number_input("Quantity", value=1, min_value=1, step=1)
    notes = cols[4].text_input("Expiry/Notes", "e.g., Mar 2026")

    submitted = st.form_submit_button("Add Leg")
    if submitted:
        new_leg = {
            "type": leg_type,
            "strike": strike,
            "premium": premium,
            "quantity": quantity,
            "notes": notes,
        }
        st.session_state.legs.append(new_leg)
        st.success("Leg added successfully!")

# Display and manage current legs
if st.session_state.legs:
    st.subheader("Current Strategy Legs")
    for i, leg in enumerate(st.session_state.legs):
        cols = st.columns([2, 1, 1, 1, 2, 1])
        cols[0].write(f"**{leg['type']}**")
        cols[1].write(f"${leg['strike']:.2f}")
        cols[2].write(f"${leg['premium']:.2f}")
        cols[3].write(f"{leg['quantity']}")
        cols[4].write(f"{leg['notes']}")
        if cols[5].button("Remove", key=f"remove_{i}"):
            st.session_state.legs.pop(i)
            st.rerun()
    st.divider()

# --- ANALYSIS & VISUALIZATION ---
if st.session_state.legs and chart_min < chart_max:
    st.header("Payoff Analysis")

    # 1. Generate price range for X-axis
    s_range = np.linspace(chart_min, chart_max, 200)
    
    # 2. Calculate P&L for each leg and sum them up
    total_pnl = np.zeros_like(s_range)
    for leg in st.session_state.legs:
        total_pnl += calculate_payoff(
            leg_type=leg['type'],
            strike=leg['strike'],
            premium=leg['premium'],
            quantity=leg['quantity'],
            s=s_range,
            initial_price=st.session_state.ref_price
        )
    
    # 3. Subtract brokerage fees
    total_pnl -= fee
    
    # 4. Calculate Benchmark P&L (Buy and Hold)
    benchmark_pnl = (s_range - st.session_state.ref_price) - (fee / 100) # Simple fee adjust
    
    # 5. Create Plotly Chart
    fig = go.Figure()

    # Add Strategy P&L trace
    fig.add_trace(go.Scatter(
        x=s_range, 
        y=total_pnl, 
        mode='lines', 
        name='Strategy P&L',
        line=dict(color='royalblue', width=3)
    ))
    
    # Add Benchmark P&L trace
    fig.add_trace(go.Scatter(
        x=s_range, 
        y=benchmark_pnl, 
        mode='lines', 
        name='Buy & Hold Benchmark',
        line=dict(color='grey', width=2, dash='dash')
    ))

    # Add Zero line
    fig.add_hline(
        y=0, 
        line_width=1, 
        line_dash="dot", 
        line_color="black"
    )

    # Fill colors for profit/loss zones for the strategy
    fig.add_trace(go.Scatter(
        x=s_range,
        y=total_pnl.clip(lower=0),
        fill='tozeroy',
        fillcolor='rgba(0, 200, 100, 0.2)',
        line=dict(width=0),
        name='Profit Zone',
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=s_range,
        y=total_pnl.clip(upper=0),
        fill='tozeroy',
        fillcolor='rgba(255, 100, 100, 0.2)',
        line=dict(width=0),
        name='Loss Zone',
        showlegend=False
    ))

    # Update layout
    fig.update_layout(
        title=f'Strategy Payoff vs. Buy & Hold for {stock_code}',
        xaxis_title='Share Price at Maturity ($)',
        yaxis_title='Profit / Loss ($)',
        legend=dict(x=0.01, y=0.99),
        xaxis=dict(gridcolor='lightgrey'),
        yaxis=dict(gridcolor='lightgrey'),
        plot_bgcolor='white'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # --- SUMMARY TABLE ---
    st.subheader("Strategy Summary")
    
    max_profit = np.max(total_pnl)
    max_loss = np.min(total_pnl)
    breakeven_points = find_breakeven_points(s_range, total_pnl)

    summary_data = {
        "Metric": ["Estimated Max Profit", "Estimated Max Loss", "Breakeven Point(s)"],
        "Value": [
            f"${max_profit:,.2f}" if not np.isinf(max_profit) else "Unlimited",
            f"${max_loss:,.2f}" if not np.isinf(max_loss) else "Unlimited",
            ", ".join([f"${p:.2f}" for p in breakeven_points]) if breakeven_points else "N/A"
        ]
    }
    
    st.table(pd.DataFrame(summary_data))
    
elif not st.session_state.legs:
    st.info("Add one or more legs to the strategy to see the analysis.")
