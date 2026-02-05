import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from datetime import date, timedelta

# --- Page Config ---
st.set_page_config(page_title="ASX Options Strategy Visualizer", layout="wide")

# --- Session State Initialization ---
if 'legs' not in st.session_state:
    st.session_state.legs = []
if 'reference_price' not in st.session_state:
    st.session_state.reference_price = 10.0  # Default price
if 'asx_code' not in st.session_state:
    st.session_state.asx_code = "BHP" # Default code

# --- Helper Functions ---
def fetch_price():
    code = st.session_state.asx_code_input.upper()
    if not code.endswith('.AX'):
        code += '.AX'
    try:
        ticker = yf.Ticker(code)
        # Try fetching current price, fallback to previous close
        price = ticker.info.get('currentPrice')
        if price is None:
            price = ticker.info.get('previousClose')
            st.warning(f"Could not fetch live price. Using previous close for {code}.")

        if price:
            st.session_state.reference_price = price
            # Update share price for any existing 'Share' legs
            for i, leg in enumerate(st.session_state.legs):
                if leg['type'] == 'Share':
                    st.session_state.legs[i]['price'] = price

    except Exception as e:
        st.error(f"Error fetching price for {code}: {e}")

def add_leg(leg_type='Long Call'):
    # Default values for new legs
    ref_price = st.session_state.reference_price
    new_leg = {
        'type': leg_type,
        'quantity': 100,
        'expiry': date.today() + timedelta(days=30),
        'id': len(st.session_state.legs) + 1 # Unique ID for keys
    }
    if leg_type == 'Share':
        new_leg.update({'price': ref_price})
    else:
        # Default to ATM strike and some premium
        strike = round(ref_price, 2)
        new_leg.update({
            'strike': strike,
            'strike_pct': 100.0,
            'premium_per_opt': round(ref_price * 0.05, 2)
        })
    st.session_state.legs.append(new_leg)

def remove_leg(index):
    st.session_state.legs.pop(index)

# --- Strategy Preset Functions ---
def clear_strategies():
    st.session_state.legs = []

def add_bull_call_spread():
    clear_strategies()
    ref_price = st.session_state.reference_price
    # Long ATM Call
    st.session_state.legs.append({
        'type': 'Long Call', 'quantity': 100, 'strike': ref_price,
        'strike_pct': 100.0, 'premium_per_opt': ref_price*0.05,
        'expiry': date.today() + timedelta(days=30), 'id': 1
    })
    # Short OTM Call
    st.session_state.legs.append({
        'type': 'Short Call', 'quantity': 100, 'strike': ref_price*1.05,
        'strike_pct': 105.0, 'premium_per_opt': ref_price*0.02,
        'expiry': date.today() + timedelta(days=30), 'id': 2
    })

def add_covered_call():
    clear_strategies()
    ref_price = st.session_state.reference_price
    # Long Share
    st.session_state.legs.append({
        'type': 'Share', 'quantity': 100, 'price': ref_price, 'id': 1
    })
    # Short OTM Call
    st.session_state.legs.append({
        'type': 'Short Call', 'quantity': 100, 'strike': ref_price*1.05,
        'strike_pct': 105.0, 'premium_per_opt': ref_price*0.03,
        'expiry': date.today() + timedelta(days=30), 'id': 2
    })

def add_protected_put():
    clear_strategies()
    ref_price = st.session_state.reference_price
    # Long Share
    st.session_state.legs.append({
        'type': 'Share', 'quantity': 100, 'price': ref_price, 'id': 1
    })
    # Long ATM/OTM Put
    st.session_state.legs.append({
        'type': 'Long Put', 'quantity': 100, 'strike': ref_price*0.95,
        'strike_pct': 95.0, 'premium_per_opt': ref_price*0.03,
        'expiry': date.today() + timedelta(days=30), 'id': 2
    })

def add_bear_put_spread():
    clear_strategies()
    ref_price = st.session_state.reference_price
    # Long ATM Put
    st.session_state.legs.append({
        'type': 'Long Put', 'quantity': 100, 'strike': ref_price,
        'strike_pct': 100.0, 'premium_per_opt': ref_price*0.05,
        'expiry': date.today() + timedelta(days=30), 'id': 1
    })
    # Short OTM Put
    st.session_state.legs.append({
        'type': 'Short Put', 'quantity': 100, 'strike': ref_price*0.95,
        'strike_pct': 95.0, 'premium_per_opt': ref_price*0.02,
        'expiry': date.today() + timedelta(days=30), 'id': 2
    })

def add_bear_call_spread():
    clear_strategies()
    ref_price = st.session_state.reference_price
    # Short ATM Call
    st.session_state.legs.append({
        'type': 'Short Call', 'quantity': 100, 'strike': ref_price,
        'strike_pct': 100.0, 'premium_per_opt': ref_price*0.05,
        'expiry': date.today() + timedelta(days=30), 'id': 1
    })
    # Long OTM Call
    st.session_state.legs.append({
        'type': 'Long Call', 'quantity': 100, 'strike': ref_price*1.05,
        'strike_pct': 105.0, 'premium_per_opt': ref_price*0.02,
        'expiry': date.today() + timedelta(days=30), 'id': 2
    })

def add_long_straddle():
    clear_strategies()
    ref_price = st.session_state.reference_price
    # Long ATM Call
    st.session_state.legs.append({
        'type': 'Long Call', 'quantity': 100, 'strike': ref_price,
        'strike_pct': 100.0, 'premium_per_opt': ref_price*0.05,
        'expiry': date.today() + timedelta(days=30), 'id': 1
    })
    # Long ATM Put
    st.session_state.legs.append({
        'type': 'Long Put', 'quantity': 100, 'strike': ref_price,
        'strike_pct': 100.0, 'premium_per_opt': ref_price*0.05,
        'expiry': date.today() + timedelta(days=30), 'id': 2
    })

def add_collar():
    clear_strategies()
    ref_price = st.session_state.reference_price
    # Long Share
    st.session_state.legs.append({
        'type': 'Share', 'quantity': 100, 'price': ref_price, 'id': 1
    })
    # Long OTM Put
    st.session_state.legs.append({
        'type': 'Long Put', 'quantity': 100, 'strike': ref_price*0.95,
        'strike_pct': 95.0, 'premium_per_opt': ref_price*0.03,
        'expiry': date.today() + timedelta(days=30), 'id': 2
    })
    # Short OTM Call
    st.session_state.legs.append({
        'type': 'Short Call', 'quantity': 100, 'strike': ref_price*1.05,
        'strike_pct': 105.0, 'premium_per_opt': ref_price*0.03,
        'expiry': date.today() + timedelta(days=30), 'id': 3
    })

# --- Callbacks for Bi-directional Updates ---
def update_strike_from_pct(i, key):
    leg = st.session_state.legs[i]
    new_pct = st.session_state[key]
    leg['strike'] = round(st.session_state.reference_price * (new_pct / 100), 2)

def update_pct_from_strike(i, key):
    leg = st.session_state.legs[i]
    new_strike = st.session_state[key]
    if st.session_state.reference_price > 0:
        leg['strike_pct'] = round((new_strike / st.session_state.reference_price) * 100, 1)
    else:
        leg['strike_pct'] = 0.0


# --- Payoff Calculation ---
def calculate_payoff(price_range):
    total_pnl = np.zeros_like(price_range)
    
    for leg in st.session_state.legs:
        quantity = leg['quantity']
        
        if leg['type'] == 'Share':
            entry_price = leg['price']
            pnl = (price_range - entry_price) * quantity
            
        else: # Option leg
            strike = leg['strike']
            premium = leg['premium_per_opt']
            cost = premium * quantity
            
            if 'Call' in leg['type']:
                intrinsic_value = np.maximum(price_range - strike, 0)
            else: # Put
                intrinsic_value = np.maximum(strike - price_range, 0)
                
            if 'Long' in leg['type']:
                pnl = (intrinsic_value * quantity) - cost
            else: # Short
                pnl = cost - (intrinsic_value * quantity)
                
        total_pnl += pnl
        
    return total_pnl


# ================= MAIN UI =================

# --- Sidebar ---
with st.sidebar:
    st.header("1. Stock & Price")
    st.text_input("ASX Stock Code", key='asx_code_input', value=st.session_state.asx_code)
    st.button("Fetch Price", on_click=fetch_price)
    
    st.number_input("Reference Share Price ($)", 
                    value=st.session_state.reference_price, 
                    key='reference_price',
                    format="%.2f",
                    step=0.1)

    st.markdown("---")
    st.header("2. Chart Setting")
    min_price_default = st.session_state.reference_price * 0.7
    max_price_default = st.session_state.reference_price * 1.3
    
    min_chart_price = st.number_input("Min. Share Price ($)", value=float(round(min_price_default, 2)), step=1.0, format="%.2f")
    max_chart_price = st.number_input("Max Share Price ($)", value=float(round(max_price_default, 2)), step=1.0, format="%.2f")


# --- Main Page ---
st.title("ASX Options Strategy Visualizer")

# --- Strategy Presets ---
st.subheader("Common Options Strategies")
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.write("**Bullish**")
    st.button("Bull Call Spread", on_click=add_bull_call_spread, use_container_width=True)
    st.button("Covered Call", on_click=add_covered_call, use_container_width=True)
    st.button("Protected Put", on_click=add_protected_put, use_container_width=True)
with c2:
    st.write("**Bearish**")
    st.button("Bear Put Spread", on_click=add_bear_put_spread, use_container_width=True)
    st.button("Bear Call Spread", on_click=add_bear_call_spread, use_container_width=True)
with c3:
    st.write("**Neutral/Vol**")
    st.button("Long Straddle", on_click=add_long_straddle, use_container_width=True)
    st.button("Collar", on_click=add_collar, use_container_width=True)


# --- Strategy Builder ---
st.markdown("---")
st.subheader("Strategy Builder")

if not st.session_state.legs:
    st.info("No legs added yet. Click a preset strategy above or add a leg below.")

for i, leg in enumerate(st.session_state.legs):
    st.markdown(f"#### Leg {i+1}")
    cols = st.columns([1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1])
    
    # Leg Type selector - Update session state directly
    leg_type = cols[0].selectbox(
        "Type", 
        ["Share", "Long Call", "Short Call", "Long Put", "Short Put"], 
        index=["Share", "Long Call", "Short Call", "Long Put", "Short Put"].index(leg['type']),
        key=f"type_{leg['id']}"
    )
    # If type changed in the UI, update it in the state
    if leg_type != leg['type']:
        st.session_state.legs[i]['type'] = leg_type
        st.rerun() # Rerun to update the fields shown

    
    if leg['type'] == 'Share':
        qty = cols[1].number_input("No. of Share", value=leg['quantity'], min_value=1, step=100, key=f"qty_{leg['id']}")
        # Share price is fixed to the reference price for simplicity in this view
        price = cols[2].number_input("Share Price", value=st.session_state.reference_price, disabled=True, format="%.2f")
        val = qty * price
        cols[3].number_input("Share Value", value=val, disabled=True, format="$%.2f")
        # Update state
        st.session_state.legs[i]['quantity'] = qty
        st.session_state.legs[i]['price'] = price

    else: # Option Leg
        qty = cols[1].number_input("No. of Options", value=leg['quantity'], min_value=1, step=1, key=f"qty_{leg['id']}")
        
        # Bi-directional Strike Inputs
        strike_key = f"strike_{leg['id']}"
        pct_key = f"pct_{leg['id']}"

        strike = cols[2].number_input(
            "Exercise Price", 
            value=leg['strike'],
            format="%.2f",
            step=0.5,
            key=strike_key,
            on_change=update_pct_from_strike,
            args=(i, strike_key)
        )
        
        pct = cols[3].number_input(
            "Exercise %",
            value=leg['strike_pct'],
            format="%.1f",
            step=0.5,
            suffix="%",
            key=pct_key,
            on_change=update_strike_from_pct,
            args=(i, pct_key)
        )

        # Premium Inputs
        prem_per = cols[4].number_input("Cost/Opt", value=leg['premium_per_opt'], format="%.2f", step=0.05, key=f"prem_{leg['id']}")
        total_prem = qty * prem_per
        cols[5].number_input("Total Premium", value=total_prem, disabled=True, format="$%.2f")
        
        expiry = cols[6].date_input("Expiry Date", value=leg['expiry'], key=f"exp_{leg['id']}")

        # Update state
        st.session_state.legs[i]['quantity'] = qty
        st.session_state.legs[i]['premium_per_opt'] = prem_per
        st.session_state.legs[i]['expiry'] = expiry


    if cols[-1].button("Remove", key=f"rem_{leg['id']}"):
        remove_leg(i)
        st.rerun()

st.button("Add New Leg", on_click=add_leg)


# --- Payoff Analysis & Chart ---
st.markdown("---")
st.subheader("Payoff Analysis at Maturity")

if st.session_state.legs and min_chart_price < max_chart_price:
    # Generate price range for X-axis
    price_range = np.linspace(min_chart_price, max_chart_price, 500)
    
    # Calculate P&L
    total_pnl = calculate_payoff(price_range)
    
    # Find Breakeven Points
    # Find where sign changes from - to + or + to -
    signs = np.sign(total_pnl)
    sign_changes = ((np.roll(signs, 1) - signs) != 0).astype(int)
    sign_changes[0] = 0 # Ignore first element
    be_indices = np.where(sign_changes == 1)[0]
    breakeven_prices = price_range[be_indices]
    
    be_text = ", ".join([f"${p:.2f}" for p in breakeven_prices]) if len(breakeven_prices) > 0 else "None in range"

    # Create Plotly Chart
    fig = go.Figure()

    # 1. Add the Profit/Loss Line
    fig.add_trace(go.Scatter(
        x=price_range, 
        y=total_pnl,
        mode='lines',
        name='Total P&L',
        line=dict(color='#1f77b4', width=3),
        fill='tozeroy', # Fill to the Y=0 line
        fillcolor='rgba(31, 119, 180, 0.3)' # Semi-transparent blue
    ))

    # 2. Add Zero Line (Breakeven Line)
    fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=1.5)
    
    # 3. Add Breakeven Annotations
    for be_price in breakeven_prices:
        fig.add_annotation(
            x=be_price, y=0,
            text=f"BE: ${be_price:.2f}",
            showarrow=True, arrowhead=2, ax=0, ay=-40
        )

    # 4. Chart Layout Styling
    fig.update_layout(
        title="Strategy Profit/Loss at Expiration",
        xaxis_title="Stock Price at Expiration ($)",
        yaxis_title="Profit / Loss ($)",
        xaxis_tickformat='$.2f',
        yaxis_tickformat='$.0f',
        hovermode="x unified",
        height=600,
        annotations=[dict(
            x=0.02, y=0.98, xref='paper', yref='paper',
            text=f"<b>Breakeven Point(s):</b> {be_text}",
            showarrow=False,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='gray'
        )]
    )
    
    # Color the fill based on profit/loss (advanced trick)
    # We need to split the line into two traces, one for profit (green) and one for loss (red)
    # This is complex to do perfectly in Plotly without more advanced data manipulation.
    # The simple 'tozeroy' fill provides a good enough visual representation for now.

    st.plotly_chart(fig, use_container_width=True)

else:
    st.write("Add legs to the strategy to see the payoff diagram.")
    if min_chart_price >= max_chart_price:
        st.error("Error: Chart Min Price must be less than Max Price.")
