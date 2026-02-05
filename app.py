import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from datetime import date, timedelta
import uuid

# --- Page Config ---
st.set_page_config(page_title="ASX Options Visualizer", layout="wide")

# --- Session State Init ---
if 'legs' not in st.session_state:
    st.session_state.legs = []
if 'reference_price' not in st.session_state:
    st.session_state.reference_price = 10.0
if 'asx_code' not in st.session_state:
    st.session_state.asx_code = "BHP"

# --- Helper Functions ---
def fetch_price():
    code = st.session_state.asx_code_input.upper()
    if not code.endswith('.AX'):
        code += '.AX'
    try:
        ticker = yf.Ticker(code)
        # Try fast fetch, fallback to history
        price = ticker.info.get('currentPrice') or ticker.info.get('previousClose')
        # If still none, try history (sometimes reliable for ASX)
        if not price:
            hist = ticker.history(period="1d")
            if not hist.empty:
                price = hist['Close'].iloc[-1]

        if price:
            st.session_state.reference_price = float(price)
            # Update Share legs automatically
            for leg in st.session_state.legs:
                if leg['type'] == 'Share':
                    leg['price'] = float(price)
    except Exception as e:
        st.error(f"Error fetching price: {e}")

def create_leg(leg_type='Long Call'):
    ref_price = float(st.session_state.reference_price)
    leg_id = str(uuid.uuid4())
    
    new_leg = {
        'id': leg_id,
        'type': leg_type,
        'quantity': 100,
        'expiry': date.today() + timedelta(days=30),
    }
    
    if leg_type == 'Share':
        new_leg['price'] = ref_price
    else:
        # Defaults
        new_leg['strike'] = ref_price
        new_leg['strike_pct'] = 100.0
        new_leg['premium_per_opt'] = float(round(ref_price * 0.05, 2))
        
    return new_leg

def add_leg():
    st.session_state.legs.append(create_leg('Long Call'))

def remove_leg(leg_id):
    st.session_state.legs = [leg for leg in st.session_state.legs if leg['id'] != leg_id]

# --- Strategy Logic ---
def set_strategy(strategy_name):
    st.session_state.legs = []
    ref = float(st.session_state.reference_price)
    
    # --- BULLISH ---
    if strategy_name == "Bull Call Spread":
        l1 = create_leg('Long Call')
        l1['strike'] = ref; l1['strike_pct'] = 100.0
        
        l2 = create_leg('Short Call')
        l2['strike'] = ref * 1.05; l2['strike_pct'] = 105.0
        l2['premium_per_opt'] = ref * 0.02
        st.session_state.legs.extend([l1, l2])

    elif strategy_name == "Covered Call":
        l1 = create_leg('Share')
        l2 = create_leg('Short Call')
        l2['strike'] = ref * 1.05; l2['strike_pct'] = 105.0
        l2['premium_per_opt'] = ref * 0.02
        st.session_state.legs.extend([l1, l2])

    elif strategy_name == "Protected Put":
        l1 = create_leg('Share')
        l2 = create_leg('Long Put')
        l2['strike'] = ref * 0.95; l2['strike_pct'] = 95.0
        l2['premium_per_opt'] = ref * 0.03
        st.session_state.legs.extend([l1, l2])

    # --- BEARISH ---
    elif strategy_name == "Bear Put Spread":
        l1 = create_leg('Long Put')
        l1['strike'] = ref; l1['strike_pct'] = 100.0
        l1['premium_per_opt'] = ref * 0.05

        l2 = create_leg('Short Put')
        l2['strike'] = ref * 0.90; l2['strike_pct'] = 90.0
        l2['premium_per_opt'] = ref * 0.02
        st.session_state.legs.extend([l1, l2])
        
    elif strategy_name == "Bear Call Spread":
        l1 = create_leg('Short Call')
        l1['strike'] = ref; l1['strike_pct'] = 100.0
        l1['premium_per_opt'] = ref * 0.05

        l2 = create_leg('Long Call')
        l2['strike'] = ref * 1.10; l2['strike_pct'] = 110.0
        l2['premium_per_opt'] = ref * 0.01
        st.session_state.legs.extend([l1, l2])

    # --- NEUTRAL / VOL ---
    elif strategy_name == "Long Straddle":
        l1 = create_leg('Long Call')
        l1['strike'] = ref; l1['strike_pct'] = 100.0
        
        l2 = create_leg('Long Put')
        l2['strike'] = ref; l2['strike_pct'] = 100.0
        st.session_state.legs.extend([l1, l2])
        
    elif strategy_name == "Collar":
        l1 = create_leg('Share')
        l2 = create_leg('Long Put')
        l2['strike'] = ref * 0.95; l2['strike_pct'] = 95.0
        l3 = create_leg('Short Call')
        l3['strike'] = ref * 1.05; l3['strike_pct'] = 105.0
        st.session_state.legs.extend([l1, l2, l3])


# --- Bi-Directional Callbacks ---
def on_strike_change(lid, key):
    # User typed in Strike ($) -> We update %
    val = st.session_state[key]
    ref = st.session_state.reference_price
    for leg in st.session_state.legs:
        if leg['id'] == lid:
            leg['strike'] = val
            if ref > 0:
                leg['strike_pct'] = round((val / ref) * 100, 1)
            break

def on_pct_change(lid, key):
    # User typed in % -> We update Strike ($)
    val = st.session_state[key]
    ref = st.session_state.reference_price
    for leg in st.session_state.legs:
        if leg['id'] == lid:
            leg['strike_pct'] = val
            leg['strike'] = round(ref * (val / 100.0), 2)
            break

# --- Calculation ---
def calculate_payoff(price_range):
    total = np.zeros_like(price_range)
    for leg in st.session_state.legs:
        qty = leg['quantity']
        if leg['type'] == 'Share':
            pnl = (price_range - leg['price']) * qty
        else:
            strike = leg['strike']
            cost = leg['premium_per_opt'] * qty
            if 'Call' in leg['type']:
                intrinsic = np.maximum(price_range - strike, 0)
            else:
                intrinsic = np.maximum(strike - price_range, 0)
            
            if 'Long' in leg['type']:
                pnl = (intrinsic * qty) - cost
            else:
                pnl = cost - (intrinsic * qty)
        total += pnl
    return total

# ================= UI =================
with st.sidebar:
    st.header("1. Stock Setup")
    st.text_input("ASX Code", key='asx_code_input', value=st.session_state.asx_code)
    st.button("Fetch Price", on_click=fetch_price)
    st.number_input("Ref Price ($)", value=float(st.session_state.reference_price), key='reference_price', format="%.2f", step=0.1)

    st.markdown("---")
    st.header("2. Chart Zoom")
    ref = st.session_state.reference_price
    min_def = float(round(ref*0.7, 2))
    max_def = float(round(ref*1.3, 2))
    
    c1, c2 = st.columns(2)
    min_chart = c1.number_input("Min X", value=min_def, format="%.2f")
    max_chart = c2.number_input("Max X", value=max_def, format="%.2f")

# --- Main Area ---
st.title("ASX Options Visualizer")

st.markdown("### Quick Strategies")
row1 = st.columns(4)
if row1[0].button("Bull Call Spread", use_container_width=True): set_strategy("Bull Call Spread")
if row1[1].button("Covered Call", use_container_width=True): set_strategy("Covered Call")
if row1[2].button("Protected Put", use_container_width=True): set_strategy("Protected Put")
if row1[3].button("Collar", use_container_width=True): set_strategy("Collar")

row2 = st.columns(3)
if row2[0].button("Bear Put Spread", use_container_width=True): set_strategy("Bear Put Spread")
if row2[1].button("Bear Call Spread", use_container_width=True): set_strategy("Bear Call Spread")
if row2[2].button("Long Straddle", use_container_width=True): set_strategy("Long Straddle")

st.markdown("---")
st.markdown("### Strategy Builder")

if not st.session_state.legs:
    st.info("Select a strategy above or click '+ Add Leg'")

# Render Legs
for i, leg in enumerate(st.session_state.legs):
    lid = leg['id']
    st.markdown(f"**Leg {i+1}**")
    cols = st.columns([1.5, 1.2, 1.5, 1.5, 1.2, 1.2, 1.5, 0.5])
    
    # Type
    new_type = cols[0].selectbox("Type", ["Share", "Long Call", "Short Call", "Long Put", "Short Put"], 
                                 index=["Share", "Long Call", "Short Call", "Long Put", "Short Put"].index(leg['type']), 
                                 key=f"type_{lid}")
    if new_type != leg['type']:
        leg['type'] = new_type
        st.rerun()

    # Qty
    qty = cols[1].number_input("Qty", value=int(leg['quantity']), step=100, key=f"qty_{lid}")
    leg['quantity'] = qty

    if leg['type'] == 'Share':
        price = cols[2].number_input("Price", value=float(leg['price']), format="%.2f", key=f"s_price_{lid}")
        leg['price'] = price
        cols[3].metric("Value", f"${qty*price:,.0f}")
    else:
        # Bi-Directional Strike Logic
        # 1. Strike Input
        s_key = f"strike_{lid}"
        cols[2].number_input("Strike ($)", value=float(leg['strike']), step=0.5, format="%.2f", 
                           key=s_key, on_change=on_strike_change, args=(lid, s_key))
        
        # 2. Pct Input
        p_key = f"pct_{lid}"
        cols[3].number_input("Strike %", value=float(leg['strike_pct']), step=1.0, format="%.1f", 
                           key=p_key, on_change=on_pct_change, args=(lid, p_key))

        # Premium
        prem = cols[4].number_input("Prem ($)", value=float(leg['premium_per_opt']), step=0.05, format="%.2f", key=f"prem_{lid}")
        leg['premium_per_opt'] = prem
        cols[5].metric("Total", f"${qty*prem:,.0f}")
        cols[6].date_input("Expiry", value=leg['expiry'], key=f"exp_{lid}")

    if cols[7].button("X", key=f"del_{lid}"):
        remove_leg(lid)
        st.rerun()

st.button("+ Add Leg", on_click=add_leg)

# --- Visualization ---
st.markdown("---")
st.markdown("### Payoff Analysis")

if st.session_state.legs and min_chart < max_chart:
    price_range = np.linspace(min_chart, max_chart, 500)
    pnl = calculate_payoff(price_range)
    
    # Split PnL into Profit (Green) and Loss (Red) arrays for shading
    pnl_pos = np.where(pnl >= 0, pnl, 0)
    pnl_neg = np.where(pnl < 0, pnl, 0)

    fig = go.Figure()
    
    # 1. Green Area (Profit)
    fig.add_trace(go.Scatter(
        x=price_range, y=pnl_pos, mode='lines', name='Profit',
        line=dict(width=0), # Invisible line, just fill
        fill='tozeroy', fillcolor='rgba(144, 238, 144, 0.5)' # Light Green
    ))
    
    # 2. Red Area (Loss)
    fig.add_trace(go.Scatter(
        x=price_range, y=pnl_neg, mode='lines', name='Loss',
        line=dict(width=0), 
        fill='tozeroy', fillcolor='rgba(255, 182, 193, 0.5)' # Light Red
    ))
    
    # 3. Main P&L Line (Dark Blue)
    fig.add_trace(go.Scatter(
        x=price_range, y=pnl, mode='lines', name='Total P&L',
        line=dict(color='darkblue', width=3)
    ))

    # 4. Zero Line
    fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=1)

    # 5. Benchmark (Share Only)
    bench_pnl = (price_range - st.session_state.reference_price) * 100
    fig.add_trace(go.Scatter(
        x=price_range, y=bench_pnl, mode='lines', name='Share Only',
        line=dict(color='gray', dash='dot', width=1)
    ))

    # Layout - Taller Height
    fig.update_layout(
        title="Profit / Loss at Expiration",
        xaxis_title="Stock Price",
        yaxis_title="P&L ($)",
        hovermode="x unified",
        height=600, # Taller chart as requested
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    st.plotly_chart(fig, use_container_width=True)
