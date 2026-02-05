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

# --- Strategy Info Data (For the Popups) ---
STRATEGY_INFO = {
    "Bull Call Spread": {
        "mech": "Buy ITM/ATM Call + Sell OTM Call",
        "use": "Moderately Bullish. You want to profit from a rise but limit your upfront cost.",
    },
    "Covered Call": {
        "mech": "Own the Share + Sell a Call Option",
        "use": "Neutral to Slightly Bullish. You want to generate extra income (yield) from shares you already own.",
    },
    "Protected Put": {
        "mech": "Own the Share + Buy a Put Option",
        "use": "Bullish but Nervous. You want to hold the share but insure against a crash.",
    },
    "Bear Put Spread": {
        "mech": "Buy ITM/ATM Put + Sell OTM Put",
        "use": "Moderately Bearish. You expect a drop but want to reduce the cost of the Put.",
    },
    "Bear Call Spread": {
        "mech": "Sell ITM/ATM Call + Buy OTM Call",
        "use": "Bearish. You earn premium if the stock stays flat or drops.",
    },
    "Long Straddle": {
        "mech": "Buy Call + Buy Put (Same Strike)",
        "use": "High Volatility. You expect a HUGE move, but don't know which direction (e.g. Earnings report).",
    },
    "Collar": {
        "mech": "Own Share + Buy Put + Sell Call",
        "use": "Conservative. You want 'Free Insurance'. The sold Call pays for the protective Put.",
    }
}

# --- Helper Functions ---
def fetch_price():
    code = st.session_state.asx_code_input.upper()
    if not code.endswith('.AX'):
        code += '.AX'
    try:
        ticker = yf.Ticker(code)
        price = ticker.info.get('currentPrice') or ticker.info.get('previousClose')
        if not price:
            hist = ticker.history(period="1d")
            if not hist.empty:
                price = hist['Close'].iloc[-1]

        if price:
            st.session_state.reference_price = float(price)
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
    
    # BULLISH
    if strategy_name == "Bull Call Spread":
        l1 = create_leg('Long Call'); l1['strike'] = ref; l1['strike_pct'] = 100.0
        l2 = create_leg('Short Call'); l2['strike'] = ref * 1.05; l2['strike_pct'] = 105.0; l2['premium_per_opt'] = ref * 0.02
        st.session_state.legs.extend([l1, l2])

    elif strategy_name == "Covered Call":
        l1 = create_leg('Share')
        l2 = create_leg('Short Call'); l2['strike'] = ref * 1.05; l2['strike_pct'] = 105.0; l2['premium_per_opt'] = ref * 0.02
        st.session_state.legs.extend([l1, l2])

    elif strategy_name == "Protected Put":
        l1 = create_leg('Share')
        l2 = create_leg('Long Put'); l2['strike'] = ref * 0.95; l2['strike_pct'] = 95.0; l2['premium_per_opt'] = ref * 0.03
        st.session_state.legs.extend([l1, l2])

    # BEARISH
    elif strategy_name == "Bear Put Spread":
        l1 = create_leg('Long Put'); l1['strike'] = ref; l1['strike_pct'] = 100.0; l1['premium_per_opt'] = ref * 0.05
        l2 = create_leg('Short Put'); l2['strike'] = ref * 0.90; l2['strike_pct'] = 90.0; l2['premium_per_opt'] = ref * 0.02
        st.session_state.legs.extend([l1, l2])
        
    elif strategy_name == "Bear Call Spread":
        l1 = create_leg('Short Call'); l1['strike'] = ref; l1['strike_pct'] = 100.0; l1['premium_per_opt'] = ref * 0.05
        l2 = create_leg('Long Call'); l2['strike'] = ref * 1.10; l2['strike_pct'] = 110.0; l2['premium_per_opt'] = ref * 0.01
        st.session_state.legs.extend([l1, l2])

    # NEUTRAL
    elif strategy_name == "Long Straddle":
        l1 = create_leg('Long Call'); l1['strike'] = ref; l1['strike_pct'] = 100.0
        l2 = create_leg('Long Put'); l2['strike'] = ref; l2['strike_pct'] = 100.0
        st.session_state.legs.extend([l1, l2])
        
    elif strategy_name == "Collar":
        l1 = create_leg('Share')
        l2 = create_leg('Long Put'); l2['strike'] = ref * 0.95; l2['strike_pct'] = 95.0
        l3 = create_leg('Short Call'); l3['strike'] = ref * 1.05; l3['strike_pct'] = 105.0
        st.session_state.legs.extend([l1, l2, l3])

# --- Callbacks ---
def on_strike_change(lid, strike_key, pct_key):
    new_strike = st.session_state[strike_key]
    ref = st.session_state.reference_price
    for leg in st.session_state.legs:
        if leg['id'] == lid:
            leg['strike'] = new_strike
            if ref > 0:
                new_pct = round((new_strike / ref) * 100, 1)
                leg['strike_pct'] = new_pct
                st.session_state[pct_key] = new_pct
            break

def on_pct_change(lid, strike_key, pct_key):
    new_pct = st.session_state[pct_key]
    ref = st.session_state.reference_price
    for leg in st.session_state.legs:
        if leg['id'] == lid:
            leg['strike_pct'] = new_pct
            new_strike = round(ref * (new_pct / 100.0), 2)
            leg['strike'] = new_strike
            st.session_state[strike_key] = new_strike
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

# --- Helper UI: Strategy Button with Popup ---
def render_strat_btn(name):
    c1, c2 = st.columns([5, 1])
    if c1.button(name, use_container_width=True):
        set_strategy(name)
    
    # The "?" Popup
    info = STRATEGY_INFO.get(name, {})
    with c2.popover("?", use_container_width=True):
        st.markdown(f"**{name}**")
        st.markdown(f"**Mechanics:** {info.get('mech','')}")
        st.markdown(f"**When to use?** {info.get('use','')}")

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

# --- Strategy Buttons ---
st.markdown("### Common Strategies")
col_bull, col_bear, col_other = st.columns(3)

with col_bull:
    st.caption("BULLISH")
    render_strat_btn("Bull Call Spread")
    render_strat_btn("Covered Call")
    render_strat_btn("Protected Put")

with col_bear:
    st.caption("BEARISH")
    render_strat_btn("Bear Put Spread")
    render_strat_btn("Bear Call Spread")

with col_other:
    st.caption("OTHER")
    render_strat_btn("Long Straddle")
    render_strat_btn("Collar")

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
                                 key=f"type_{lid}", label_visibility="collapsed")
    if new_type != leg['type']:
        leg['type'] = new_type
        st.rerun()

    # Qty
    qty = cols[1].number_input("Qty", value=int(leg['quantity']), step=100, key=f"qty_{lid}", label_visibility="collapsed")
    leg['quantity'] = qty

    if leg['type'] == 'Share':
        price = cols[2].number_input("Price", value=float(leg['price']), format="%.2f", key=f"s_price_{lid}", label_visibility="collapsed")
        leg['price'] = price
        # Small Value Text
        cols[3].markdown(f"**Val:** ${qty*price:,.0f}")
    else:
        s_key = f"strike_{lid}"
        p_key = f"pct_{lid}"
        
        cols[2].number_input("Strike", value=float(leg['strike']), step=0.5, format="%.2f", 
                           key=s_key, on_change=on_strike_change, args=(lid, s_key, p_key), label_visibility="collapsed")
        
        cols[3].number_input("Pct", value=float(leg['strike_pct']), step=1.0, format="%.1f", 
                           key=p_key, on_change=on_pct_change, args=(lid, s_key, p_key), label_visibility="collapsed")

        prem = cols[4].number_input("Prem", value=float(leg['premium_per_opt']), step=0.05, format="%.2f", key=f"prem_{lid}", label_visibility="collapsed")
        leg['premium_per_opt'] = prem
        # Small Value Text
        cols[5].markdown(f"**Tot:** ${qty*prem:,.0f}")
        cols[6].date_input("Exp", value=leg['expiry'], key=f"exp_{lid}", label_visibility="collapsed")

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
    
    pnl_pos = np.where(pnl >= 0, pnl, 0)
    pnl_neg = np.where(pnl < 0, pnl, 0)

    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=price_range, y=pnl_pos, mode='lines', name='Profit', line=dict(width=0), fill='tozeroy', fillcolor='rgba(144, 238, 144, 0.5)'))
    fig.add_trace(go.Scatter(x=price_range, y=pnl_neg, mode='lines', name='Loss', line=dict(width=0), fill='tozeroy', fillcolor='rgba(255, 182, 193, 0.5)'))
    fig.add_trace(go.Scatter(x=price_range, y=pnl, mode='lines', name='Total P&L', line=dict(color='darkblue', width=3)))
    
    fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=1)

    bench_pnl = (price_range - st.session_state.reference_price) * 100
    fig.add_trace(go.Scatter(x=price_range, y=bench_pnl, mode='lines', name='Share Only', line=dict(color='gray', dash='dot', width=1)))

    fig.update_layout(
        title="Profit / Loss at Expiration",
        xaxis_title="Stock Price",
        yaxis_title="P&L ($)",
        hovermode="x unified",
        height=700,  # Taller Chart
        font=dict(size=14), # Bigger Font
        margin=dict(l=20, r=20, t=40, b=20),
        yaxis=dict(autorange=True) # Auto-Scale Y Axis
    )
    
    st.plotly_chart(fig, use_container_width=True)
