import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from datetime import date, timedelta
import uuid

# --- Page Config ---
st.set_page_config(page_title="ASX Options Strategy Visualizer", layout="wide")

# --- Session State Initialization ---
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
        price = ticker.info.get('currentPrice') or ticker.info.get('previousClose')
        
        if price:
            st.session_state.reference_price = float(price)
            # Update Share legs
            for leg in st.session_state.legs:
                if leg['type'] == 'Share':
                    leg['price'] = float(price)
    except Exception as e:
        st.error(f"Error fetching price: {e}")

def create_leg(leg_type='Long Call'):
    """Creates a leg with a GUARANTEED unique ID to prevent crashes"""
    ref_price = float(st.session_state.reference_price)
    leg_id = str(uuid.uuid4()) # Unique ID
    
    new_leg = {
        'id': leg_id,
        'type': leg_type,
        'quantity': 100,
        'expiry': date.today() + timedelta(days=30),
    }
    
    if leg_type == 'Share':
        new_leg['price'] = ref_price
    else:
        # Options logic
        strike = ref_price
        new_leg['strike'] = float(strike)
        new_leg['strike_pct'] = 100.0
        new_leg['premium_per_opt'] = float(round(ref_price * 0.05, 2))
        
    return new_leg

def add_leg():
    st.session_state.legs.append(create_leg('Long Call'))

def remove_leg(leg_id):
    st.session_state.legs = [leg for leg in st.session_state.legs if leg['id'] != leg_id]

# --- Strategy Preset Functions ---
def set_strategy(strategy_name):
    st.session_state.legs = [] # Clear existing
    ref_price = float(st.session_state.reference_price)
    
    if strategy_name == "Bull Call Spread":
        l1 = create_leg('Long Call')
        l1['strike'] = ref_price
        l1['strike_pct'] = 100.0
        
        l2 = create_leg('Short Call')
        l2['strike'] = ref_price * 1.05
        l2['strike_pct'] = 105.0
        l2['premium_per_opt'] = ref_price * 0.02
        st.session_state.legs.extend([l1, l2])

    elif strategy_name == "Covered Call":
        l1 = create_leg('Share')
        
        l2 = create_leg('Short Call')
        l2['strike'] = ref_price * 1.05
        l2['strike_pct'] = 105.0
        l2['premium_per_opt'] = ref_price * 0.03
        st.session_state.legs.extend([l1, l2])

    elif strategy_name == "Protected Put":
        l1 = create_leg('Share')
        
        l2 = create_leg('Long Put')
        l2['strike'] = ref_price * 0.95
        l2['strike_pct'] = 95.0
        l2['premium_per_opt'] = ref_price * 0.03
        st.session_state.legs.extend([l1, l2])
        
    elif strategy_name == "Bear Put Spread":
        l1 = create_leg('Long Put')
        l1['strike'] = ref_price
        l1['strike_pct'] = 100.0
        
        l2 = create_leg('Short Put')
        l2['strike'] = ref_price * 0.95
        l2['strike_pct'] = 95.0
        l2['premium_per_opt'] = ref_price * 0.02
        st.session_state.legs.extend([l1, l2])

    elif strategy_name == "Long Straddle":
        l1 = create_leg('Long Call')
        l2 = create_leg('Long Put')
        st.session_state.legs.extend([l1, l2])
        
    elif strategy_name == "Collar":
        l1 = create_leg('Share')
        l2 = create_leg('Long Put') # Protective Put
        l2['strike'] = ref_price * 0.95
        l2['strike_pct'] = 95.0
        
        l3 = create_leg('Short Call') # Financing Call
        l3['strike'] = ref_price * 1.05
        l3['strike_pct'] = 105.0
        st.session_state.legs.extend([l1, l2, l3])

# --- Callbacks for Auto-Calc ---
def update_strike_from_pct(leg_id, key):
    # Find the leg
    for leg in st.session_state.legs:
        if leg['id'] == leg_id:
            new_pct = st.session_state[key]
            leg['strike'] = float(round(st.session_state.reference_price * (new_pct / 100.0), 2))
            break

def update_pct_from_strike(leg_id, key):
    # Find the leg
    for leg in st.session_state.legs:
        if leg['id'] == leg_id:
            new_strike = st.session_state[key]
            if st.session_state.reference_price > 0:
                leg['strike_pct'] = float(round((new_strike / st.session_state.reference_price) * 100.0, 1))
            break

# --- Calculation Engine ---
def calculate_payoff(price_range):
    total_pnl = np.zeros_like(price_range)
    
    for leg in st.session_state.legs:
        qty = leg['quantity']
        
        if leg['type'] == 'Share':
            entry_price = leg['price']
            pnl = (price_range - entry_price) * qty
            
        else: # Option leg
            strike = leg['strike']
            premium = leg['premium_per_opt']
            cost = premium * qty
            
            if 'Call' in leg['type']:
                intrinsic = np.maximum(price_range - strike, 0)
            else: # Put
                intrinsic = np.maximum(strike - price_range, 0)
                
            if 'Long' in leg['type']:
                pnl = (intrinsic * qty) - cost
            else: # Short
                pnl = cost - (intrinsic * qty)
                
        total_pnl += pnl
    return total_pnl

# ================= UI LAYOUT =================

with st.sidebar:
    st.header("1. Stock & Price")
    st.text_input("ASX Code", key='asx_code_input', value=st.session_state.asx_code)
    st.button("Fetch Price", on_click=fetch_price)
    
    st.number_input("Ref Price ($)", value=float(st.session_state.reference_price), key='reference_price', format="%.2f", step=0.1)

    st.markdown("---")
    st.header("2. Chart Axis")
    # Dynamic defaults based on current price
    ref = st.session_state.reference_price
    min_chart = st.number_input("Min X-Axis", value=float(round(ref*0.7, 2)), format="%.2f")
    max_chart = st.number_input("Max X-Axis", value=float(round(ref*1.3, 2)), format="%.2f")

st.title("ASX Options Visualizer")

# Strategy Buttons
st.markdown("### Quick Strategies")
c1, c2, c3, c4 = st.columns(4)
if c1.button("Bull Call Spread", use_container_width=True): set_strategy("Bull Call Spread")
if c2.button("Covered Call", use_container_width=True): set_strategy("Covered Call")
if c3.button("Protected Put", use_container_width=True): set_strategy("Protected Put")
if c4.button("Collar", use_container_width=True): set_strategy("Collar")

# Legs Section
st.markdown("### Strategy Builder")
if not st.session_state.legs:
    st.info("Click a strategy above or 'Add Leg' to start.")

for i, leg in enumerate(st.session_state.legs):
    # ID-based keys are crucial for stability
    lid = leg['id']
    
    st.markdown(f"**Leg {i+1}**")
    cols = st.columns([1.5, 1.2, 1.5, 1.5, 1.2, 1.2, 1.5, 0.8])
    
    # 1. Type
    new_type = cols[0].selectbox("Type", ["Share", "Long Call", "Short Call", "Long Put", "Short Put"], 
                                 index=["Share", "Long Call", "Short Call", "Long Put", "Short Put"].index(leg['type']), 
                                 key=f"type_{lid}")
    
    if new_type != leg['type']:
        leg['type'] = new_type
        st.rerun()

    # 2. Quantity
    qty = cols[1].number_input("Qty", value=int(leg['quantity']), step=100, key=f"qty_{lid}")
    leg['quantity'] = qty

    if leg['type'] == 'Share':
        price = cols[2].number_input("Price", value=float(leg.get('price', st.session_state.reference_price)), format="%.2f", key=f"s_price_{lid}")
        leg['price'] = price
        cols[3].metric("Value", f"${qty*price:,.0f}")
        
    else:
        # 3. Strike (Auto-Calc)
        strike_key = f"strike_{lid}"
        pct_key = f"pct_{lid}"
        
        cols[2].number_input("Strike ($)", value=float(leg['strike']), step=0.5, format="%.2f", 
                           key=strike_key, on_change=update_pct_from_strike, args=(lid, strike_key))
        
        cols[3].number_input("Strike %", value=float(leg['strike_pct']), step=1.0, format="%.1f", 
                           key=pct_key, on_change=update_strike_from_pct, args=(lid, pct_key))

        # 4. Premium
        prem = cols[4].number_input("Prem ($)", value=float(leg['premium_per_opt']), step=0.05, format="%.2f", key=f"prem_{lid}")
        leg['premium_per_opt'] = prem
        
        cols[5].metric("Total", f"${qty*prem:,.0f}")
        
        # 5. Expiry
        cols[6].date_input("Expiry", value=leg['expiry'], key=f"exp_{lid}")

    # Remove Button
    if cols[7].button("X", key=f"del_{lid}"):
        remove_leg(lid)
        st.rerun()

st.button("+ Add Leg", on_click=add_leg)

# --- Visualization ---
st.markdown("---")
if st.session_state.legs:
    price_range = np.linspace(min_chart, max_chart, 200)
    pnl = calculate_payoff(price_range)
    
    fig = go.Figure()
    
    # Strategy P&L
    fig.add_trace(go.Scatter(x=price_range, y=pnl, mode='lines', name='Strategy', line=dict(color='blue', width=3), fill='tozeroy'))
    
    # Benchmark (Long Share Only)
    benchmark_pnl = (price_range - st.session_state.reference_price) * 100
    fig.add_trace(go.Scatter(x=price_range, y=benchmark_pnl, mode='lines', name='Share Only', line=dict(color='gray', dash='dash')))
    
    fig.update_layout(title="Payoff at Expiration", xaxis_title="Stock Price", yaxis_title="Profit/Loss", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)
