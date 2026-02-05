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

LEG_TYPES = ["Long Share", "Long Call", "Short Call", "Long Put", "Short Put"]

# --- Strategy Definitions ---
def get_strat_legs(name, ref=100.0):
    legs = []
    # Note: Using 100 qty as standard template base
    if name == "Bull Call Spread":
        legs.append({'type': 'Long Call', 'strike': ref, 'quantity': 100, 'premium': ref*0.05})
        legs.append({'type': 'Short Call', 'strike': ref*1.1, 'quantity': 100, 'premium': ref*0.02})
    elif name == "Covered Call":
        legs.append({'type': 'Long Share', 'price': ref, 'quantity': 100})
        legs.append({'type': 'Short Call', 'strike': ref*1.1, 'quantity': 100, 'premium': ref*0.02})
    elif name == "Protected Put":
        legs.append({'type': 'Long Share', 'price': ref, 'quantity': 100})
        legs.append({'type': 'Long Put', 'strike': ref*0.9, 'quantity': 100, 'premium': ref*0.03})
    elif name == "Bear Put Spread":
        legs.append({'type': 'Long Put', 'strike': ref, 'quantity': 100, 'premium': ref*0.05})
        legs.append({'type': 'Short Put', 'strike': ref*0.9, 'quantity': 100, 'premium': ref*0.02})
    elif name == "Bear Call Spread":
        legs.append({'type': 'Short Call', 'strike': ref, 'quantity': 100, 'premium': ref*0.05})
        legs.append({'type': 'Long Call', 'strike': ref*1.1, 'quantity': 100, 'premium': ref*0.01})
    elif name == "Long Straddle":
        legs.append({'type': 'Long Call', 'strike': ref, 'quantity': 100, 'premium': ref*0.05})
        legs.append({'type': 'Long Put', 'strike': ref, 'quantity': 100, 'premium': ref*0.05})
    elif name == "Collar":
        legs.append({'type': 'Long Share', 'price': ref, 'quantity': 100})
        legs.append({'type': 'Long Put', 'strike': ref*0.9, 'quantity': 100, 'premium': ref*0.03})
        legs.append({'type': 'Short Call', 'strike': ref*1.1, 'quantity': 100, 'premium': ref*0.02})
    return legs

STRATEGY_INFO = {
    "Bull Call Spread": {"mech": "Buy ATM Call + Sell OTM Call", "use": "Moderately Bullish. Limited profit, limited risk."},
    "Covered Call": {"mech": "Own Share + Sell OTM Call", "use": "Neutral/Bullish. Generate income from shares you own."},
    "Protected Put": {"mech": "Own Share + Buy OTM Put", "use": "Bullish but Cautious. Insurance against a crash."},
    "Bear Put Spread": {"mech": "Buy ATM Put + Sell OTM Put", "use": "Moderately Bearish. Cheaper than buying a straight Put."},
    "Bear Call Spread": {"mech": "Sell ATM Call + Buy OTM Call", "use": "Bearish. Profit from time decay."},
    "Long Straddle": {"mech": "Buy Call + Buy Put (Same Strike)", "use": "High Volatility. Expect a big move in EITHER direction."},
    "Collar": {"mech": "Share + Buy Put + Sell Call", "use": "Conservative. 'Free' insurance funded by selling upside."},
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
                if leg['type'] == 'Long Share':
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
    if leg_type == 'Long Share':
        new_leg['price'] = ref_price
        new_leg['share_val'] = ref_price * 100
    else:
        new_leg['strike'] = ref_price
        new_leg['strike_pct'] = 100.0
        new_leg['premium_per_opt'] = float(round(ref_price * 0.05, 2))
        new_leg['total_prem'] = new_leg['premium_per_opt'] * 100
    return new_leg

def add_leg():
    st.session_state.legs.append(create_leg('Long Call'))

def remove_leg(leg_id):
    st.session_state.legs = [leg for leg in st.session_state.legs if leg['id'] != leg_id]

def set_strategy(strategy_name):
    st.session_state.legs = []
    ref = float(st.session_state.reference_price)
    templates = get_strat_legs(strategy_name, ref)
    
    for t in templates:
        l = create_leg(t['type'])
        l['quantity'] = t['quantity']
        if t['type'] == 'Long Share':
            l['price'] = t['price']
            l['share_val'] = t['price'] * t['quantity']
        else:
            l['strike'] = t['strike']
            l['strike_pct'] = round((t['strike']/ref)*100, 1)
            l['premium_per_opt'] = float(round(t['premium'], 2))
            l['total_prem'] = l['premium_per_opt'] * l['quantity']
        st.session_state.legs.append(l)

# --- Calculation Logic ---
def calc_pnl_for_legs(legs, price_range):
    total = np.zeros_like(price_range)
    for leg in legs:
        qty = leg.get('quantity', 100)
        if leg['type'] == 'Long Share':
            pnl = (price_range - leg.get('price', 100)) * qty
        else:
            strike = leg.get('strike', 100)
            prem = leg.get('premium_per_opt', 0)
            cost = prem * qty
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

# --- Callbacks (Sync Logic) ---
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

def on_share_qty_change(lid, qty_key, val_key):
    new_qty = st.session_state[qty_key]
    for leg in st.session_state.legs:
        if leg['id'] == lid:
            leg['quantity'] = new_qty
            new_val = new_qty * leg['price']
            leg['share_val'] = new_val
            st.session_state[val_key] = new_val
            break

def on_share_val_change(lid, qty_key, val_key):
    new_val = st.session_state[val_key]
    for leg in st.session_state.legs:
        if leg['id'] == lid:
            leg['share_val'] = new_val
            if leg['price'] > 0:
                new_qty = int(new_val / leg['price'])
                leg['quantity'] = new_qty
                st.session_state[qty_key] = new_qty
            break

# Premium Sync Callbacks
def on_prem_unit_change(lid, qty_key, unit_key, total_key):
    new_unit = st.session_state[unit_key]
    qty = st.session_state[qty_key]
    for leg in st.session_state.legs:
        if leg['id'] == lid:
            leg['premium_per_opt'] = new_unit
            new_total = new_unit * qty
            leg['total_prem'] = new_total
            st.session_state[total_key] = new_total
            break

def on_prem_total_change(lid, qty_key, unit_key, total_key):
    new_total = st.session_state[total_key]
    qty = st.session_state[qty_key]
    for leg in st.session_state.legs:
        if leg['id'] == lid:
            leg['total_prem'] = new_total
            if qty > 0:
                new_unit = new_total / qty
                leg['premium_per_opt'] = new_unit
                st.session_state[unit_key] = new_unit
            break

# --- Mini Chart Generator ---
def render_mini_chart(name):
    dummy_ref = 100
    dummy_legs = get_strat_legs(name, dummy_ref)
    rng = np.linspace(70, 130, 100)
    pnl = calc_pnl_for_legs(dummy_legs, rng)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rng, y=pnl, mode='lines', line=dict(color='darkblue', width=2)))
    fig.add_hline(y=0, line_color="black", line_width=1)
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=150, 
        xaxis=dict(visible=False), 
        yaxis=dict(visible=False),
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(240,240,240,0.5)',
        hovermode=False # Disable hover on mini chart
    )
    return fig

# --- UI Components ---
def render_strat_btn(name):
    c1, c2 = st.columns([5, 1])
    if c1.button(name, use_container_width=True):
        set_strategy(name)
    
    info = STRATEGY_INFO.get(name, {})
    with c2.popover("?", use_container_width=True):
        st.subheader(name)
        st.write(f"**Mechanics:** {info.get('mech','')}")
        st.write(f"**When to use?** {info.get('use','')}")
        st.markdown("---")
        st.write("**Typical Payoff Shape:**")
        st.plotly_chart(render_mini_chart(name), use_container_width=True, config={'displayModeBar': False})

# ================= MAIN UI =================
with st.sidebar:
    st.header("1. Stock Setup")
    st.text_input("ASX Code", key='asx_code_input', value=st.session_state.asx_code)
    st.button("Fetch Price", on_click=fetch_price)
    st.number_input("Ref Price ($)", value=float(st.session_state.reference_price), key='reference_price', format="%.2f", step=0.1)

    st.markdown("---")
    st.header("2. Chart Zoom")
    ref = st.session_state.reference_price
    c1, c2 = st.columns(2)
    min_chart = c1.number_input("Min X", value=float(round(ref*0.7, 2)), format="%.2f")
    max_chart = c2.number_input("Max X", value=float(round(ref*1.3, 2)), format="%.2f")

st.title("ASX Options Visualizer")

st.markdown("### Quick Strategies")
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

for i, leg in enumerate(st.session_state.legs):
    lid = leg['id']
    st.markdown(f"**Leg {i+1}**")
    cols = st.columns([1.5, 1.2, 1.5, 1.5, 1.5, 1.5, 1.5, 0.5])
    
    # 1. Type (Updated list)
    new_type = cols[0].selectbox("Type", LEG_TYPES, 
                                 index=LEG_TYPES.index(leg['type']), 
                                 key=f"type_{lid}")
    if new_type != leg['type']:
        leg['type'] = new_type
        st.rerun()

    if leg['type'] == 'Long Share':
        # Share Sync Logic
        q_key = f"qty_{lid}"
        v_key = f"val_{lid}"
        
        qty = cols[1].number_input("Quantity", value=int(leg['quantity']), step=100, key=q_key, 
                                 on_change=on_share_qty_change, args=(lid, q_key, v_key))
        
        price = cols[2].number_input("Price ($)", value=float(leg['price']), format="%.2f", key=f"s_price_{lid}")
        leg['price'] = price
        
        if 'share_val' not in leg: leg['share_val'] = qty * price
        
        cols[3].number_input("Value ($)", value=float(leg['share_val']), step=1000.0, format="%.2f", key=v_key,
                           on_change=on_share_val_change, args=(lid, q_key, v_key))
        
    else:
        # Option Logic
        q_key = f"qty_{lid}"
        qty = cols[1].number_input("Qty", value=int(leg['quantity']), step=100, key=q_key)
        leg['quantity'] = qty
        
        s_key, p_key = f"strike_{lid}", f"pct_{lid}"
        cols[2].number_input("Strike ($)", value=float(leg['strike']), step=0.5, format="%.2f", key=s_key, on_change=on_strike_change, args=(lid, s_key, p_key))
        cols[3].number_input("Strike (%)", value=float(leg['strike_pct']), step=1.0, format="%.1f", key=p_key, on_change=on_pct_change, args=(lid, s_key, p_key))
        
        # Bi-Directional Premium Sync
        prem_unit_key = f"prem_{lid}"
        prem_total_key = f"prem_tot_{lid}"
        if 'total_prem' not in leg: leg['total_prem'] = leg['premium_per_opt'] * qty
        
        prem = cols[4].number_input("Prem ($) per Opt", value=float(leg['premium_per_opt']), step=0.05, format="%.2f", 
                                  key=prem_unit_key, on_change=on_prem_unit_change, args=(lid, q_key, prem_unit_key, prem_total_key))
        
        cols[5].number_input("Total Prem ($)", value=float(leg['total_prem']), step=10.0, format="%.2f", 
                           key=prem_total_key, on_change=on_prem_total_change, args=(lid, q_key, prem_unit_key, prem_total_key))
        
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
    total_pnl = calc_pnl_for_legs(st.session_state.legs, price_range)
    
    fig = go.Figure()

    # 1. Shading (Total P&L) - hoverinfo skip to hide from tooltip
    pnl_pos = np.where(total_pnl >= 0, total_pnl, 0)
    pnl_neg = np.where(total_pnl < 0, total_pnl, 0)
    fig.add_trace(go.Scatter(x=price_range, y=pnl_pos, mode='lines', line=dict(width=0), fill='tozeroy', fillcolor='rgba(144, 238, 144, 0.5)', showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=price_range, y=pnl_neg, mode='lines', line=dict(width=0), fill='tozeroy', fillcolor='rgba(255, 182, 193, 0.5)', showlegend=False, hoverinfo='skip'))
    
    # 2. Individual Leg Traces
    colors = ['#FF9999', '#99FF99', '#9999FF', '#FFFF99', '#FFCC99']
    for i, leg in enumerate(st.session_state.legs):
        single_leg_pnl = calc_pnl_for_legs([leg], price_range)
        fig.add_trace(go.Scatter(
            x=price_range, y=single_leg_pnl,
            mode='lines',
            name=f"Leg {i+1}: {leg['type']}",
            line=dict(width=1.5, color=colors[i % len(colors)], dash='dashdot'),
            hovertemplate=f"<b>{leg['type']} P&L: $ %{{y:,.0f}}</b><extra></extra>"
        ))

    # 3. Main Total P&L Line
    # Note: Removed "Stock Price: $..." from here because it's now in the X-Axis Header
    fig.add_trace(go.Scatter(
        x=price_range, y=total_pnl, mode='lines', name='Total P&L',
        line=dict(color='darkblue', width=3),
        hovertemplate="<b>Total P&L: $%{y:,.0f}</b><extra></extra>"
    ))

    # 4. CONDITIONAL Benchmark (Only show if 'Long Share' exists)
    has_share = any(leg['type'] == 'Long Share' for leg in st.session_state.legs)
    if has_share:
        share_qty = sum(leg['quantity'] for leg in st.session_state.legs if leg['type'] == 'Long Share')
        bench_pnl = (price_range - st.session_state.reference_price) * share_qty
        fig.add_trace(go.Scatter(
            x=price_range, y=bench_pnl,
            mode='lines', name=f'Only Share P&L',
            line=dict(color='gray', dash='dot', width=2),
            hovertemplate="Only Share P&L: $%{y:,.0f}<extra></extra>"
        ))

    fig.add_hline(y=0, line_color="black", line_width=1)
    
    # Annotations
    signs = np.sign(total_pnl)
    flips = np.where(np.diff(signs))[0]
    be_points = []
    for f in flips:
        x1, x2 = price_range[f], price_range[f+1]
        y1, y2 = total_pnl[f], total_pnl[f+1]
        if y2 != y1:
            x_zero = x1 - y1 * (x2 - x1) / (y2 - y1)
            be_points.append(x_zero)
            
    max_p = np.max(total_pnl)
    max_l = np.min(total_pnl)
    max_x = price_range[int(np.median(np.where(total_pnl == max_p)[0]))]
    min_x = price_range[int(np.median(np.where(total_pnl == max_l)[0]))]

    annotations = []
    for be in be_points:
        annotations.append(dict(x=be, y=0, xref="x", yref="y", text=f"Break-Even: ${be:.2f}", showarrow=True, arrowhead=2, ax=0, ay=-40, bgcolor="white", bordercolor="black"))
    if max_p < 1e6:
        annotations.append(dict(x=max_x, y=max_p, xref="x", yref="y", text=f"Max Profit: ${max_p:,.0f}", showarrow=True, arrowhead=2, ax=0, ay=-40, bgcolor="#e6ffe6", bordercolor="green"))
    if max_l > -1e6:
        annotations.append(dict(x=min_x, y=max_l, xref="x", yref="y", text=f"Max Loss: ${max_l:,.0f}", showarrow=True, arrowhead=2, ax=0, ay=40, bgcolor="#ffe6e6", bordercolor="red"))

    fig.update_layout(
        title="Profit / Loss at Expiration",
        xaxis=dict(
            title="Stock Price at Expiration ($)", 
            hoverformat="$.2f" # This puts the price in the Header!
        ),
        yaxis_title="P&L ($)",
        hovermode="x unified",
        height=700,
        font=dict(size=14),
        yaxis=dict(autorange=True),
        annotations=annotations,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)
