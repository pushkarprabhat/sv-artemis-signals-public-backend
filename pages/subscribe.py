import streamlit as st
import razorpay
import time
from config import PROJECT_NAME, COMMERCIAL_MODE, RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET, RAZORPAY_PLAN_MONTHLY, RAZORPAY_PLAN_ANNUAL
from utils.logger import logger

# ============================================================================
# PAGE SETUP
# ============================================================================
st.set_page_config(page_title=f"Subscribe - {PROJECT_NAME}", layout="wide", initial_sidebar_state="expanded")

# ============================================================================
# STYLING
# ============================================================================
st.markdown("""
<style>
    .main {background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);}
    h1, h2, h3 {font-family: 'Georgia', serif; color: gold; text-align: center;}
    .plan-card {
        background: rgba(255, 215, 0, 0.05);
        border: 2px solid gold;
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        transition: transform 0.3s;
    }
    .plan-card:hover {
        transform: scale(1.02);
        background: rgba(255, 215, 0, 0.1);
    }
    .price {
        font-size: 36px;
        font-weight: bold;
        color: cyan;
        margin: 20px 0;
    }
    .feature {
        margin: 10px 0;
        color: #e0e0e0;
    }
    .status-comment {
        text-align: center;
        font-style: italic;
        color: gold;
        margin-top: 50px;
        font-size: 18px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
if "subscribed" not in st.session_state:
    st.session_state.subscribed = False

# ============================================================================
# RAZORPAY INTEGRATION
# ============================================================================
try:
    client = razorpay.Client(auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET))
except Exception as e:
    logger.error(f"Razorpay Client initialization failed: {e}")
    client = None

def create_razorpay_order(amount_in_inr):
    try:
        # Amount is in paise
        data = {
            "amount": int(amount_in_inr) * 100,
            "currency": "INR",
            "receipt": f"receipt_{int(time.time())}",
            "payment_capture": 1
        }
        order = client.order.create(data=data)
        return order
    except Exception as e:
        st.error(f"Failed to create order: {e}")
        return None

# ============================================================================
# HEADER
# ============================================================================
st.markdown(f"<h1>💎 {PROJECT_NAME} Premium</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: cyan;'>Institutional-Grade Systematic Signals</h4>", unsafe_allow_html=True)

# ============================================================================
# SUBSCRIPTION PLANS
# ============================================================================
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    st.markdown(f"""
    <div class='plan-card'>
        <h3>COMMERCIAL MONTHLY</h3>
        <div class='price'>₹{RAZORPAY_PLAN_MONTHLY} <span>/ month</span></div>
        <div class='feature'>✅ Multi-Strategy Pair Signals</div>
        <div class='feature'>✅ Real-time Options Greeks</div>
        <div class='feature'>✅ GARCH Volatility Modeling</div>
        <div class='feature'>✅ Telegram & Email Alerts</div>
        <div class='feature'>✅ Standard Risk Analytics</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button(f"Subscribe Monthly (₹{RAZORPAY_PLAN_MONTHLY})", use_container_width=True, key="btn_monthly"):
        if RAZORPAY_KEY_ID == "rzp_test_placeholder":
            st.warning("⚠️ Sandbox Mode: Simulating payment...")
            with st.spinner("Processing..."):
                time.sleep(1.5)
                st.session_state.subscribed = True
                st.success("✅ Commercial License Activated! Access Granted.")
                st.balloons()
        else:
            order = create_razorpay_order(RAZORPAY_PLAN_MONTHLY)
            if order:
                st.info(f"Order created: {order['id']}. Follow instructions in browser...")
                # Real implementation would use razorpay checkout JS
                if st.button("Simulate Verification", key="verify_monthly"):
                    st.session_state.subscribed = True
                    st.success("✅ Payment Verified!")

with col2:
    st.markdown(f"""
    <div class='plan-card'>
        <h3>ENTERPRISE ANNUAL</h3>
        <div class='price'>₹{RAZORPAY_PLAN_ANNUAL} <span>/ year</span></div>
        <div class='feature'>✅ Everything in Commercial Monthly</div>
        <div class='feature'>✅ <b>Unlimited</b> Backtesting Access</div>
        <div class='feature'>✅ Priority Data Feed Integration</div>
        <div class='feature'>✅ Dedicated Support Channel</div>
        <div class='feature'>✅ 16% Discount Included</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button(f"Subscribe Annual (₹{RAZORPAY_PLAN_ANNUAL})", use_container_width=True, key="btn_annual"):
        if RAZORPAY_KEY_ID == "rzp_test_placeholder":
            st.warning("⚠️ Sandbox Mode: Simulating payment...")
            with st.spinner("Processing Annual Plan..."):
                time.sleep(2)
                st.session_state.subscribed = True
                st.success("✅ Enterprise License Activated! Welcome aboard.")
                st.balloons()
                st.snow()
        else:
            order = create_razorpay_order(RAZORPAY_PLAN_ANNUAL)
            if order:
                st.info(f"Order created: {order['id']}. Follow instructions in browser...")

# ============================================================================
# SUBSCRIPTION STATUS
# ============================================================================
st.markdown("---")
if st.session_state.get('subscribed', False):
    st.success("🌟 **ACTIVE LICENSE**: You have full access to Artemis Signals Premium.")
    if st.button("Return to Dashboard", use_container_width=True):
        st.switch_page("main.py")
else:
    st.info("💡 Note: Subscribing unlocks real-time scanning across 65+ focused instruments.")

# ============================================================================
# FOOTER
# ============================================================================
st.markdown(f"<div class='status-comment'>\"This subscription powers the future of trading excellence\"</div>", unsafe_allow_html=True)

if not COMMERCIAL_MODE:
    st.markdown("""
    <div style='text-align: center; color: #888; font-size: 14px; margin-top: 20px;'>
        Family Mission: Building the future through disciplined quant trading.
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div style='text-align: center; color: #888; font-size: 14px; margin-top: 20px;'>
        Institutional Grade Infrastructure — Systematic Execution — Precision Quantitative Analysis
    </div>
    """, unsafe_allow_html=True)
