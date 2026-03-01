import streamlit as st
import pandas as pd
import plotly.express as px

# --- Page Configuration ---
st.set_page_config(page_title="Fraud Detection Dashboard", page_icon="🛡️", layout="wide")

# --- Initialize Session State for Navigation ---
if "current_view" not in st.session_state:
    st.session_state.current_view = "main_table"
if "selected_node_id" not in st.session_state:
    st.session_state.selected_node_id = None

# --- Data Loading & Logic ---
@st.cache_data
def load_data():
    df = pd.read_csv("output_layer/b2b_bi_master.csv")
    
    # Contextual Anomaly Logic
    def get_anomaly_type(row):
        before = row['predicted_fraud_before'] == 1
        after = row['predicted_fraud_after'] == 1
        
        if before and after:
            return "Real-time & Financial Anomalies"
        elif before:
            return "Real-time infrastructure & session anomalies"
        elif after:
            return "Financial & coordinated outcome anomalies"
        else:
            return "Clear"
            
    df['anomaly_type'] = df.apply(get_anomaly_type, axis=1)
    
    # Calculate Relative Scores (0-100%) based on the maximum risk in the dataset
    max_b_score = df["max_score_before"].max()
    max_a_score = df["max_score_after"].max()
    
    df["relative_score_before"] = (df["max_score_before"] / max_b_score) * 100
    df["relative_score_after"] = (df["max_score_after"] / max_a_score) * 100
    
    # Filter ONLY to flagged bookings for the table queue
    flagged_df = df[df['anomaly_type'] != "Clear"].copy()
    
    # Priority Sorting: Descending by relative_score_before
    flagged_df = flagged_df.sort_values(by="relative_score_before", ascending=False).reset_index(drop=True)
    
    return df, flagged_df

raw_df, flagged_df = load_data()

# --- Pre-calculate Global Metrics ---
total_analyzed = len(raw_df)
total_flagged = len(flagged_df)
value_at_risk = flagged_df["booking_value"].sum()
true_frauds_caught = len(flagged_df[flagged_df["fraud_label"] == 1])

# ==========================================
# VIEW 1: MAIN TABLE DASHBOARD
# ==========================================
if st.session_state.current_view == "main_table":
    
    # --- HEADER WITH LOGO ---
    col_title, col_logo = st.columns([5, 1])
    with col_title:
        st.title("🛡️ Fraud Detection Dashboard")
        st.markdown("Unsupervised B2B Travel Fraud Detection via DONE_AdONE")
    with col_logo:
        # Replace the URL below with your local file path, e.g., st.image("logo.png", width=120)
        st.image("Sentinel Logo.png", width=120) 
        
    st.markdown("---")
    
    # Top Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        with st.container(border=True):
            st.metric("Total Bookings Analyzed", f"{total_analyzed:,}")
    with col2:
        with st.container(border=True):
            st.metric("Total Flagged Alerts", f"{total_flagged:,}")
    with col3:
        with st.container(border=True):
            st.metric("Value at Risk (Flagged)", f"${value_at_risk:,.2f}")
    with col4:
        with st.container(border=True):
            st.metric("Known Frauds Caught", f"{true_frauds_caught:,}", help="Verified offline labels")
            
    # Business-Contextual Charts
    st.markdown("---")
    col_chart1, col_chart2 = st.columns([1, 1.5])
    
    # Color palette including the subtle gray for "Clear" bookings
    anomaly_colors_full = {
        "Clear": "#1F2937",                                       # Subtle Dark Gray
        "Real-time & Financial Anomalies": "#D32F2F",             # Red (Severe)
        "Real-time infrastructure & session anomalies": "#F59E0B",# Amber (Session)
        "Financial & coordinated outcome anomalies": "#8B5CF6"    # Purple (Financial)
    }

    with col_chart1:
        st.write("**Financial Exposure by Anomaly Type**")
        fig1 = px.pie(
            flagged_df, 
            values="booking_value", 
            names="anomaly_type", 
            hole=0.4,
            template="plotly_dark",
            color="anomaly_type",
            color_discrete_map=anomaly_colors_full
        )
        # Put the legend at the bottom so it doesn't squash the donut
        fig1.update_layout(
            margin=dict(l=0, r=0, t=20, b=0), 
            height=350,
            legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig1, use_container_width=True)
        
    with col_chart2:
        st.write("**Relative Risk Evolution: Session vs Financial (0-100%)**")
        
        # Sort so "Clear" points are plotted first (in the back), and flags are on top
        df_plot = raw_df.copy()
        df_plot['is_flagged'] = df_plot['anomaly_type'] != "Clear"
        df_plot = df_plot.sort_values(by="is_flagged")
        
        # Scatterplot USING RELATIVE SCORES (0-100)
        fig2 = px.scatter(
            df_plot, 
            x="relative_score_before", 
            y="relative_score_after", 
            color="anomaly_type",
            opacity=0.75,
            hover_name="booking_id",
            template="plotly_dark",
            color_discrete_map=anomaly_colors_full,
            marginal_x="histogram",  
            marginal_y="histogram",  
            labels={
                "relative_score_before": "Relative Session Risk (%)",
                "relative_score_after": "Relative Financial Risk (%)"
            }
        )
        
        # Diagonal line showing the 1:1 risk ratio
        fig2.add_shape(type="line", x0=0, y0=0, x1=100, y1=100, line=dict(color="#4B5563", dash="dash"))
        
        fig2.update_layout(
            margin=dict(l=0, r=0, t=20, b=0), 
            height=350, 
            showlegend=False,
            xaxis=dict(range=[0, 105]), # Lock axes to strictly show 0-100% + slight padding
            yaxis=dict(range=[0, 105])
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.subheader("🚨 Priority Investigation Queue")
    st.caption("Showing ONLY flagged bookings, prioritized by Real-Time Session Risk.")

    # Prepare Display Table
    display_df = flagged_df[["booking_id", "anomaly_type", "relative_score_before", "booking_value", "fraud_label"]].copy()
    display_df["fraud_label"] = display_df["fraud_label"].apply(lambda x: "🚨 Known Fraud" if x == 1 else "Unlabeled")

    # Render Interactive Table
    event = st.dataframe(
        display_df,
        column_config={
            "booking_id": "Booking ID",
            "anomaly_type": "Anomaly Flag Reason",
            "relative_score_before": st.column_config.ProgressColumn("Relative Session Risk", format="%.1f%%", min_value=0, max_value=100),
            "booking_value": st.column_config.NumberColumn("Value", format="$%.2f"),
            "fraud_label": "Ground Truth"
        },
        column_order=["booking_id", "relative_score_before", "anomaly_type", "booking_value", "fraud_label"],
        hide_index=True,
        use_container_width=True,
        on_select="rerun",
        selection_mode="single-row"
    )

    if len(event.selection.rows) > 0:
        selected_index = event.selection.rows[0]
        st.session_state.selected_node_id = display_df.iloc[selected_index]["booking_id"]
        st.session_state.current_view = "detail_page"
        st.rerun()


# ==========================================
# VIEW 2: DETAILED SUBPAGE
# ==========================================
elif st.session_state.current_view == "detail_page":
    
    # --- HEADER WITH LOGO ---
    col_back, col_empty, col_logo2 = st.columns([2, 3, 1])
    with col_back:
        if st.button("⬅️ Back to Dashboard"):
            st.session_state.current_view = "main_table"
            st.session_state.selected_node_id = None
            st.rerun()
    with col_logo2:
        # Same here, replace with your logo file path
        st.image("Sentinel Logo.png", width=120)
        
    node_data = flagged_df[flagged_df["booking_id"] == st.session_state.selected_node_id].iloc[0]
    
    st.title(f"Investigation: {node_data['booking_id']}")
    
    # Dynamic header color based on anomaly type
    if "Real-time & Financial" in node_data['anomaly_type']:
        st.error(f"**High Priority Alert:** {node_data['anomaly_type']}")
    elif "Real-time" in node_data['anomaly_type']:
        st.warning(f"**Session Alert:** {node_data['anomaly_type']}")
    else:
        st.info(f"**Financial Alert:** {node_data['anomaly_type']}")
        
    st.markdown("---")
    
    # Top Row: 3 Beautiful Cards
    col_booking, col_graph, col_risk = st.columns(3)
    
    with col_booking:
        with st.container(border=True):
            st.subheader("🛒 Booking Context")
            st.markdown(f"**Timestamp:** `{node_data['booking_ts']}`")
            st.markdown(f"**Product:** `{str(node_data['product_type']).title()}`")
            st.markdown(f"**Value:** `${node_data['booking_value']:.2f}`")
            st.markdown(f"**Status:** `{str(node_data['booking_status']).title()}`")

    with col_graph:
        with st.container(border=True):
            st.subheader("🕸️ Graph Edges (Entities)")
            st.markdown(f"🏢 **Agency ID:** `{node_data['agency_id']}`")
            st.markdown(f"👤 **User ID:** `{node_data['user_id']}`")
            st.markdown(f"💻 **Device:** `{str(node_data['device_fingerprint'])[:10]}...`")
            st.markdown(f"🌐 **IP Address:** `{node_data['ip_address']}`")

    with col_risk:
        with st.container(border=True):
            st.subheader("⚠️ Behavioral Risk Factors")
            vpn_status = "🔴 Detected" if node_data["is_vpn_or_proxy"] == 1 else "🟢 Clean"
            st.markdown(f"**VPN/Proxy:** {vpn_status}")
            device_switch = "🔴 Yes" if node_data["device_switch_flag"] == 1 else "🟢 No"
            st.markdown(f"**Device Switch:** {device_switch}")
            st.markdown(f"**Failed Login Ratio:** `{node_data['failed_login_ratio']:.2f}`")

    # Bottom Row: Model Output Context
    st.markdown("---")
    st.markdown("### 🧠 Model Scoring Breakdown (Relative %)")
    
    col_before, col_after = st.columns(2)
    
    with col_before:
        with st.container(border=True):
            st.subheader("Phase 1: Real-Time / Session Risk")
            st.metric("Relative Risk Score", f"{node_data['relative_score_before']:.1f}%")
            if node_data['predicted_fraud_before'] == 1:
                st.error(f"Flag triggered. Raw deep graph loss: {node_data['max_score_before']:.5f}")
            else:
                st.success(f"Passed. Raw deep graph loss: {node_data['max_score_before']:.5f}")
                
    with col_after:
        with st.container(border=True):
            st.subheader("Phase 2: Financial / Coordinated Risk")
            score_delta = node_data['relative_score_after'] - node_data['relative_score_before']
            st.metric("Relative Risk Score", f"{node_data['relative_score_after']:.1f}%", delta=f"{score_delta:.1f}% shift")
            
            if node_data['predicted_fraud_after'] == 1:
                st.error(f"Flag triggered. Raw deep graph loss: {node_data['max_score_after']:.5f}")
            else:
                st.success(f"Passed. Raw deep graph loss: {node_data['max_score_after']:.5f}")
        
    st.markdown("---")
    if node_data["fraud_label"] == 1:
        st.error(f"**Historical Ground Truth:** Confirmed as **{node_data['fraud_reason']}**")
    else:
        st.info("No offline fraud labels found. (Unsupervised Discovery)")