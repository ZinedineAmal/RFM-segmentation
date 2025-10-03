# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ---------- Config ----------
st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide", initial_sidebar_state="expanded")

# ---------- Header with image ----------
st.image(
    "https://images.pexels.com/photos/5632407/pexels-photo-5632407.jpeg",
    use_container_width=True
)
st.title("🛒 Customer Segmentation Dashboard")
st.markdown("Analyze customer behaviors, sales, and profitability using RFM segmentation.")

# ---------- Helpers ----------
@st.cache_data
def load_data(path):
    df = pd.read_csv(path, low_memory=False)
    if 'order_date' in df.columns:
        df['order_date'] = pd.to_datetime(df['order_date'])
    for c in ['sales', 'profit', 'Recency', 'Frequency', 'Monetary', 'RFM_Score']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

def agg_by_segment(df):
    seg = df.groupby('Segmentasi', dropna=False).agg(
        total_sales=('sales','sum'),
        total_profit=('profit','sum'),
        customers=('customer_id', lambda x: x.nunique()),
        count_orders=('order_id','count')
    ).reset_index().sort_values('total_sales', ascending=False)
    return seg

# ---------- Load ----------
DATA_PATH = "rfm_table.csv"
df = load_data(DATA_PATH)

# ---------- Sidebar Filters ----------
st.sidebar.header("Filters")
if 'order_date' in df.columns:
    min_date, max_date = df['order_date'].min(), df['order_date'].max()
    date_range = st.sidebar.date_input("Order date range", value=(min_date.date(), max_date.date()))
    start_dt, end_dt = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    df = df[(df['order_date'] >= start_dt) & (df['order_date'] <= end_dt)]

segments = df['Segmentasi'].dropna().unique().tolist()
selected_segments = st.sidebar.multiselect("Segmentation", options=segments, default=segments)
if selected_segments:
    df = df[df['Segmentasi'].isin(selected_segments)]

# ---------- Top KPIs ----------
total_sales = df['sales'].sum()
total_profit = df['profit'].sum()
total_customers = df['customer_id'].nunique()

k1, k2, k3 = st.columns([1.2,1,1])
with k1:
    st.markdown("### 💰 Total Sales")
    st.markdown(f"# ${total_sales:,.0f}")
with k2:
    st.markdown("### 👥 Total Customers")
    st.markdown(f"# {total_customers:,}")
with k3:
    st.markdown("### 🟢 Total Profit")
    st.markdown(f"# ${total_profit:,.0f}")

st.markdown("---")

# ---------- Row: Segmentation Pie, Sales/Profit by Segment, Daily ----------
col1, col2, col3 = st.columns([1,1.2,1])

with col1:
    st.subheader("Customers by Segmentation")
    seg_count = df.groupby('Segmentasi')['customer_id'].nunique().reset_index().rename(columns={'customer_id':'unique_customers'})
    fig_pie = px.pie(seg_count, names='Segmentasi', values='unique_customers', hole=0.3)
    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    st.subheader("Sales and Profit by Segmentation")
    seg = agg_by_segment(df)
    if not seg.empty:
        fig = go.Figure()
        fig.add_trace(go.Bar(y=seg['Segmentasi'], x=seg['total_sales'], orientation='h', name='Total Sales', marker=dict(color='cornflowerblue')))
        fig.add_trace(go.Bar(y=seg['Segmentasi'], x=seg['total_profit'], orientation='h', name='Total Profit', marker=dict(color='orange')))
        fig.update_layout(barmode='group', xaxis_title='USD', yaxis_title='', height=350)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data available.")

with col3:
    st.subheader("Daily Customers (Last 30 Days)")
    if 'order_date' in df.columns:
        last_date = df['order_date'].max()
        recent = df[df['order_date'] >= last_date - pd.Timedelta(days=29)]
        daily = recent.groupby(recent['order_date'].dt.date).agg(customers=('customer_id', 'nunique')).reset_index()
        if not daily.empty:
            fig_daily = px.bar(daily, x='order_date', y='customers', labels={'order_date':'Date','customers':'Unique Customers'})
            st.plotly_chart(fig_daily, use_container_width=True)
        else:
            st.write("No orders in the last 30 days.")

st.markdown("---")

# ---------- RFM Distribution & Time Series ----------
c1, c2 = st.columns([1,1.3])

with c1:
    st.subheader("RFM Score Distribution")
    if 'RFM_Score' in df.columns:
        fig_rfm = px.histogram(df, x='RFM_Score', nbins=20, labels={'RFM_Score':'RFM Score'})
        st.plotly_chart(fig_rfm, use_container_width=True)
    else:
        st.write("No RFM Score column.")

with c2:
    st.subheader("Customers, Sales, and Profit Over Time")
    if 'order_date' in df.columns:
        yearly = df.groupby(df['order_date'].dt.year).agg(
            total_sales=('sales','sum'),
            total_profit=('profit','sum'),
            total_customers=('customer_id','nunique')
        ).reset_index().rename(columns={'order_date':'year'})
        if not yearly.empty:
            fig_ts = go.Figure()
            fig_ts.add_trace(go.Scatter(x=yearly['year'], y=yearly['total_customers'], mode='lines+markers', name='Customers'))
            fig_ts.add_trace(go.Scatter(x=yearly['year'], y=yearly['total_sales'], mode='lines+markers', name='Sales'))
            fig_ts.add_trace(go.Scatter(x=yearly['year'], y=yearly['total_profit'], mode='lines+markers', name='Profit'))
            fig_ts.update_layout(xaxis=dict(type='category'), height=380)
            st.plotly_chart(fig_ts, use_container_width=True)

st.markdown("---")

# ---------- Customer Details ----------
st.subheader("Customer Details")
top_n = st.selectbox("Show top customers by sales:", options=[10,20,50,100], index=0)
cust_agg = df.groupby(['customer_id','customer_name','Segmentasi'], as_index=False).agg(
    total_sales=('sales','sum'),
    total_profit=('profit','sum'),
    recency=('Recency', 'min'),
    frequency=('Frequency', 'max'),
    monetary=('Monetary','max'),
    rfm_score=('RFM_Score','max')
)
cust_sorted = cust_agg.sort_values('total_sales', ascending=False).head(top_n)
st.dataframe(cust_sorted.style.format({
    'total_sales': '${:,.2f}',
    'total_profit': '${:,.2f}',
    'recency': '{:.0f}',
    'frequency': '{:.0f}',
    'monetary': '${:,.2f}',
    'rfm_score': '{:.0f}'
}), height=360)

csv = cust_sorted.to_csv(index=False)
st.download_button("Download CSV", data=csv, file_name="top_customers.csv", mime="text/csv")

st.markdown("---")

# ---------- Recommendations ----------
st.subheader("Segmentation Recommendations")
st.markdown("""
- **At Risk**: run reactivation campaigns (discounts, remarketing emails, reminders).  
- **Champions**: reward with VIP programs or exclusive upsell offers.  
- **Potential Loyalist**: target promotions to increase frequency.  
- **Loyal**: cross-sell higher value products.  
- **Uncategorized**: review data quality or onboarding process.  
""")

# ---------- Footer ----------
st.markdown("---")
st.markdown(f"**Dataset rows:** {len(df):,} · **Unique customers:** {df['customer_id'].nunique():,}")

