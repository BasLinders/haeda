import pandas as pd
import numpy as np
import datetime as dt
import plotly.express as px
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
import streamlit as st

st.set_page_config(
    page_title="RFM Analysis",
    page_icon="📈",
)

# --- DATA INGESTION ---

def generate_mock_data():
    """Generates synthetic transaction data for testing."""
    np.random.seed(42)
    n_customers = 2000
    data = []
    
    # Create a date range for the last 2 years
    end_date = dt.datetime.now()
    start_date = end_date - dt.timedelta(days=730)
    
    for i in range(n_customers):
        customer_id = f"CUST-{1000 + i}"
        # Randomly decide if they are a loyal or one-time customer
        is_loyal = np.random.choice([True, False], p=[0.4, 0.6])
        n_purchases = np.random.randint(3, 15) if is_loyal else 1
        
        # Spread purchases between their "birth" and now
        first_purchase = start_date + dt.timedelta(days=np.random.randint(0, 400))
        
        for j in range(n_purchases):
            # Orders happen randomly after the first purchase
            days_since_first = np.random.randint(0, (end_date - first_purchase).days)
            order_date = first_purchase + dt.timedelta(days=days_since_first)
            
            # Whales spend much more. We need whales.
            is_whale = np.random.choice([True, False], p=[0.05, 0.95])
            amount = np.random.uniform(500, 2000) if is_whale else np.random.uniform(10, 200)
            
            data.append({
                'OrderID': f"ORD-{10000 + len(data)}",
                'CustomerID': customer_id,
                'OrderDate': order_date,
                'TotalSum': round(amount, 2)
            })
            
    return pd.DataFrame(data)

def preprocess_data(df):
    errors = []
    
    # Normalize column names
    df.columns = df.columns.str.strip().str.lower()
    
    # Define keywords
    rename_map = {
        'OrderID': ['transactie', 'transaction', 'order', 'bonid', 'transaction_id'],
        'CustomerID': ['customer', 'klant', 'id', 'user_id', 'identifier'],
        'OrderDate': ['date', 'datum', 'time', 'timestamp', 'dag'],
        'TotalSum': ['sum', 'amount', 'bedrag', 'total', 'price', 'value', 'purchase_revenue', 'revenue']
    }

    # Find and rename columns
    found_columns = {}
    for standard_name, keywords in rename_map.items():
        for keyword in keywords:
            for col in df.columns:
                if keyword in col:
                    found_columns[standard_name] = col
                    break
            if standard_name in found_columns:
                break
    
    df.rename(columns={v: k for k, v in found_columns.items()}, inplace=True)

    # Validate required columns
    required_cols = ['OrderID', 'CustomerID', 'OrderDate', 'TotalSum']
    for col in required_cols:
        if col not in df.columns:
            errors.append(f"Column '{col}' is required but not found.")
    
    if errors:
        return df, errors, False

    # --- CLEANING & STRIPPING ---

    # Keep columns needed for RFM analysis
    df = df[required_cols].copy()

    # Drop rows with missing values in these specific columns
    df.dropna(inplace=True)

    # Convert types
    try:
        df['OrderDate'] = pd.to_datetime(df['OrderDate'])
    except:
        errors.append("Invalid date format detected.")
        return df, errors, False

    df['TotalSum'] = df['TotalSum'].astype(str)
    df['TotalSum'] = df['TotalSum'].str.replace(',', '.', regex=False).str.replace(r'[^\d.-]', '', regex=True) 
    df['TotalSum'] = pd.to_numeric(df['TotalSum'], errors='coerce')
    df.dropna(subset=['TotalSum'], inplace=True)
    
    # Filter for positive transactions only
    df = df[df['TotalSum'] > 0]

    df['OrderID'] = df['OrderID'].astype(str)
    df['CustomerID'] = df['CustomerID'].astype(str)

    return df, errors

# --- HELPERS ---

def calculate_rfm(df):
    snapshot_date = df['OrderDate'].max() + dt.timedelta(days=1)

    rfm = df.groupby('CustomerID').agg({
        'OrderDate': lambda x: (snapshot_date - x.max()).days, 
        'OrderID': 'count',
        'TotalSum': 'sum'
    })

    rfm.rename(columns={
        'OrderDate': 'Recency',
        'OrderID': 'Frequency',
        'TotalSum': 'Monetary'
    }, inplace=True)

    # Quantiles / scores (1-5)
    r_labels = range(5, 0 , -1) # labels reversed because lower number is better in recency
    f_labels = range(1, 6)
    m_labels = range(1, 6)

    rfm['R'] = pd.qcut(rfm['Recency'].rank(method='first'), q=5, labels=r_labels)
    rfm['F'] = pd.qcut(rfm['Frequency'].rank(method='first'), q=5, labels=f_labels)
    rfm['M'] = pd.qcut(rfm['Monetary'].rank(method='first'), q=5, labels=m_labels)

    rfm['RFM_ID'] = rfm.apply(lambda x: f"{x['R']}{x['F']}{x['M']}", axis=1)
    rfm['RFM_score'] = rfm[['R', 'F', 'M']].sum(axis=1)

    return rfm

def get_segment_name(row, whale_threshold):

    if row['Monetary'] >= whale_threshold:
        return 'Whale'
    
    if row['R'] >= 4 and row['F'] >= 4:
        return 'Champions'
    elif row['R'] >= 3 and row['F'] >= 3:
        return 'Loyal Customers'
    elif row['R'] >= 4 and row['F'] <= 2:
        return 'Recent / New'
    elif row['R'] <= 2 and row['F'] >= 4:
        return 'At Risk'
    elif row['R'] <= 2 and row['F'] <= 2:
        return 'Lost'
    else:
        return 'Average / Others'
    
def calculate_predictive_rfm(df):
    snapshot_date = df['OrderDate'].max() + dt.timedelta(days=1)
    
    # Calculate basic birth/death/age dates per customer
    predictive_rfm = df.groupby('CustomerID').agg(
        first_purchase=('OrderDate', 'min'),
        last_purchase=('OrderDate', 'max'),
        total_orders=('OrderID', 'nunique')
    )
    
    # Frequency (x): Only repeat purchases
    predictive_rfm['x'] = predictive_rfm['total_orders'] - 1
    
    # Recency (tx): Duration between first and last purchase
    predictive_rfm['t_x'] = (predictive_rfm['last_purchase'] - predictive_rfm['first_purchase']).dt.days
    
    # Age (T): Duration between first purchase and snapshot
    predictive_rfm['T'] = (snapshot_date - predictive_rfm['first_purchase']).dt.days
    
    # Monetary Value (m): Average value of REPEAT purchases
    predictive_rfm['m'] = df.groupby('CustomerID')['TotalSum'].mean()
    
    return predictive_rfm[['x', 't_x', 'T', 'm']]

def predictions(predictive_rfm):
    bgf = BetaGeoFitter(penalizer_coef=0.0)
    bgf.fit(predictive_rfm['x'], predictive_rfm['t_x'], predictive_rfm['T'])

    # Predict purchases for the next 30 days
    t = 30 
    predictive_rfm['predicted_purchases'] = bgf.conditional_expected_number_of_purchases_up_to_time(
        t, predictive_rfm['x'], predictive_rfm['t_x'], predictive_rfm['T']
    )

    # Probability the customer is still active
    predictive_rfm['p_alive'] = bgf.conditional_probability_alive(
        predictive_rfm['x'], predictive_rfm['t_x'], predictive_rfm['T']
    )

    return predictive_rfm, bgf

def calculate_clv(predictive_rfm, bgf, months=12):
    # Filter for customers with at least one repeat purchase
    returning_customers = predictive_rfm[(predictive_rfm['x'] > 0) & (predictive_rfm['m'] > 0)]
    
    corr = returning_customers[['x', 'm']].corr().iloc[0,1]
    if abs(corr) > 0.3:
        st.warning(f"Warning: High correlation ({corr:.2f}) between Frequency and Monetary value detected. CLV estimates may be unstable.")

    # Fit the Gamma-Gamma Model
    ggf = GammaGammaFitter(penalizer_coef=0.0)
    ggf.fit(returning_customers['x'], returning_customers['m'])
    
    # Predict the average transaction value
    predictive_rfm['exp_avg_sales'] = ggf.conditional_expected_average_profit(
        predictive_rfm['x'],
        predictive_rfm['m']
    )
    
    # Calculate Customer Lifetime Value
    # This combines BG/NBD (frequency) and Gamma-Gamma (monetary) to get CLV
    predictive_rfm['clv'] = ggf.customer_lifetime_value(
        bgf, # The BG/NBD model from 'predictions'
        predictive_rfm['x'],
        predictive_rfm['t_x'],
        predictive_rfm['T'],
        predictive_rfm['m'],
        time=months, # Forecast horizon
        discount_rate=0.01 # Monthly discount rate (standard is ~1%)
    )
    
    return predictive_rfm

def style_rfm_table(df):
    def highlight_segments(row):
        # Default style
        style = [''] * len(row)
        
        # Highlight Whales in Gold
        if row['Segment'] == 'Whale':
            style = ['background-color: #FFF9C4; color: black; font-weight: bold'] * len(row)
        
        # Highlight At Risk/Lost in Light Red
        elif row['Segment'] in ['At Risk', 'Lost']:
            style = ['background-color: #FFEBEE; color: #B71C1C'] * len(row)
            
        # Highlight Champions in Light Green
        elif row['Segment'] == 'Champions':
            style = ['background-color: #E8F5E9; color: #2E7D32'] * len(row)
            
        return style

    styled_df = df.style.apply(highlight_segments, axis=1)\
                        .format({
                            'Monetary': '€{:,.2f}',
                            'clv': '€{:,.2f}',
                            'p_alive': '{:.2%}',
                            'predicted_purchases': '{:.2f}'
                        })
    return styled_df
    
# --- VISUALIZATIONS ---

def plotCustomers(rfm):
    plot_data = rfm.sample(n=min(10000, len(rfm)), random_state=42)

    fig = px.scatter_3d(
        plot_data, 
        x='Recency', 
        y='Frequency', 
        z='Monetary',
        color='Segment',  # Color-coded by the labels we created
        opacity=0.7,
        title='RFM Customer Segmentation',
        hover_data=['RFM_ID'], # Shows '555' etc on hover
        height=700
    )
    
    # Improve layout for Streamlit
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=50))
    
    return fig

def plotSegmentDistribution(rfm):
    # Count how many customers in each segment
    segment_counts = rfm['Segment'].value_counts().reset_index()
    segment_counts.columns = ['Segment', 'Count']
    
    fig = px.treemap(
        segment_counts, 
        path=['Segment'], 
        values='Count',
        color='Count',
        color_continuous_scale='RdBu',
        title='Customer Segment Proportions'
    )
    return fig

# --- MAIN LOGIC ---

def run():
    st.title("RFM Analysis")
    st.markdown("""
        This tool conducts **RFM (Recency, Frequency, Monetary)** analysis to segment your customers based on their historical purchase behavior. 
        
        By combining classic segmentation with **probabilistic forecasting (BG/NBD & Gamma-Gamma models)**, it goes beyond "what happened" to predict "what’s next"—estimating the probability of customer churn and forecasting future **Lifetime Value (CLV)**. 
        
        Upload your transaction data to identify your "Whales," save "At Risk" customers, and optimize your marketing ROI.
    """)
    with st.expander("How the Predictive Models Work", expanded=False):
        st.markdown("""
            While standard RFM looks at the **past**, this tool uses two statistical models to look into the **future**:
            
            ### 1. The BG/NBD "Heartbeat" Model (Beta Geometric/Negative Binomial Distribution)
            Think of every customer as having a "heartbeat" (their purchase frequency). 
            * **The Logic:** If a customer usually buys every 30 days but hasn't bought in 90, the model senses their "heartbeat" is fading. 
            * **The Output:** This gives us the **P(Alive)** metric—the probability that the customer is still a "living" part of your business versus someone who has churned (left permanently).
            
            ### 2. The "Spending Habit" Model (Gamma-Gamma)
            This model looks at **how much** a customer spends when they do decide to buy.
            * **The Logic:** It filters out "one-off" anomalies (like a single massive holiday purchase) to find the customer's *true* average spending habit.
            * **The Output:** This helps us predict the **Expected Average Sales**, ensuring our future estimates aren't skewed by lucky outliers.
            
            ### 3. Customer Lifetime Value (CLV)
            We combine the two models above to answer the ultimate business question: 
            > *"Based on their heartbeat and their spending habits, how much is this person worth to us over the next 12 months?"*
            
            **Why should I trust this?** These models are the industry standard for "Non-Contractual" businesses (like e-commerce or retail), where customers are free to come and go without canceling a subscription or order.
            """)
    with st.expander("CSV Data Requirements & Formatting", expanded=False):
        st.markdown("""
        To ensure a successful analysis, your uploaded CSV should contain the following four pieces of information. The tool will automatically attempt to find these columns even if they have different names (e.g., in Dutch or English).
        """)
        
        # Create a simple table for clarity
        requirements_data = {
            "Requirement": ["Order ID", "Customer ID", "Order Date", "Total Amount"],
            "Accepted Keywords": [
                "transactie, transaction, order, bonid, transaction_id",
                "customer, klant, id, user_id, identifier",
                "date, datum, time, timestamp, dag",
                "sum, amount, bedrag, total, price, value, purchase_revenue, revenue"
            ],
            "Format": ["Text or Number", "Text or Number", "YYYY-MM-DD", "Numeric (e.g. 10.50)"]
        }
        st.table(requirements_data)

        st.info("""
            **Tip:** Ensure your 'Total Amount' column does not contain currency symbols (like € or $) within the cells. 
            The tool will automatically remove rows with negative values (returns) to ensure the predictive models remain accurate.
        """)
    
    # --- STATE INITIALIZATION ---
    if 'df_raw' not in st.session_state:
        st.session_state['df_raw'] = None
    if 'results' not in st.session_state:
        st.session_state['results'] = None

    # --- SIDEBAR INPUTS ---
    uploaded_file = st.sidebar.file_uploader("1. Upload Customer Data", type="csv")
    use_mock = st.sidebar.button("Load Mock Data")

    # --- DATA LOADING LOGIC ---
    if use_mock:
        st.session_state['df_raw'] = generate_mock_data()
        st.sidebar.success("Mock data loaded")
        
    elif uploaded_file is not None:
        st.session_state['df_raw'] = pd.read_csv(uploaded_file)

    # --- MAIN VARIABLE ASSIGNMENT ---
    df = st.session_state['df_raw']
    
    if df is not None:
        # --- PREPROCESS ---
        clean_df, errors = preprocess_data(df)
        
        if errors:
            for error in errors:
                st.sidebar.error(error)
            if any("required but not found" in e for e in errors):
                return # Stop if critical columns are missing
        
        st.sidebar.success("Data cleaned and mapped.")

        # --- DATA PREVIEW ---
        st.header("Data Preview")
        st.write("Below is a sample of your cleaned data. Please verify the columns before analyzing.")
        st.dataframe(clean_df.sample(min(len(clean_df), 5)), width='stretch')
        
        st.divider()

        # --- ANALYSIS TRIGGER ---
        st.subheader("Execute Analysis")
        analyze_btn = st.button("Run Full RFM & Predictive Analysis")

        # CALCULATION BLOCK
        if analyze_btn:
            with st.spinner("Calculating segments and forecasting future value..."):
                # Classic RFM
                rfm_df = calculate_rfm(clean_df)
                whale_threshold = rfm_df['Monetary'].quantile(0.95)
                rfm_df['Segment'] = rfm_df.apply(
                    get_segment_name, 
                    axis=1, 
                    whale_threshold=whale_threshold
                )

                # Predictive Analysis
                try:
                    predictive_df = calculate_predictive_rfm(clean_df)
                    predictive_df, bgf_model = predictions(predictive_df) 
                    final_predictive = calculate_clv(predictive_df, bgf_model)
                    
                    # Merge
                    final_report = rfm_df.join(final_predictive[['predicted_purchases', 'p_alive', 'clv']])
                    st.success("Analysis complete including Predictive Models")
                except Exception as e:
                    st.warning(f"Could not run predictions (possibly not enough repeat data): {e}. Showing Classic RFM only.")
                    final_report = rfm_df.copy()

                # Save result to session state
                st.session_state['results'] = final_report

        # VISUALIZATION BLOCK
        # Runs if results exist in memory (either just calculated, or from history)
        if st.session_state['results'] is not None:
            final_report = st.session_state['results']

            # --- OUTPUT ---
            with st.expander("How to interpret the Analysis", expanded=False):
                st.subheader("Customer Segmentation (Classic RFM)")
                st.markdown("""
                * **Whale**: The top-tier revenue drivers. These customers are in the **95th percentile** of total spend. 
                    * *Suggestion: High-touch VIP support and exclusive rewards.*
                * **Champions**: The best of the best. They bought recently, buy often, and spend heavily.
                    * *Suggestion: Reward them. They can be early adopters for new products.*
                * **Loyal Customers**: Steady and reliable. They return frequently and spend well.
                    * *Suggestion: Use upsell strategies to move them into the 'Champion' or 'Whale' category.*
                * **Recent / New**: Customers who made their first/only purchases very recently.
                    * *Suggestion: Onboarding campaigns to trigger that critical second purchase.*
                * **At Risk**: Former frequent buyers who haven't returned in a while.
                    * *Suggestion: Send 'We miss you' discount codes or personalized re-engagement emails.*
                * **Lost**: Low frequency, low monetary value, and haven't bought in a long time.
                    * *Suggestion: Don't overspend on marketing here; focus on low-cost automated reach-outs.*
                """)

                st.divider()

                st.subheader("Predictive Analytics (BG/NBD & Gamma-Gamma)")
                st.markdown("""
                Unlike classic RFM which looks backward, these metrics forecast future behavior:
                
                * **Frequency ($x$)**: The count of **repeat** transactions. We ignore the first purchase to establish a baseline of "loyalty."
                * **Recency ($t_x$)**: The "active lifespan"—the time between their very first and very last purchase.
                * **Age ($T$)**: How long it has been since we first saw this customer. 
                * **P(Alive)**: The probability (0.0 to 1.0) that a customer is still "active." If this drops, the customer is likely churning.
                * **Expected Purchases**: How many orders this specific person is statistically likely to place in the next 30 days.
                * **CLV**: The predicted total value of the customer over the next 12 months, adjusted for the "time value of money."
                """)

            st.subheader("Customer Insights Table")
            st.dataframe(style_rfm_table(final_report), width='stretch')
                
            # Download Button for the processed data
            csv = final_report.to_csv().encode('utf-8')
            st.download_button(
                label="Download Full Analysis as CSV",
                data=csv,
                file_name='rfm_analysis_output.csv',
                mime='text/csv',
            )

            st.subheader("Visual Insights")

            tab1, tab2 = st.tabs(["Customer Segmentation", "Predictive Trends"])

            with tab1:
                st.subheader("The Customer Landscape")
                col1, col2 = st.columns([2, 1]) 
                
                with col1:
                    st.plotly_chart(plotCustomers(final_report),use_container_width=True, key='customers_3d_tab')
                    st.caption("Each dot is a customer. Clusters show how segments vary by Recency, Frequency, and Monetary value.")
                    
                with col2:
                    st.plotly_chart(plotSegmentDistribution(final_report),use_container_width=True, key='segment_tree_tab')
                    st.caption("Proportion of your total customer base by segment.")

            with tab2:
                st.subheader("Predictive Value by Segment")
                
                # Prepare Data for Visualization
                # Group by Segment and calculate mean CLV and P(Alive)
                segment_analysis = final_report.reset_index().groupby('Segment').agg({
                    'clv': 'mean',
                    'p_alive': 'mean',
                    'predicted_purchases': 'mean',
                    'CustomerID': 'count'
                }).reset_index()
                
                # Sort by CLV so the chart is ordered meaningfully
                segment_analysis = segment_analysis.sort_values('clv', ascending=False)
                
                # Key Metrics Row
                col1, col2, col3 = st.columns(3)
                col1.metric("Highest Avg CLV Segment", 
                          f"{segment_analysis.iloc[0]['Segment']}",
                          f"€{segment_analysis.iloc[0]['clv']:,.0f}")
                
                # Find the segment with the highest churn risk (lowest p_alive)
                highest_risk = segment_analysis.sort_values('p_alive').iloc[0]
                col2.metric("Highest Churn Risk", 
                          f"{highest_risk['Segment']}",
                          f"{(1 - highest_risk['p_alive']):.1%} Churn Prob")
                
                col3.metric("Total Forecasted Revenue", 
                          f"€{final_report['clv'].sum():,.0f}")

                st.divider()

                # Visualization: CLV vs Churn Risk
                c1, c2 = st.columns(2)
                
                with c1:
                    st.markdown("##### Average Lifetime Value (12 months) by Segment")
                    fig_clv = px.bar(
                        segment_analysis, 
                        x='Segment', 
                        y='clv', 
                        color='Segment',
                        text_auto='.0f',
                        title="Which segment drives future value?"
                    )
                    fig_clv.update_layout(showlegend=False, xaxis_title=None, yaxis_title="Avg CLV (€)")
                    st.plotly_chart(fig_clv, use_container_width=True)
                    st.caption("Note: If 'Whales' have lower CLV than 'Champions', they may be 'one-hit wonders' who spent a lot once but aren't likely to return.")

                with c2:
                    st.markdown("##### Customer Health (Probability Alive)")
                    # Interpret p_alive as "Health". 1.0 = Healthy, 0.0 = Churned.
                    fig_alive = px.scatter(
                        segment_analysis,
                        x='p_alive',
                        y='Segment',
                        size='CustomerID', # Bubble size = number of customers
                        color='Segment',
                        title="Segment Health vs. Size"
                    )
                    fig_alive.update_layout(
                        showlegend=False, 
                        xaxis_title="Probability of Being Alive (0=Lost, 1=Active)", 
                        xaxis_range=[0, 1.1]
                    )
                    # Add a vertical line at 0.5 to show the "danger zone"
                    fig_alive.add_vline(x=0.5, line_dash="dash", line_color="red", annotation_text="Churn Threshold")
                    st.plotly_chart(fig_alive, use_container_width=True)
                    st.caption("Bubbles to the **left** are effectively lost. Bubbles to the **right** are active. Size represents the number of customers.")

                # 4. Drills Down: Actionable Lists
                st.divider()
                st.subheader("Actionable Intelligence")
                
                # Find "Champions" who are actually at risk (High RFM score, but low p_alive)
                risky_champions = final_report[
                    (final_report['Segment'] == 'Champions') & 
                    (final_report['p_alive'] < 0.5)
                ].sort_values('clv', ascending=False).head(10)
                
                if not risky_champions.empty:
                    st.warning(f"**Urgent Attention:** Found {len(risky_champions)} 'Champions' with high churn risk.")
                    st.write("These customers spent heavily in the past but the model predicts they have stopped engaging. **Contact them immediately.**")
                    st.dataframe(risky_champions[['p_alive', 'clv', 'Recency', 'Frequency', 'Monetary']].style.format({'p_alive': '{:.2%}', 'clv': '€{:.2f}'}))
                else:
                    st.success("Your Champions are healthy! No immediate churn risk detected in the top tier.")
    else:
        # State when no file is uploaded
        st.info("Please upload a CSV file in the sidebar to get started.")

if __name__ == "__main__":
    run()
