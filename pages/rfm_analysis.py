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
@st.cache_data
def generate_mock_data():
    """Generates robust synthetic transaction data for RFM testing."""
    np.random.seed(42)
    n_customers = 2000 
    data = []
    
    end_date = dt.datetime.now()
    # 2 years of history
    start_date = end_date - dt.timedelta(days=730)
    
    for i in range(n_customers):
        customer_id = f"CUST-{1000 + i}"
        
        # Decide Churn Status (20% Churned)
        is_churned = np.random.choice([True, False], p=[0.2, 0.8])
        
        # Define the "Active Window" for this customer
        # If churned, their window ends 100-600 days ago.
        # If active, their window goes up to today.
        if is_churned:
            churn_gap = np.random.randint(100, 600)
            customer_end_date = end_date - dt.timedelta(days=churn_gap)
        else:
            customer_end_date = end_date

        # Calculate total days in their specific timeline
        timeline_days = (customer_end_date - start_date).days
        if timeline_days < 10: timeline_days = 10 # Safety buffer

        # Decide Loyalty (Frequency)
        # 40% are repeat buyers (Loyal)
        is_loyal = np.random.choice([True, False], p=[0.4, 0.6])
        
        if is_loyal:
            n_purchases = np.random.randint(2, 12)
        else:
            n_purchases = 1

        # Generate Dates
        # We pick N unique days from their available timeline.
        # replace=False ensures no same-day duplicates (which mess up Frequency counts)
        if n_purchases > timeline_days: n_purchases = timeline_days
        
        days_from_start = np.random.choice(range(timeline_days), n_purchases, replace=False)
        days_from_start.sort() # Important: Transaction history must be chronological
        
        for day_offset in days_from_start:
            order_date = start_date + dt.timedelta(days=int(day_offset))
            
            # Whales spend much more
            is_whale = np.random.choice([True, False], p=[0.05, 0.95])
            amount = np.random.uniform(500, 2000) if is_whale else np.random.uniform(20, 200)
            
            data.append({
                'OrderID': f"ORD-{len(data)}",
                'CustomerID': customer_id,
                'OrderDate': order_date,
                'TotalSum': round(amount, 2)
            })
            
    return pd.DataFrame(data)
    
def preprocess_data(df):
    errors = []
    
    # Normalize column names
    df.columns = df.columns.str.strip().str.lower()
    
    # Define keywords (Removed dangerous generic words like 'id', 'date', 'sum')
    # Order matters: Put specific keywords FIRST.
    rename_map = {
        'OrderID': ['transactie', 'transaction_id', 'order_id', 'bonid', 'order', 'Order ID'],
        'CustomerID': ['customer', 'klant', 'user_id', 'identifier', 'email', 'contact', 'Customer ID'],
        'OrderDate': ['order_date', 'datum', 'timestamp', 'time', 'created_at', 'Order Date'],
        'TotalSum': ['total_amount', 'purchase_revenue', 'price', 'value', 'bedrag', 'amount', 'total', 'Total Amount']
    }

    found_columns = {}
    used_columns = set() # Track which columns we have already claimed

    for standard_name, keywords in rename_map.items():
        match_found = False
        
        # Priority 1: Exact Match (e.g., "date" == "date")
        for col in df.columns:
            if col == standard_name.lower() and col not in used_columns:
                found_columns[standard_name] = col
                used_columns.add(col)
                match_found = True
                break
        
        if match_found: continue

        # Priority 2: Keyword Match
        for keyword in keywords:
            for col in df.columns:
                # Check if keyword is in column AND column isn't already used
                if keyword in col and col not in used_columns:
                    found_columns[standard_name] = col
                    used_columns.add(col)
                    match_found = True
                    break
            if match_found: break

    # Rename mapped columns
    df.rename(columns={v: k for k, v in found_columns.items()}, inplace=True)

    # Validate required columns
    required_cols = ['OrderID', 'CustomerID', 'OrderDate', 'TotalSum']
    missing = [col for col in required_cols if col not in df.columns]
    
    if missing:
        # Return specific error to help user debug
        errors.append(f"Missing required columns: {', '.join(missing)}")
        return df, errors, False

    # --- CLEANING ---
    
    # Keep only what we need
    df = df[required_cols].copy()
    
    # Clean ID
    df['CustomerID'] = df['CustomerID'].astype(str).str.strip().str.lower()
    
    # Clean Date
    df['OrderDate'] = pd.to_datetime(df['OrderDate'], dayfirst=True, errors='coerce')
    if df['OrderDate'].isna().all():
         errors.append("Could not parse any valid dates from 'OrderDate' column.")
         return df, errors, False
    df.dropna(subset=['OrderDate'], inplace=True)

    # Clean TotalSum
    df['TotalSum'] = df['TotalSum'].astype(str).str.replace(r'[^\d,.-]', '', regex=True)
    
    # Get the last non-numeric character for each row (the separator)
    last_separators = df['TotalSum'].str.replace(r'\d', '', regex=True).str[-1]
    
    comma_count = last_separators[last_separators == ','].count() # Probably European notation
    dot_count = last_separators[last_separators == '.'].count() # Probably American notation

    # Apply Logic based on the "Winner"
    if comma_count > dot_count:
        # EUROPEAN LOGIC DETECTED (Majority use comma as decimal)
        # Remove thousands separator (.) then swap decimal (,) to (.)
        df['TotalSum'] = df['TotalSum'].str.replace('.', '', regex=False)
        df['TotalSum'] = df['TotalSum'].str.replace(',', '.', regex=False)
    else:
        # AMERICAN LOGIC DETECTED (Majority use dot as decimal)
        # Remove thousands separator (,) -> Python handles the dot natively
        df['TotalSum'] = df['TotalSum'].str.replace(',', '', regex=False)

    # Final Convert
    df['TotalSum'] = pd.to_numeric(df['TotalSum'], errors='coerce')
    
    # Drop failures
    df.dropna(subset=['TotalSum'], inplace=True)
    df = df[df['TotalSum'] > 0]

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
    bgf = BetaGeoFitter(penalizer_coef=0.01)
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
    ggf = GammaGammaFitter(penalizer_coef=0.01)
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
        
        By combining classic segmentation with **probabilistic forecasting (BG/NBD & Gamma-Gamma models)**, it goes beyond "what happened" to predict "what’s next": estimating the probability of customer churn and forecasting future **Lifetime Value (CLV)**. 
        
        Upload your transaction data to identify your "Whales," save "At Risk" customers, and optimize your marketing ROI.
    """)
    with st.expander("How the Predictive Models Work", expanded=False):
        st.markdown("""
            While standard RFM looks at the **past**, this tool uses two statistical models to look into the **future**:
            
            ### 1. The BG/NBD "Heartbeat" Model (Beta Geometric/Negative Binomial Distribution)
            Think of every customer as having a "heartbeat" (their purchase frequency). 
            * **The Logic:** If a customer usually buys every 30 days but hasn't bought in 90, the model senses their "heartbeat" is fading. 
            * **The Output:** This gives us the **P(Alive)** metric: the probability that the customer is still a "living" part of your business versus someone who has churned (left permanently).
            
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
                "customer, klant, id, user_id, identifier, e-mail address",
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

        st.subheader("Timeframe Settings")
        
        # 1. Show the user the detected range
        min_date = clean_df['OrderDate'].min()
        max_date = clean_df['OrderDate'].max()
        st.info(f"Detected Data Range: **{min_date.date()}** to **{max_date.date()}**")
        
        # 2. Add an interactive slider to trim the data
        # Default: Start from 3 years ago, End at "Today"
        default_start = max(min_date, max_date - dt.timedelta(days=365*3))
        
        date_range = st.slider(
            "Select Analysis Period",
            min_value=min_date.date(),
            max_value=max_date.date(),
            value=(default_start.date(), max_date.date())
        )
        
        # 3. Filter the dataframe based on slider selection
        mask = (clean_df['OrderDate'].dt.date >= date_range[0]) & \
               (clean_df['OrderDate'].dt.date <= date_range[1])
        clean_df = clean_df[mask]
        
        st.write(f"Analyzing **{len(clean_df)}** transactions within this period.")
        st.divider()
        
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
        analyze_btn = st.button("Run Full RFM & Predictive Analysis", type='primary')

        # CALCULATION BLOCK
        if analyze_btn:
            progress_bar = st.progress(0, text="Initializing analysis...")
            
            # Initialize final_report to None to ensure scope safety
            final_report = None 
            
            try:
                # --- Classic RFM ---
                progress_bar.progress(10, text="Calculating historical RFM metrics...")
                rfm_df = calculate_rfm(clean_df)

                progress_bar.progress(30, text="Identifying Whales and Champions...")
                whale_threshold = rfm_df['Monetary'].quantile(0.95)
                rfm_df['Segment'] = rfm_df.apply(
                    get_segment_name, 
                    axis=1, 
                    whale_threshold=whale_threshold
                )

                # --- Predictive Models ---
                progress_bar.progress(50, text="Preparing data for statistical models...")
                predictive_df = calculate_predictive_rfm(clean_df)

                progress_bar.progress(70, text="Fitting BG/NBD model (Predicting Churn Risk)...")
                predictive_df, bgf_model = predictions(predictive_df)

                progress_bar.progress(90, text="Fitting Gamma-Gamma model (Forecasting CLV)...")
                final_predictive = calculate_clv(predictive_df, bgf_model)
                
                # Merge
                final_report = rfm_df.join(final_predictive[['predicted_purchases', 'p_alive', 'clv']])
                
                progress_bar.progress(100, text="Analysis Complete!")
                st.success("Analysis complete, including Predictive Models.")
                
            except Exception as e:
                progress_bar.empty()
                st.warning(f"Could not run predictions: {e}. Showing Classic RFM only.")
                
                # Fallback: Use the basic RFM dataframe if predictions fail
                if 'rfm_df' in locals():
                    final_report = rfm_df.copy()
                else:
                    st.error("Critical error in RFM calculation.")
                    st.stop()

            if final_report is not None:
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
                * **Recency ($t_x$)**: The "active lifespan": the time between their very first and very last purchase.
                * **Age ($T$)**: How long it has been since we first saw this customer. 
                * **P(Alive)**: The probability (0.0 to 1.0) that a customer is still "active." If this drops, the customer is likely churning.
                * **Expected Purchases**: How many orders this specific person is statistically likely to place in the next 30 days.
                * **CLV**: The predicted total value of the customer over the next 12 months, adjusted for the "time value of money."
                """)

            st.subheader("Customer Insights Table")
            st.caption("*Displaying top 1,000 customers sorted by Monetary value. Download the CSV for the full list.*")

            sort_col = 'clv' if 'clv' in final_report.columns else 'Monetary'
            preview_df = final_report.sort_values(sort_col, ascending=False).head(1000)
            st.dataframe(style_rfm_table(preview_df), width='stretch')

            # --- DEBUG: CHECK TOTALSUM CONVERSION ---
            if uploaded_file is not None:
                with st.expander("Debug: Check Number Conversion"):
                    st.write("This shows the raw text from your CSV vs. the number Python understood.")
                    
                    # 1. Get raw values from the uploaded file (before cleaning)
                    uploaded_file.seek(0)
                    df_raw_debug = pd.read_csv(uploaded_file)
                    
                    # Find the TotalSum column (using the same mapping logic)
                    raw_col_name = None
                    possible_names = ['total_amount', 'purchase_revenue', 'price', 'value', 'bedrag', 'amount', 'total']
                    for col in df_raw_debug.columns:
                        if any(x in col.lower() for x in possible_names):
                            raw_col_name = col
                            break
                    
                    if raw_col_name:
                        sample_indices = clean_df.sample(20).index
                        debug_comparison = pd.DataFrame({
                            'Raw Text (CSV)': df_raw_debug.loc[sample_indices, raw_col_name].astype(str),
                            'Converted Number': clean_df.loc[sample_indices, 'TotalSum']
                        })
                        st.table(debug_comparison.style.format({'Converted Number': '{:,.2f}'}))
                    else:
                        st.warning("Could not automatically find the original TotalSum column for debugging.")
                
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
                if 'clv' not in final_report.columns:
                    st.warning("Predictive Analytics could not be generated.")
                    st.info("""
                        **Why?** The statistical models (BG/NBD & Gamma-Gamma) failed to converge. 
                        This usually happens when:
                        1. The dataset is too small (< 100 customers).
                        2. There are very few repeat purchases (everyone is a one-time buyer).
                        3. The data does not span a long enough time period.
                        
                        **Solution:** Try loading a larger dataset or one with more repeat transaction history.
                    """)
                    st.stop()
                    
                st.subheader("Predictive Value by Segment")
                
                # Prepare Data for Visualization
                # Group by Segment and calculate mean CLV and P(Alive)
                
                repeat_customers = final_report[final_report['Frequency'] > 1]
                segment_analysis = repeat_customers.reset_index().groupby('Segment').agg({
                    'clv': 'mean',
                    'p_alive': 'mean',
                    'predicted_purchases': 'mean',
                    'CustomerID': 'count'
                }).reset_index()
                
                # Sort by CLV so the chart is ordered meaningfully
                segment_analysis = segment_analysis.sort_values('clv', ascending=False)
                
                # Key Metrics Row
                st.subheader("Actionable Customer Lists")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Top 5 MVPs (Highest CLV)")
                    # Sort by CLV descending
                    top_clv = final_report.sort_values('clv', ascending=False).head(5).copy()
                    top_clv['p_alive'] = top_clv['p_alive'] * 100 # convert to percentage
                    
                    # Create a clean display table
                    st.dataframe(
                        top_clv,
                        column_config={
                            "clv": st.column_config.NumberColumn(
                                "Customer Lifetime Value",
                                help="Predicted total revenue from this customer over the next 12 months.",
                                format="€%.2f"
                            ),
                            "predicted_purchases": st.column_config.NumberColumn(
                                "Exp. Purchases (30d)",
                                help="Expected number of transactions in the next 30 days.",
                                format="%.2f"
                            ),
                            "Segment": st.column_config.TextColumn(
                                "Customer Segment",
                                help="The segment in which the RFM analysis clustered the customer."
                            ),
                            "p_alive": st.column_config.NumberColumn(
                                "Probability Alive", 
                                format="%.1f%%",
                                help="The probability that a customer remains active."
                            )
                        },
                        width="stretch"
                    )
                    st.caption("These 5 customers are predicted to generate the most revenue in the next 12 months.")

                with col2:
                    st.markdown("#### Top 5 At-Risk VIPs")
                    # LOGIC: Filter for customers with High Monetary value (> 75th percentile) 
                    # but Low P(Alive) (< 50%)
                    high_value_mask = final_report['Monetary'] > final_report['Monetary'].quantile(0.75)
                    at_risk_mask = final_report['p_alive'] < 0.5
                    
                    risky_vips = final_report[high_value_mask & at_risk_mask]
                    
                    # Sort by Monetary (past spend) to find the biggest potential losses
                    risky_vips = risky_vips.sort_values('Monetary', ascending=False).head(5)
                    
                    if not risky_vips.empty:
                        st.dataframe(
                            risky_vips[['p_alive', 'Monetary', 'Segment']]
                            .style.format({'p_alive': '{:.1%}', 'Monetary': '€{:.2f}'}),
                            width='stretch'
                        )
                        st.caption("These are high-spenders showing signs of churn. **Contact them immediately.**")
                    else:
                        st.success("No high-value churn risk detected right now.")

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
                    #st.markdown("##### Customer Health (Probability Alive)")
                    # Interpret p_alive as "Health". 1.0 = Healthy, 0.0 = Churned.
                    #fig_alive = px.scatter(
                    #    segment_analysis,
                    #    x='p_alive',
                    #    y='Segment',
                    #    size='CustomerID', # Bubble size = number of customers
                    #    color='Segment',
                    #    title="Segment Health vs. Size"
                    #)
                    #fig_alive.update_layout(
                    #    showlegend=False, 
                    #    xaxis_title="Probability of Being Alive (0=Lost, 1=Active)", 
                    #    xaxis_range=[0, 1.1]
                    #)
                    # Add a vertical line at 0.5 to show the "danger zone"
                    #fig_alive.add_vline(x=0.5, line_dash="dash", line_color="red", annotation_text="Churn Threshold")
                    #st.plotly_chart(fig_alive, use_container_width=True)
                    #st.caption("Bubbles to the **left** are effectively lost. Bubbles to the **right** are active. Size represents the number of customers.")

                    st.markdown("##### Customer Health vs. Value")
                    
                    # Plot 'p_alive'
                    fig_health = px.scatter(
                        segment_analysis, 
                        x='p_alive', 
                        y='clv', 
                        size='CustomerID', 
                        color='Segment',
                        title="Segment Health Matrix (Right = Healthy, Left = Lost)"
                    )
                    
                    # Add a vertical line at 50% probability
                    fig_health.add_vline(x=0.5, line_dash="dash", line_color="red", annotation_text="Churn Threshold")
                    
                    fig_health.update_layout(
                        xaxis_title="Probability Alive (0.0 = Dead, 1.0 = Active)",
                        yaxis_title="Avg CLV (€)",
                        xaxis_range=[-0.05, 1.05] # Add padding so bubbles don't get cut off
                    )
                    st.plotly_chart(fig_health, use_container_width=True)
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
