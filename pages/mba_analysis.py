import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

st.set_page_config(
    page_title="Market Basket Analysis",
    page_icon="�",
)

def preprocess_data(df):
    """
    Preprocesses the uploaded dataframe.
    - Normalizes column names (strips whitespace, converts to lowercase).
    - Renames columns to standard names ('transaction_id', 'item', 'category').
    - Checks for missing required columns and null values.
    """
    errors = []
    
    # Normalize column names
    df.columns = df.columns.str.strip().str.lower()
    
    # Define keywords to find the correct columns
    rename_map = {
        'transaction_id': ['transactie', 'transaction', 'order', 'bonid'],
        'item': ['item', 'product'],
        'category': ['categorie', 'category']
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

    # Validate that required columns exist
    required_cols = ['transaction_id', 'item']
    for col in required_cols:
        if col not in df.columns:
            errors.append(f"Column '{col}' is required but not found. Please ensure a column named like 'TransactionID' or 'Item' exists.")
    
    # If category analysis is possible, check for that column
    if 'category' not in df.columns:
        st.info("Note: No 'category' column found. Analysis at the category level is not possible.")
    
    if errors:
        return df, errors, False # Return with errors

    # Check for missing values in required columns
    for col in df.columns:
        if col in ['transaction_id', 'item', 'category']:
             # Drop rows where transaction_id or item is null, as they are essential
            if df[col].isnull().any() and col in ['transaction_id', 'item']:
                errors.append(f"Warning: Column '{col}' contains null values. Rows with null values in this column will be removed.")
                df.dropna(subset=[col], inplace=True)

    # Convert columns to appropriate types
    df['transaction_id'] = df['transaction_id'].astype(str)
    df['item'] = df['item'].astype(str)
    if 'category' in df.columns:
        df['category'] = df['category'].astype(str)

    has_category = 'category' in df.columns
    return df, errors, has_category


def market_basket_analysis(df, analysis_level='product', min_support=0.01, min_confidence=0.1, min_lift=1.0):
    """
    Performs market basket analysis using the Apriori algorithm.
    """
    messages = []
    
    if analysis_level == 'category':
        if 'category' not in df.columns:
            messages.append("Error: Analysis on category level selected, but no 'category' column was found.")
            return pd.DataFrame(), messages
        # Group by transaction and get unique categories
        transactions_list = df.groupby('transaction_id')['category'].unique().apply(list).tolist()
    else: # Default to product level
        # Group by transaction and get unique items
        transactions_list = df.groupby('transaction_id')['item'].unique().apply(list).tolist()

    if not transactions_list:
        messages.append("No transactions found to analyze.")
        return pd.DataFrame(), messages

    # Transform data into a one-hot encoded format
    te = TransactionEncoder()
    te_ary = te.fit(transactions_list).transform(transactions_list)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

    if df_encoded.empty:
        messages.append("The encoded DataFrame is empty. Please check the transaction structure.")
        return pd.DataFrame(), messages

    # Generate frequent itemsets using Apriori
    frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)
    
    messages.append(f"Number of frequent sets found at {analysis_level} level: {len(frequent_itemsets)}")
    if frequent_itemsets.empty:
        messages.append("Note: No rules could be defined because no frequent sets were found. Try lowering the 'Minimum support'.")
        return pd.DataFrame(), messages

    # Generate association rules
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    messages.append(f"Number of rules found (before lift filter): {len(rules)}")
    if rules.empty:
        messages.append("Note: Frequent sets were found, but no rules match the chosen 'Minimum confidence'. Try lowering it.")

    # Filter rules by lift
    rules = rules[rules['lift'] >= min_lift]
    messages.append(f"Number of rules found after lift filter: {len(rules)}")

    # Format and return results
    rules = rules.sort_values(by=['lift', 'confidence'], ascending=[False, False])
    
    return rules, messages


def run():
    st.title("🛒 Market Basket Analysis")
    st.markdown("""
                This tool helps you uncover hidden patterns in your sales data.
                By analyzing transactions, it identifies products or categories that are frequently purchased together. 
                Use the results to analyze the effectiveness of promotions, cross-selling opportunities, and to find customer preferences.

                The resulting table shows the most significant associations, ranked by lift, confidence, and support.
    """)
    with st.expander("How to interpret the results", expanded=False):
        # Alle content die je hieronder plaatst (met inspringing)
        # verschijnt BINNEN de uitklapbare sectie.
        st.markdown("""
            To interpret the results, think of each rule as an "if-then" statement: If a customer buys the Antecedent, then they are also likely to buy the Consequent.

            Explanation of the terms used:
            - **Antecedents**: The "if..." part of the rule. This is the item or set of items found in a basket.
            - **Consequents**: The "...then" part of the rule. This is the item or set of items also found in the basket.
            - **Support** (popularity): How frequently the combination appears in all transactions.
            - **Confidence** (reliability): The probability that, if a customer buys the antecedent, the consequent will also be purchased.
            - **Lift** (strength): Measures how much more often A and B occur together than expected if they were statistically independent.
        """)
    st.markdown("""
                Upload a CSV file with your transaction data to discover association rules.
                The file must contain at least columns for **transaction ID** and **item/product**. 
                A **category** column is optional.
    """)
    st.write("")

    # Provide a correct template for download
    template_df = pd.DataFrame({
        "transaction_id": ["Can be any string or number", "1-NDINV00", "10052456", "10254BE"],
        "item": ["Product name or SKU as a string", "1-bc054", "Men's Jacket Size M", "Butter"],
        "category": ["Can be any string", "Video games", "Men's wear", "Dairy"]
    })
    
    st.download_button(
        label="Download CSV Template",
        data=template_df.to_csv(index=False).encode('utf-8'),
        file_name="mba_template.csv",
        mime="text/csv"
    )

    data = st.file_uploader("Choose a CSV file", type="csv")
    
    if data is not None:
        try:
            # Try reading with different separators
            df = pd.read_csv(data, sep=None, engine='python')
        except Exception as e:
            st.error(f"Error reading the CSV file: {e}")
            return
            
        df, errors, has_category = preprocess_data(df.copy()) # Use a copy to avoid mutation issues
        
        if errors:
            for error in errors:
                st.warning(error)

        # Check if preprocessing resulted in critical errors
        if 'transaction_id' not in df.columns or 'item' not in df.columns:
            st.error("Preprocessing failed. Please ensure your file contains the required columns.")
            return

        st.success("File successfully loaded and preprocessed.")
        st.write("### A random sample of your data:")
        st.dataframe(df.sample(min(10, len(df))))

        # Sidebar for controls
        st.sidebar.header("Analysis Settings")
        
        level_options = ['product']
        if has_category:
            level_options.append('category')
        
        level = st.sidebar.selectbox("Analyze at level:", level_options)
        min_support = st.sidebar.slider("Minimum support:", 
                                        0.001, 
                                        0.5, 
                                        0.01, 
                                        step=0.001, 
                                        format="%.3f", 
                                        help="How frequently an itemset (e.g., {Game console, Video game X}) appears in all transactions. A minimum support of 0.02 means the itemset must appear in at least 2% of all transactions.")
        min_confidence = st.sidebar.slider("Minimum confidence:", 
                                           0.01, 
                                           1.0, 
                                           0.1, 
                                           step=0.01, 
                                           help="The conditional probability. For a rule 'If A then B', confidence is P(B|A). A minimum confidence of 0.6 means that in 60% of the transactions containing A, B is also present.")
        min_lift = st.sidebar.slider("Minimum lift:", 
                                     1.0, 
                                     10.0, 
                                     1.0, 
                                     step=0.1, 
                                     help="Measures how much more often A and B occur together than expected if they were statistically independent. A lift of 1 means A and B are independent. A lift > 1 means they are positively correlated.")

        if st.sidebar.button("Analyze"):
            with st.spinner("Analysis in progress..."):
                mba_results, messages = market_basket_analysis(
                    df=df,
                    analysis_level=level,
                    min_support=min_support,
                    min_confidence=min_confidence,
                    min_lift=min_lift
                )
            
            analysis_log = st.empty()
            logged_messages = []
            for msg in messages:
                logged_messages.append(str(msg))
                log_string = "### Analysis Log\n\n" + "\n\n- ".join(logged_messages)
                analysis_log.info(log_string)

            if not mba_results.empty:
                st.header(f"Analysis Results ({level.capitalize()} level)")
                
                # Format for display
                display_results = mba_results.copy()
                display_results['antecedents'] = display_results['antecedents'].apply(lambda x: ', '.join(list(x)))
                display_results['consequents'] = display_results['consequents'].apply(lambda x: ', '.join(list(x)))
                display_results.reset_index(drop=True, inplace=True)
                display_results.insert(0, 'Rank', display_results.index + 1)
                
                st.dataframe(
                    display_results[[
                        'Rank', 'antecedents', 'consequents', 'support', 'confidence', 'lift'
                    ]],
                    hide_index=True,
                    column_config={
                        "support": st.column_config.NumberColumn(format="%.4f"),
                        "confidence": st.column_config.NumberColumn(format="%.4f"),
                        "lift": st.column_config.NumberColumn(format="%.4f"),
                    }
                )
                
                # Conclusion for the top rule
                st.header("Explanation of the Strongest Rule")
                top_rule = display_results.iloc[0]
                antecedents = top_rule['antecedents']
                consequents = top_rule['consequents']
                support = top_rule['support']
                confidence = top_rule['confidence']
                lift = top_rule['lift']

                st.markdown(f"The combination of **'{antecedents}'** and **'{consequents}'** is bought together the most.")
                st.markdown(f"- This combination appears in **{support * 100:.2f}%** of all transactions.")
                st.markdown(f"- If a customer buys '{antecedents}', there is a **{confidence * 100:.2f}%** chance they will also buy '{consequents}'.")
                st.markdown(f"- This happens **{lift:.2f} times more often** than would be expected by chance.")
            else:
                st.warning("\nNo association rules found with the current parameters.")

if __name__ == "__main__":
    run()