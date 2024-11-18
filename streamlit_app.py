import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Apply modern theme
sns.set_theme(style="whitegrid", palette="pastel")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv('bank01_baranda_revised.csv')

data = load_data()

# Title and Introduction
st.title("Bank Data Interactive Dashboard")

# Introduction Section
st.header("Introduction")
st.write("""
This dashboard explores bank customer data to address the following key objectives:
- **Predict New Sales**: Identify customer behaviors and trends to estimate potential new purchases.
- **Customer Segmentation**: Analyze customer profiles to group individuals with similar characteristics.
- **Customer Retention**: Identify at-risk customers and suggest strategies for retention.

Our interactive dashboard provides data exploration, insights, and recommendations that will help inform decision-making.
""")

# Sidebar Filters
st.sidebar.header("Filters")

# Gender Filter
gender = st.sidebar.selectbox('Select Gender', ['All', 'Male', 'Female'], key='gender_filter')
filtered_data = data.copy()
if gender != 'All':
    filtered_data = filtered_data[filtered_data[f'demog_gen{gender[0].lower()}'] == 'yes']

# Age Filter
age_range = st.sidebar.slider('Select Age Range', 
                               int(filtered_data['demog_age'].min()), 
                               int(filtered_data['demog_age'].max()), 
                               (20, 30),  # Default range: 20 to 30
                               key='age_slider')
filtered_data = filtered_data[(filtered_data['demog_age'] >= age_range[0]) & 
                              (filtered_data['demog_age'] <= age_range[1])]

# New Sales (INT_TGT) Filter
sales_min = int(data['int_tgt'].min())
sales_max = int(data['int_tgt'].max())
sales_range = st.sidebar.slider("Filter by New Sales (INT_TGT) Range", 
                                 sales_min,  # Minimum value dynamically set
                                 sales_max,  # Maximum value dynamically set
                                 (sales_min, sales_max),  # Slider defaults to full range
                                 key='sales_slider')
filtered_data = filtered_data[(filtered_data['int_tgt'] >= sales_range[0]) & 
                              (filtered_data['int_tgt'] <= sales_range[1])]

# Metric Display
st.header("Metrics Overview")
st.metric(label="Filtered Average Sales", value=f"${filtered_data['int_tgt'].mean():,.2f}")
st.metric(label="Filtered Total Sales", value=f"${filtered_data['int_tgt'].sum():,.2f}")
st.metric(label="Median Age of Customers", value=f"{filtered_data['demog_age'].median()} years")

# Tabs for Data Exploration, Insights, and Recommendations
tab1, tab2, tab3, tab4 = st.tabs(["Data Exploration", "Predictive Insights", "Insights", "Recommendations"])

with tab1:
    st.subheader("Data Exploration")

    # Univariate Analysis: Distribution of Sales
    st.write("### Univariate Analysis: Distribution of New Sales (INT_TGT)")
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    sns.histplot(filtered_data['int_tgt'], kde=True, ax=ax1, color='blue')
    ax1.set_title("Distribution of New Sales", fontsize=14)
    ax1.set_xlabel("New Sales (INT_TGT)", fontsize=12)
    ax1.set_ylabel("Frequency", fontsize=12)
    st.pyplot(fig1)

    # Bivariate Analysis: Scatter Plot
    st.write("### Bivariate Analysis: Sales vs. Age")
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    sns.scatterplot(x='demog_age', y='int_tgt', data=filtered_data, ax=ax2, color='green')
    ax2.set_title("Sales vs. Age", fontsize=14)
    ax2.set_xlabel("Age", fontsize=12)
    ax2.set_ylabel("New Sales (INT_TGT)", fontsize=12)
    st.pyplot(fig2)

    # Multivariate Analysis: Heatmap
    st.write("### Multivariate Analysis: Correlation Heatmap")
    # Select only numeric columns for correlation
    numeric_columns = filtered_data.select_dtypes(include=['float64', 'int64'])
    corr_matrix = numeric_columns.corr()
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", ax=ax3)
    ax3.set_title("Correlation Heatmap", fontsize=14)
    st.pyplot(fig3)

with tab2:
    st.subheader("Predictive Insights")
    input_sales = st.number_input("Enter Average Sales (RFM1):", min_value=0.0, value=50.0, step=1.0, key='predict_sales')
    predicted_sales = input_sales * 1.2  # Hypothetical multiplier for prediction
    st.write(f"Predicted New Sales: **${predicted_sales:.2f}**")
    
    st.write("""
    Use this tool to estimate sales based on historical averages. 
    Increase the predictive multiplier by analyzing past promotion impact.
    """)

with tab3:
    st.subheader("Insights")
    st.write("""
    Key findings from the analysis:
    - **Age Group Insights**: Younger customers (20-30 years) tend to have higher average new sales.
    - **Sales Correlations**: New Sales (INT_TGT) correlate strongly with promotional responsiveness.
    - **Customer Segmentation**: Female customers exhibit slightly higher average sales in specific segments.
    """)

    # Supporting Visualization: Box Plot by Gender
    st.write("### Supporting Insight: Sales by Gender")
    fig4, ax4 = plt.subplots(figsize=(8, 5))
    sns.boxplot(x='int_tgt', y='demog_genf', data=filtered_data, ax=ax4, palette='Set2')
    ax4.set_title("Sales by Gender (Female)", fontsize=14)
    ax4.set_xlabel("New Sales (INT_TGT)", fontsize=12)
    ax4.set_ylabel("Gender (Female)", fontsize=12)
    st.pyplot(fig4)

with tab4:
    st.subheader("Recommendations")
    st.write("""
    Based on the analysis, here are some actionable insights:
    - **Target Younger Customers**: Develop tailored campaigns for customers aged 20-30 to maximize sales potential.
    - **Focus on Promotions**: High responsiveness to promotions suggests that marketing campaigns can drive significant growth.
    - **Retain High-Value Customers**: Monitor customers with decreasing engagement and provide personalized offers to encourage loyalty.
    """)

# Download Filtered Data
@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

csv_data = convert_df_to_csv(filtered_data)
st.download_button(label="Download Filtered Data as CSV", 
                   data=csv_data, 
                   file_name='filtered_data.csv', 
                   mime='text/csv')
