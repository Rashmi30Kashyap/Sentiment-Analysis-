import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer



df=pd.read_csv("https://docs.google.com/spreadsheets/d/1C8FdK5HWTIcDrGSb1x-8TQkv6_4UJ9rBZlOuS_phWqY/export?format=csv&usp=sharing")
x=df['summary']
mymodel=SentimentIntensityAnalyzer()
l=[]
for k in x:
    pred=mymodel.polarity_scores(k)
    if pred['compound']>0.01:
        l.append("Positive")
    elif pred['compound']< -0.01:
        l.append(' Negative')
    else:
        l.append('Neutral')
df['Sentiment']=l

print(df)
df.to_csv("/Users/akshitkashyap/Desktop/RASHMI/Job_projects/projects/project_3/results.csv",index=False)
















































 #######  comparison between two files


import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import plotly.express as px
import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# ------------------- PAGE CONFIG -------------------
st.set_page_config(page_title="Amazon Sentiment Analysis System", layout="wide")

# ------------------- CUSTOM CSS -------------------
st.markdown("""
    <style>
    .section-header {
        color: green;
        font-size: 28px;
        font-weight: bold;
        margin-top: 20px;
    }
    .stButton>button {
        background-color: green;
        color: white;
    }
    .home-description {
        font-size: 18px;
        line-height: 1.6;
        color: #333;
        background-color: rgba(255, 255, 255, 0.8);
        padding: 20px;
        border-radius: 8px;
    }
    .stApp {
        background-image: url("https://previews.123rf.com/images/romvo/romvo1003/romvo100300108/6637000-simple-abstract-blue-background-for-design.jpg");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------- TITLE -------------------
st.markdown('<h1 style="color: black;">Amazon Sentiment Analysis System</h1>', unsafe_allow_html=True)

# ------------------- SIDEBAR MENU -------------------
st.sidebar.markdown('<div class="sidebar-heading">Navigation Menu</div>', unsafe_allow_html=True)
menu = st.sidebar.selectbox("", ["Home", "Analysis", "Visualization"])

st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        background-image: url("https://w0.peakpx.com/wallpaper/852/928/HD-wallpaper-green-blur-blur-color-design-green-thumbnail.jpg");
        background-size: cover;
        background-position: center;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        font-weight: bold;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# ==============================================================
# ------------------- HOME PAGE -------------------
# ==============================================================
if menu == "Home":
    st.markdown('<h2 style="color: green; font-weight: bold;">Welcome to the Amazon Sentiment Analysis System!</h2>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Home</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="home-description">
        <p>This system helps you compare customer sentiments from <b>Amazon Fashion</b> and <b>Amazon Electronics</b> reviews.</p>
        <p>It classifies each review into Positive, Neutral, or Negative sentiment using the VADER sentiment analyzer.</p>
        <p>You can upload two CSV files — one for each category — to explore and visualize the emotional tone of customer feedback.</p>
        <p>Visual insights include sentiment comparison charts, category-wise word clouds, and interactive analysis tools.</p>
    </div>
    """, unsafe_allow_html=True)
    st.image("https://miro.medium.com/1*_JW1JaMpK_fVGld8pd1_JQ.gif", use_container_width=True)


# ==============================================================
# ------------------- ANALYSIS PAGE -------------------
# ==============================================================
elif menu == "Analysis":
    st.markdown('<div class="section-header">Amazon Review Analysis</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        fashion_file = st.file_uploader("Upload Amazon Fashion CSV", type=['csv'])
    with col2:
        electronics_file = st.file_uploader("Upload Amazon Electronics CSV", type=['csv'])

    column_name = st.text_input("Enter Column Name containing review text (e.g., 'reviewText'):")
    analyze_btn = st.button("Run Comparative Analysis")

    if analyze_btn:
        if not column_name or not (fashion_file or electronics_file):
            st.warning("Please upload at least one file and provide column name.")
        else:
            try:
                analyzer = SentimentIntensityAnalyzer()
                all_dfs = []

                def analyze_reviews(file, category):
                    df = pd.read_csv(file)
                    if column_name not in df.columns:
                        st.error(f"'{column_name}' not found in {category} dataset.")
                        return None
                    df['Category'] = category
                    sentiments = []
                    progress = st.progress(0)
                    for i, text in enumerate(df[column_name]):
                        pred = analyzer.polarity_scores(str(text))
                        if pred['compound'] > 0.01:
                            sentiments.append("Positive")
                        elif pred['compound'] < -0.01:
                            sentiments.append("Negative")
                        else:
                            sentiments.append("Neutral")
                        progress.progress((i + 1) / len(df))
                    df['Sentiment'] = sentiments
                    return df

                if fashion_file:
                    st.info("Analyzing Amazon Fashion Reviews...")
                    fashion_df = analyze_reviews(fashion_file, "Fashion")
                    if fashion_df is not None:
                        all_dfs.append(fashion_df)

                if electronics_file:
                    st.info("Analyzing Amazon Electronics Reviews...")
                    electronics_df = analyze_reviews(electronics_file, "Electronics")
                    if electronics_df is not None:
                        all_dfs.append(electronics_df)

                if all_dfs:
                    combined_df = pd.concat(all_dfs, ignore_index=True)
                    st.success("Analysis completed successfully!")

                    # Summary Section
                    st.markdown("### 📊 Sentiment Summary")
                    summary = combined_df.groupby(['Category', 'Sentiment']).size().reset_index(name='Count')
                    st.dataframe(summary)

                    total_reviews = combined_df.groupby('Category').size()
                    for cat in total_reviews.index:
                        st.write(f"**{cat} Reviews:** {total_reviews[cat]} total")

                    # Download results
                    csv = combined_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Combined Results (CSV)",
                        data=csv,
                        file_name='amazon_sentiment_results.csv',
                        mime='text/csv'
                    )

                    # Comparative Bar Chart
                    st.markdown("### 📈 Sentiment Comparison by Category")
                    fig = px.bar(
                        summary, 
                        x='Category', y='Count', color='Sentiment', 
                        barmode='group', title="Sentiment Distribution per Category",
                        color_discrete_sequence=px.colors.qualitative.Set2
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Word Clouds
                    st.markdown("### ☁️ Word Clouds per Category")
                    for cat in combined_df['Category'].unique():
                        st.subheader(f"{cat} Reviews Word Clouds")
                        pos_text = ' '.join(combined_df[(combined_df['Category']==cat) & (combined_df['Sentiment']=='Positive')][column_name].astype(str))
                        neg_text = ' '.join(combined_df[(combined_df['Category']==cat) & (combined_df['Sentiment']=='Negative')][column_name].astype(str))
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"Positive ({cat})")
                            if pos_text.strip():
                                wc_pos = WordCloud(width=400, height=300, background_color='white').generate(pos_text)
                                fig, ax = plt.subplots()
                                ax.imshow(wc_pos, interpolation='bilinear')
                                ax.axis('off')
                                st.pyplot(fig)
                            else:
                                st.info("No positive reviews available.")
                        with col2:
                            st.write(f"Negative ({cat})")
                            if neg_text.strip():
                                wc_neg = WordCloud(width=400, height=300, background_color='white').generate(neg_text)
                                fig, ax = plt.subplots()
                                ax.imshow(wc_neg, interpolation='bilinear')
                                ax.axis('off')
                                st.pyplot(fig)
                            else:
                                st.info("No negative reviews available.")

                    # Save for Visualization
                    st.session_state['combined_df'] = combined_df

            except Exception as e:
                st.error(f"Error: {e}")


# ==============================================================
# ------------------- VISUALIZATION PAGE -------------------
# ==============================================================
elif menu == "Visualization":
    st.markdown('<div class="section-header">Visualization</div>', unsafe_allow_html=True)

    if 'combined_df' not in st.session_state:
        uploaded_file = st.file_uploader("Upload the analyzed CSV file", type=['csv'])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
        else:
            st.info("Please upload or analyze data first from the 'Analysis' page.")
            st.stop()
    else:
        df = st.session_state['combined_df']

    # Filter Options
    st.markdown("### 🔍 Filter Options")
    categories = st.multiselect("Select Categories", df['Category'].unique(), default=list(df['Category'].unique()))
    sentiments = st.multiselect("Select Sentiments", df['Sentiment'].unique(), default=list(df['Sentiment'].unique()))
    filtered_df = df[(df['Category'].isin(categories)) & (df['Sentiment'].isin(sentiments))]

    # Visualization Selector
    vis_option = st.selectbox(
        "Select Visualization Type",
        ["Table", "Pie Chart", "Histogram", "Bar Chart", "Box Plot", "Scatter Plot"]
    )

    if vis_option == "Table":
        st.dataframe(filtered_df)

    elif vis_option == "Pie Chart":
        sentiment_counts = filtered_df['Sentiment'].value_counts().reset_index()
        sentiment_counts.columns = ['Sentiment', 'Count']
        fig = px.pie(sentiment_counts, names='Sentiment', values='Count', title="Overall Sentiment Distribution")
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)

    elif vis_option == "Histogram":
        column = st.selectbox("Select Column for Histogram", filtered_df.columns)
        fig = px.histogram(filtered_df, x=column, color='Sentiment', barmode='group', title=f"Histogram of {column} by Sentiment")
        st.plotly_chart(fig, use_container_width=True)

    elif vis_option == "Bar Chart":
        sentiment_counts = filtered_df.groupby(['Category', 'Sentiment']).size().reset_index(name='Count')
        fig = px.bar(sentiment_counts, x='Category', y='Count', color='Sentiment', barmode='group', title="Sentiment Count by Category")
        st.plotly_chart(fig, use_container_width=True)

    elif vis_option == "Box Plot":
        numeric_columns = filtered_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        if numeric_columns:
            column = st.selectbox("Select Numeric Column for Box Plot", numeric_columns)
            fig = px.box(filtered_df, x='Sentiment', y=column, color='Category', title=f"Box Plot of {column} by Sentiment & Category")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No numeric columns available for Box Plot.")

    elif vis_option == "Scatter Plot":
        numeric_columns = filtered_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        if len(numeric_columns) >= 2:
            x_col = st.selectbox("Select X-axis Column", numeric_columns, index=0)
            y_col = st.selectbox("Select Y-axis Column", numeric_columns, index=1)
            fig = px.scatter(filtered_df, x=x_col, y=y_col, color='Sentiment', symbol='Category', title=f"Scatter Plot of {y_col} vs {x_col}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("At least two numeric columns are required for Scatter Plot.")










































































#################  dashboard and  different analysis page 




# sentiment_app.py

import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# -----------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------
st.set_page_config(
    page_title="Luxury Product Reviews Sentiment Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------
# DOWNLOAD VADER LEXICON
# -----------------------------------------------------
nltk.download('vader_lexicon', quiet=True)

# -----------------------------------------------------
# LOAD AND PROCESS DATA
# -----------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("/Users/akshitkashyap/Desktop/RASHMI/Job_projects/projects/project_3/luxury_results.csv")

    # Fill missing text
    df['reviewText'] = df['reviewText'].fillna("")

    # Initialize VADER analyzer
    sia = SentimentIntensityAnalyzer()

    # Compute sentiment scores
    df['compound_score'] = df['reviewText'].apply(lambda x: sia.polarity_scores(x)['compound'])
    df['Sentiment'] = df['compound_score'].apply(
        lambda c: 'Positive' if c >= 0.05 else ('Negative' if c <= -0.05 else 'Neutral')
    )

    # Clean text for WordCloud
    df['clean_text'] = df['reviewText'].str.lower().str.replace(r'[^a-z\s]', '', regex=True)

    return df


df = load_data()

# -----------------------------------------------------
# SIDEBAR NAVIGATION
# -----------------------------------------------------
st.sidebar.title("🧭 Navigation")
selected = st.sidebar.radio(
    "Go to",
    ["Dashboard", "Word Cloud", "Sentiment by Product", "Category Analysis", "Insights & Analysis", "Raw Data"]
)

st.sidebar.markdown("---")
st.sidebar.info("📊 Analyze luxury product reviews using VADER sentiment analysis and interactive visuals.")

# -----------------------------------------------------
# DASHBOARD PAGE
# -----------------------------------------------------
if selected == "Dashboard":
    st.title("💎 Luxury Product Reviews — Sentiment Dashboard")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Reviews", len(df))
    with col2:
        pos = df[df['Sentiment'] == 'Positive'].shape[0]
        st.metric("Positive Reviews", pos)
    with col3:
        neg = df[df['Sentiment'] == 'Negative'].shape[0]
        st.metric("Negative Reviews", neg)

    # Sentiment Distribution (Bar + Pie)
    st.subheader("📊 Sentiment Distribution")

    sentiment_counts = df['Sentiment'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment', 'count']

    col_a, col_b = st.columns(2)

    with col_a:
        fig_bar = px.bar(
            sentiment_counts,
            x='Sentiment',
            y='count',
            color='Sentiment',
            text='count',
            title='Overall Sentiment (Bar Chart)',
            labels={'count': 'Number of Reviews'}
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_b:
        fig_pie = px.pie(
            sentiment_counts,
            names='Sentiment',
            values='count',
            title='Overall Sentiment (Pie Chart)',
            color='Sentiment',
            hole=0.4
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # Rating vs Sentiment
    if 'overall' in df.columns:
        st.subheader("⭐ Rating Distribution by Sentiment")
        fig_hist = px.histogram(
            df,
            x='overall',
            color='Sentiment',
            nbins=10,
            barmode='overlay',
            title="Histogram of Ratings by Sentiment",
            labels={'overall': 'Rating'}
        )
        st.plotly_chart(fig_hist, use_container_width=True)

# -----------------------------------------------------
# WORD CLOUD PAGE
# -----------------------------------------------------
elif selected == "Word Cloud":
    st.title("☁️ Word Cloud by Sentiment")

    sentiment_choice = st.selectbox("Select Sentiment", ["Positive", "Negative", "Neutral"])
    text = " ".join(df[df['Sentiment'] == sentiment_choice]['clean_text'].dropna())

    if text.strip():
        wordcloud = WordCloud(width=1000, height=500, background_color='white').generate(text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        st.pyplot(plt)
    else:
        st.warning("No text found for this sentiment.")

# -----------------------------------------------------
# SENTIMENT BY PRODUCT
# -----------------------------------------------------
elif selected == "Sentiment by Product":
    st.title("👜 Product-wise Sentiment Analysis")

    if 'asin' in df.columns:
        product_counts = df.groupby(['asin', 'Sentiment']).size().reset_index(name='count')
        fig3 = px.bar(
            product_counts,
            x='asin',
            y='count',
            color='Sentiment',
            title="Sentiment by Product (ASIN)",
            labels={'asin': 'Product ASIN', 'count': 'Number of Reviews'}
        )
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.warning("No 'asin' column found in the dataset.")

# -----------------------------------------------------
# CATEGORY ANALYSIS
# -----------------------------------------------------
elif selected == "Category Analysis":
    st.title("🧩 Category-wise Sentiment Analysis")

    if 'category' in df.columns:
        category_counts = df.groupby(['category', 'Sentiment']).size().reset_index(name='count')

        fig_cat = px.bar(
            category_counts,
            x='category',
            y='count',
            color='Sentiment',
            title="Sentiment Distribution by Product Category",
            labels={'category': 'Category', 'count': 'Number of Reviews'},
            barmode='group'
        )
        st.plotly_chart(fig_cat, use_container_width=True)

        # Pie chart per category (optional)
        selected_category = st.selectbox("Select Category for Pie Chart", df['category'].unique())
        subset = category_counts[category_counts['category'] == selected_category]
        fig_pie_cat = px.pie(
            subset,
            names='Sentiment',
            values='count',
            title=f"Sentiment Split — {selected_category}",
            hole=0.4
        )
        st.plotly_chart(fig_pie_cat, use_container_width=True)
    else:
        st.warning("No 'category' column found in the dataset.")

# -----------------------------------------------------
# INSIGHTS & ANALYSIS PAGE
# -----------------------------------------------------
elif selected == "Insights & Analysis":
    st.title("🧠 Insights & Sentiment Analysis Summary")

    st.markdown("### 🔹 Key Summary Statistics")
    avg_rating = round(df['overall'].mean(), 2) if 'overall' in df.columns else None
    pos_ratio = round((df[df['Sentiment'] == 'Positive'].shape[0] / len(df)) * 100, 2)
    neg_ratio = round((df[df['Sentiment'] == 'Negative'].shape[0] / len(df)) * 100, 2)
    neu_ratio = round((df[df['Sentiment'] == 'Neutral'].shape[0] / len(df)) * 100, 2)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Reviews", len(df))
    if avg_rating:
        col2.metric("Average Rating", avg_rating)
    col3.metric("Positive %", f"{pos_ratio}%")
    col4.metric("Negative %", f"{neg_ratio}%")

    st.markdown("---")

    # Average rating by sentiment
    if 'overall' in df.columns:
        st.subheader("⭐ Average Rating by Sentiment")
        avg_rating_by_sentiment = df.groupby('Sentiment')['overall'].mean().reset_index()
        fig_avg = px.bar(
            avg_rating_by_sentiment,
            x='Sentiment',
            y='overall',
            color='Sentiment',
            text='overall',
            title="Average Rating by Sentiment",
            labels={'overall': 'Average Rating'}
        )
        st.plotly_chart(fig_avg, use_container_width=True)

    # Top words or products
    st.subheader("🔝 Most Reviewed Products")
    if 'asin' in df.columns:
        top_products = df['asin'].value_counts().head(10).reset_index()
        top_products.columns = ['ASIN', 'Review Count']
        st.dataframe(top_products)

# -----------------------------------------------------
# RAW DATA PAGE
# -----------------------------------------------------
elif selected == "Raw Data":
    st.title("🧾 View Raw Data")
    st.dataframe(df)

    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download CSV", csv, "luxury_results_with_sentiment.csv", "text/csv")






















