import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import plotly.express as px
import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt

#  PAGE CONFIG
st.set_page_config(page_title="Amazon Sentiment Analysis System", layout="wide")

# CUSTOM CSS 
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

#  TITLE 
st.markdown('<h1 style="color: black;">Amazon Sentiment Analysis System</h1>', unsafe_allow_html=True)

#  SIDEBAR MENU 
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


# HOME PAGE 

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
    st.image("https://miro.medium.com/1*_JW1JaMpK_fVGld8pd1_JQ.gif",  use_container_width=True) 





#  ANALYSIS PAGE 

elif menu == "Analysis":
    st.markdown('<div class="section-header">Amazon Review Analysis</div>', unsafe_allow_html=True)

    st.markdown("###  Upload Datasets")
    num_files = st.number_input("Select Number of Categories to Upload", min_value=1, max_value=10, value=2, step=1)
    
    uploaded_files = []
    for i in range(num_files):
        st.write(f"**Upload File {i+1}**")
        file = st.file_uploader(f"Upload CSV File for Category {i+1}", type=['csv'], key=f"file_{i}")
        category_name = st.text_input(f"Enter Category Name for File {i+1}", key=f"cat_{i}")
        if file and category_name:
            uploaded_files.append((file, category_name))

    column_name = st.text_input("Enter Column Name containing review text (e.g., 'reviewText'):")
    analyze_btn = st.button("Run Sentiment Analysis")

    if analyze_btn:
        if not uploaded_files or not column_name:
            st.warning(" Please upload at least one file and provide the column name.")
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

                for file, category in uploaded_files:
                    st.info(f"Analyzing {category} Reviews...")
                    analyzed_df = analyze_reviews(file, category)
                    if analyzed_df is not None:
                        all_dfs.append(analyzed_df)

                if all_dfs:
                    combined_df = pd.concat(all_dfs, ignore_index=True)
                    st.success(" Sentiment analysis completed successfully!")

                    # Summary
                    st.markdown("###  Sentiment Summary")
                    summary = combined_df.groupby(['Category', 'Sentiment']).size().reset_index(name='Count')
                    st.dataframe(summary)

                    total_reviews = combined_df.groupby('Category').size()
                    for cat in total_reviews.index:
                        st.write(f"**{cat} Reviews:** {total_reviews[cat]} total")

                    # Download results
                    csv = combined_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label=" Download Combined Results (CSV)",
                        data=csv,
                        file_name='amazon_sentiment_results.csv',
                        mime='text/csv'
                    )

                    # Comparative Bar Chart
                    st.markdown("###  Sentiment Comparison by Category")
                    fig = px.bar(
                        summary, 
                        x='Category', y='Count', color='Sentiment', 
                        barmode='group', title="Sentiment Distribution per Category",
                        color_discrete_sequence=px.colors.qualitative.Set2
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Word Clouds
                    st.markdown("### ️ Word Clouds per Category")
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

                    st.session_state['combined_df'] = combined_df

            except Exception as e:
                st.error(f"Error: {e}")



# VISUALIZATION PAGE

elif menu == "Visualization":
    st.markdown('<div class="section-header">Comparative Visualization</div>', unsafe_allow_html=True)

    if 'combined_df' not in st.session_state:
        uploaded_file = st.file_uploader("Upload the analyzed CSV file", type=['csv'])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
        else:
            st.info("Please upload or analyze data first from the 'Analysis' page.")
            st.stop()
    else:
        df = st.session_state['combined_df']

    
    st.markdown("###  Filter Options")
    categories = st.multiselect("Select Categories", df['Category'].unique(), default=list(df['Category'].unique()))
    sentiments = st.multiselect("Select Sentiments", df['Sentiment'].unique(), default=list(df['Sentiment'].unique()))
    filtered_df = df[(df['Category'].isin(categories)) & (df['Sentiment'].isin(sentiments))]

    if filtered_df.empty:
        st.warning(" No data available for the selected filters.")
        st.stop()

    
    vis_option = st.selectbox(
        "Select Visualization Type",
        ["Table", "Pie Chart", "Bar Chart", "Stacked Bar Chart", "Category Comparison", "Box Plot", "Scatter Plot"]
    )

    # Table
    if vis_option == "Table":
        st.dataframe(filtered_df)

    # Pie Chart
    elif vis_option == "Pie Chart":
        st.markdown("###  Sentiment Distribution")
        sentiment_counts = filtered_df['Sentiment'].value_counts().reset_index()
        sentiment_counts.columns = ['Sentiment', 'Count']
        fig = px.pie(sentiment_counts, names='Sentiment', values='Count', title="Overall Sentiment Distribution")
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)

    # Bar Chart
    elif vis_option == "Bar Chart":
        sentiment_counts = filtered_df.groupby(['Category', 'Sentiment']).size().reset_index(name='Count')
        fig = px.bar(sentiment_counts, x='Category', y='Count', color='Sentiment', barmode='group',
                     title="Sentiment Count by Category")
        st.plotly_chart(fig, use_container_width=True)

    #  Stacked Bar Chart
    elif vis_option == "Stacked Bar Chart":
        sentiment_counts = filtered_df.groupby(['Category', 'Sentiment']).size().reset_index(name='Count')
        fig = px.bar(sentiment_counts, x='Category', y='Count', color='Sentiment', barmode='stack',
                     title="Stacked Sentiment Distribution per Category")
        st.plotly_chart(fig, use_container_width=True)

    # Category Comparison 
    elif vis_option == "Category Comparison":
        st.markdown("### ️ Sentiment Percentage Comparison")
        sentiment_summary = filtered_df.groupby(['Category', 'Sentiment']).size().reset_index(name='Count')
        total_counts = sentiment_summary.groupby('Category')['Count'].sum().reset_index(name='Total')
        sentiment_summary = sentiment_summary.merge(total_counts, on='Category')
        sentiment_summary['Percentage'] = round((sentiment_summary['Count'] / sentiment_summary['Total']) * 100, 2)

        fig = px.bar(sentiment_summary, x='Category', y='Percentage', color='Sentiment', 
                     barmode='group', text='Percentage',
                     title="Sentiment Percentage by Category")
        fig.update_traces(texttemplate='%{text}%', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("###  Detailed Sentiment Breakdown")
        st.dataframe(sentiment_summary)

    #  Box Plot 
    elif vis_option == "Box Plot":
        numeric_columns = filtered_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        if numeric_columns:
            column = st.selectbox("Select Numeric Column for Box Plot", numeric_columns)
            fig = px.box(filtered_df, x='Category', y=column, color='Sentiment',
                         title=f"Box Plot of {column} by Category and Sentiment")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No numeric columns available for Box Plot.")

    # Scatter Plot
    elif vis_option == "Scatter Plot":
        numeric_columns = filtered_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        if len(numeric_columns) >= 2:
            x_col = st.selectbox("Select X-axis Column", numeric_columns, index=0)
            y_col = st.selectbox("Select Y-axis Column", numeric_columns, index=1)
            fig = px.scatter(filtered_df, x=x_col, y=y_col, color='Sentiment', symbol='Category',
                             title=f"Scatter Plot of {y_col} vs {x_col}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("At least two numeric columns are required for Scatter Plot.")




