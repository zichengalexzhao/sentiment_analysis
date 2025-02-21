# Sentiment Analysis App
# Alex Zhao

##############################################
## 1. Data Acquisition and Preprocessing
##############################################

import pandas as pd
import re
from textblob import TextBlob
import nltk

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords

# Define stopwords once
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """
    Clean the input text by removing URLs, mentions, hashtags, punctuation, and stopwords.
    """
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove Twitter mentions
    text = re.sub(r'@\w+', '', text)
    # Remove hashtags symbol (keeping the word)
    text = re.sub(r'#', '', text)
    # Remove non-alphanumeric characters (punctuation, etc.)
    text = re.sub(r'[^A-Za-z0-9\s]+', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove stopwords
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

def analyze_sentiment(text):
    """
    Analyze sentiment using TextBlob and return the polarity score.
    """
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

def label_sentiment(polarity, pos_threshold=0.1, neg_threshold=-0.1):
    """
    Label sentiment based on polarity score thresholds.
    """
    if polarity > pos_threshold:
        return 'positive'
    elif polarity < neg_threshold:
        return 'negative'
    else:
        return 'neutral'

def preprocess_data(input_file, output_file):
    """
    Read the raw tweets CSV, clean and analyze the text,
    then save the processed data to a new CSV file.
    """
    # Load the dataset
    df = pd.read_csv(input_file)
    
    # Drop duplicate tweets based on tweet_id and rows with missing text
    df.drop_duplicates(subset=['tweet_id'], inplace=True)
    df.dropna(subset=['text'], inplace=True)
    
    # Clean the tweet text
    df['clean_text'] = df['text'].apply(clean_text)
    
    # Perform sentiment analysis on the cleaned text
    df['polarity'] = df['clean_text'].apply(analyze_sentiment)
    df['computed_sentiment'] = df['polarity'].apply(label_sentiment)
    
    # Convert tweet_created to datetime for time-based analysis
    df['tweet_created'] = pd.to_datetime(df['tweet_created'], errors='coerce')
    
    # Create additional time features for visualization (if needed)
    df['tweet_date'] = df['tweet_created'].dt.date
    df['tweet_hour'] = df['tweet_created'].dt.hour
    
    # Save the processed data to a new CSV file
    df.to_csv(output_file, index=False)

# Optionally, you can run preprocessing locally by uncommenting the following block.
# if __name__ == "__main__":
#     input_csv = '/path/to/Tweets.csv'  # Adjust this path as needed
#     output_csv = 'tweets_processed.csv'
#     preprocess_data(input_csv, output_csv)


##############################################
## 2. Sentiment Analysis with NLP - Static Visualizations
##############################################

import plotly.express as px
import plotly.io as pio

# Set the renderer to 'browser' for local testing—but we won't call .show() in production.
pio.renderers.default = 'browser'

def generate_static_plots():
    """
    Generate static Plotly figures and save them as HTML files.
    (Note: .show() calls are commented out so no browser is opened.)
    """
    # Load the processed CSV file
    df = pd.read_csv('tweets_processed.csv')
    df['tweet_created'] = pd.to_datetime(df['tweet_created'], errors='coerce')
    df['tweet_date'] = df['tweet_created'].dt.date

    # 1. Overall Sentiment Distribution (Pie Chart)
    sentiment_counts = df['computed_sentiment'].value_counts().reset_index()
    sentiment_counts.columns = ['sentiment', 'count']
    fig_overall = px.pie(
        sentiment_counts,
        values='count',
        names='sentiment',
        title='Overall Sentiment Distribution'
    )
    # fig_overall.show()  # Disabled for production
    pio.write_html(fig_overall, file='overall_sentiment_distribution.html', auto_open=False)

    # 2. Sentiment Trends Over Time (Line Chart)
    sentiment_time = df.groupby(['tweet_date', 'computed_sentiment']).size().reset_index(name='count')
    fig_trend = px.line(
        sentiment_time,
        x='tweet_date',
        y='count',
        color='computed_sentiment',
        title='Sentiment Trends Over Time'
    )
    # fig_trend.show()
    pio.write_html(fig_trend, file='sentiment_trends_over_time.html', auto_open=False)

    # 3. Sentiment Distribution by Airline (Grouped Bar Chart)
    airline_sentiment = df.groupby(['airline', 'computed_sentiment']).size().reset_index(name='count')
    fig_airline = px.bar(
        airline_sentiment,
        x='airline',
        y='count',
        color='computed_sentiment',
        barmode='group',
        title='Sentiment Distribution by Airline'
    )
    # fig_airline.show()
    pio.write_html(fig_airline, file='sentiment_distribution_by_airline.html', auto_open=False)

    # 4. Distribution of Sentiment Polarity (Histogram)
    fig_polarity = px.histogram(
        df,
        x='polarity',
        nbins=30,
        title='Distribution of Sentiment Polarity'
    )
    # fig_polarity.show()
    pio.write_html(fig_polarity, file='polarity_distribution.html', auto_open=False)

# Uncomment the next line to generate static plots locally.
# generate_static_plots()


##############################################
## 3. Interactive Dashboard & Wordcloud
##############################################

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from wordcloud import WordCloud
import io, base64

# Initialize the Dash app
app = dash.Dash(__name__)
app.title = "Social Media Sentiment Analysis Dashboard"
server = app.server  # Expose the Flask server for deployment

# Load data for the dashboard
df_dash = pd.read_csv('tweets_processed.csv')
df_dash['tweet_created'] = pd.to_datetime(df_dash['tweet_created'], errors='coerce')
df_dash['tweet_date'] = df_dash['tweet_created'].dt.date

# Create list of unique airlines for the dropdown filter
airlines = df_dash['airline'].unique()
airline_options = [{'label': airline, 'value': airline} for airline in airlines]
airline_options.insert(0, {'label': 'All Airlines', 'value': 'all'})

# Define the layout of the dashboard
app.layout = html.Div([
    html.H1("Social Media Sentiment Analysis Dashboard", style={'textAlign': 'center'}),
    
    html.Div([
        html.Label("Select Airline:"),
        dcc.Dropdown(
            id='airline-dropdown',
            options=airline_options,
            value='all',
            clearable=False,
            style={'width': '50%'}
        )
    ], style={'padding': '20px', 'textAlign': 'center'}),
    
    dcc.Graph(id='sentiment-pie-chart'),
    dcc.Graph(id='sentiment-line-chart'),
    dcc.Graph(id='sentiment-airline-bar-chart'),
    dcc.Graph(id='sentiment-polarity-histogram'),
    
    html.Div([
        html.H3("Wordcloud Analysis"),
        html.Img(id='wordcloud-image', style={
            'width': '80%',
            'display': 'block',
            'margin-left': 'auto',
            'margin-right': 'auto'
        })
    ], style={'padding': '20px', 'textAlign': 'center'})
])

# Define callback to update charts and wordcloud based on the selected airline
@app.callback(
    [Output('sentiment-pie-chart', 'figure'),
     Output('sentiment-line-chart', 'figure'),
     Output('sentiment-airline-bar-chart', 'figure'),
     Output('sentiment-polarity-histogram', 'figure'),
     Output('wordcloud-image', 'src')],
    [Input('airline-dropdown', 'value')]
)
def update_charts(selected_airline):
    if selected_airline == 'all':
        df_filtered = df_dash.copy()
    else:
        df_filtered = df_dash[df_dash['airline'] == selected_airline]
    
    # 1. Overall Sentiment Distribution (Pie Chart)
    sentiment_counts = df_filtered['computed_sentiment'].value_counts().reset_index()
    sentiment_counts.columns = ['sentiment', 'count']
    fig_pie = px.pie(
        sentiment_counts,
        values='count',
        names='sentiment',
        title='Overall Sentiment Distribution'
    )
    
    # 2. Sentiment Trends Over Time (Line Chart)
    sentiment_time = df_filtered.groupby(['tweet_date', 'computed_sentiment']).size().reset_index(name='count')
    fig_line = px.line(
        sentiment_time,
        x='tweet_date',
        y='count',
        color='computed_sentiment',
        title='Sentiment Trends Over Time'
    )
    
    # 3. Sentiment Distribution by Airline (Grouped Bar Chart)
    airline_sentiment = df_filtered.groupby(['airline', 'computed_sentiment']).size().reset_index(name='count')
    fig_bar = px.bar(
        airline_sentiment,
        x='airline',
        y='count',
        color='computed_sentiment',
        barmode='group',
        title='Sentiment Distribution by Airline'
    )
    
    # 4. Distribution of Sentiment Polarity (Histogram)
    fig_hist = px.histogram(
        df_filtered,
        x='polarity',
        nbins=30,
        title='Distribution of Sentiment Polarity'
    )
    
    # 5. Wordcloud Analysis
    text_for_wc = " ".join(df_filtered["clean_text"].dropna().tolist())
    if not text_for_wc.strip():
        text_for_wc = "No text available"
    wc = WordCloud(width=800, height=400, background_color='white').generate(text_for_wc)
    img = io.BytesIO()
    wc.to_image().save(img, format='PNG')
    img.seek(0)
    wordcloud_src = "data:image/png;base64," + base64.b64encode(img.getvalue()).decode()
    
    return fig_pie, fig_line, fig_bar, fig_hist, wordcloud_src

# Only run the Dash server when this script is executed directly.
if __name__ == '__main__':
    app.run_server(debug=True)