#This file contains all the helper functions for analysis which use data processed in preprocessor.py
import emoji
from urlextract import URLExtract
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd 
from collections import Counter
import seaborn as sns
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
extract = URLExtract()
def _get_text_col(df):
    candidates = ['message', 'messages', 'msg', 'text', 'body', 'message_text', 'Message', 'Text']
    text_col = next((c for c in candidates if c in df.columns), None)
    if text_col is None:
        text_cols = [c for c in df.columns if df[c].dtype == object and c.lower() not in ('user', 'date', 'time')]
        if text_cols:
            text_col = text_cols[0]
        else:
            raise KeyError(f"No message/text column found. Available columns: {list(df.columns)}")
    return text_col

def fetch_stats(selected_user, df):
    text_col = _get_text_col(df)

    # filter by user if specified
    if selected_user != "Overall":
        if 'user' not in df.columns:
            raise KeyError("DataFrame has no 'user' column to filter by selected_user.")
        subset = df[df['user'] == selected_user]
    else:
        subset = df

    messages = subset[text_col].dropna().astype(str).str.strip()
    num_messages = int(messages.shape[0])
    words = int(messages.str.split().map(len).sum())

    media_mask = messages.str.lower().str.contains(
        r'media omitted|<media omitted>|image omitted|<image omitted>|video omitted|<video omitted>|<attached:|omitted>', na=False
    )
    num_media_messages = int(media_mask.sum())

    # correct link counting over the selected messages
    links_count = sum(len(extract.find_urls(m)) for m in messages)

    return num_messages, words, num_media_messages, links_count

def most_busy_users(df):
    x = df['user'].value_counts().head()
    percent = round((df['user'].value_counts() / df.shape[0]) * 100, 2).reset_index()
    percent = percent.rename(columns={'percent': 'name', 'count': 'percent'})
    return x, percent

def create_wordcloud(selected_user, df):
    text_col = _get_text_col(df)
    if selected_user.lower() != 'overall':
        df = df[df['user'] == selected_user]
    messages = df[text_col].dropna().astype(str)
    full_text = " ".join(messages.tolist()).strip()
    if not full_text:
        return None
    wc = WordCloud(width=500, height=500, min_font_size=2, background_color='white')
    df_wc = wc.generate(full_text)
    return df_wc
def most_common_words(selected_user, df):
    f=open('stop_hinglish.txt','r')
    stop_words=f.read()
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    temp=df[df['messages'] != 'group_notification']
    temp=df[df['messages'] != 'image omitted\n']
    words=[]
    for message in temp['messages']:
        for word in message.lower().split():
            if word not in stop_words:
                words.append(word)
    most_common_df=pd.DataFrame(Counter(words).most_common(20))
    return most_common_df
def emoji_helper(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    emojis = []
    # guard: check for messages/text column and iterate safely
    if 'messages' in df.columns:
        for message in df['messages'].dropna().astype(str):
            emojis.extend([c for c in message if emoji.is_emoji(c)])

    items = Counter(emojis).most_common()
    # return a DataFrame with explicit column names; empty DataFrame if no emojis
    if not items:
        return pd.DataFrame(columns=['emoji', 'count'])

    emoji_df = pd.DataFrame(items, columns=['emoji', 'count'])
    return emoji_df
def monthly_timeline(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    timeline = df.groupby(['year', 'month_num', 'month']).count()['messages'].reset_index()
    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i] + "-" + str(timeline['year'][i]))
    timeline['time'] = time
    return timeline
def daily_timeline(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    daily_timeline = df.groupby('date').count()['messages'].reset_index()
    return daily_timeline
def week_activity_map(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    return df['day_name'].value_counts()
def month_activity_map(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    return df['month'].value_counts()
def heatmap(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    user_heatmap = df.pivot_table(index='day_name', columns='period', values='messages', aggfunc='count').fillna(0)
    return user_heatmap
def add_sentiment_column(df, text_col='messages'):
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def add_sentiment(df, text_col):
    if text_col not in df.columns:
        raise KeyError(f"{text_col} not found in DataFrame")

    analyzer = SentimentIntensityAnalyzer()

    def _label(text):
        s = analyzer.polarity_scores(str(text))['compound']
        if s >= 0.05:
            return 'positive'
        elif s <= -0.05:
            return 'negative'
        else:
            return 'neutral'

    df['sentiment'] = df[text_col].fillna('').astype(str).apply(_label)
    return df