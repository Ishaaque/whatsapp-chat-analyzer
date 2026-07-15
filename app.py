#This file renders everything on the UI using streamlit
import streamlit as st
import chat_preprocessor as preprocessor
import helper as hp
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import font_manager as fm      
import seaborn as sns
st.set_page_config(
    page_title="Whatsapp Chat Analyzer",
    page_icon="📞",
)

mpl.rcParams['font.family'] = ["Segoe UI Emoji",
    "Segoe UI Symbol",
    "DejaVu Sans"]
mpl.rcParams['axes.unicode_minus'] = False

st.sidebar.title("Whatsapp Chat Analyzer")

uploaded_file = st.sidebar.file_uploader("Choose a file")
st.sidebar.write("How to use this app?")
st.sidebar.write("1. Go to your WhatsApp group chat or private chat. On top right corner, click on the three dots to open the menu. Select 'More' and then 'Export Chat'. Choose 'Without Media' option to export the chat.")
st.sidebar.write("2. A  zip file will be downloaded. Extract the zip file to get the .txt chat file.")
st.sidebar.write("3. Upload the .txt file in this app.")
st.sidebar.write("4. Select the user from the dropdown to analyze individual chat or select 'Overall' to analyze the whole chat.")
st.sidebar.write("5. Click on 'Show Analysis' button to see the statistics and visualizations.")
st.sidebar.write("Enjoy analyzing your WhatsApp chats!")


if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")
    df = preprocessor.preprocess(data)
    # after df = preprocessor.preprocess(data)
    df = hp.add_sentiment(df, 'messages')  
    user_list = df['user'].dropna().unique().tolist()
    user_list = [user for user in user_list if user != 'group_notification']
    user_list.sort()
    user_list.insert(0, 'Overall')
    # capture selected user
    selected_user = st.sidebar.selectbox("Select User", user_list)

   
    if st.sidebar.button("Show Analysis"):
        num_messages,words,num_media_messages,num_links=hp.fetch_stats(selected_user,df)
        st.title("Top Statistics")
        col1,col2,col3,col4 = st.columns(4)
    
        with col1:
            st.header("Total Messages")
            st.title(str(num_messages))
        with col2:
            st.header("Total Words")
            st.title(str(words))
        with col3:
            st.header("Media Shared")
            st.title(str(num_media_messages))
        with col4:
            st.header("Links Shared")
            st.title(str(num_links))
        timeline=hp.monthly_timeline(selected_user,df)
        st.title("Monthly Timeline")
        fig,ax=plt.subplots()
        ax.plot(timeline['time'],timeline['messages'],color='green')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)
        daily_timeline=hp.daily_timeline(selected_user,df)
        st.title("Daily Timeline")
        fig,ax=plt.subplots()
        ax.plot(daily_timeline['date'],daily_timeline['messages'],color='black')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)
         #activity map
        st.title("Activity Map")
        col1,col2=st.columns(2)
        with col1:
            st.header("Most Busy Day")
            busy_day=hp.week_activity_map(selected_user,df)
            fig,ax=plt.subplots()
            ax.bar(busy_day.index,busy_day.values,color='purple')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        with col2:
            st.header("Most Busy Month")
            busy_month=hp.month_activity_map(selected_user,df)
            fig,ax=plt.subplots()
            ax.bar(busy_month.index,busy_month.values,color='orange')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        #finding the busiest user in the group (if overall is selected)
        if selected_user == 'Overall':
           st.title('Most Busy User')
           x,new_df= hp.most_busy_users(df)
           fig,ax=plt.subplots()
           col1,col2=st.columns(2)
           with col1:
               ax.bar(x.index, x.values,color='red')
               plt.xticks(rotation='vertical')
               st.pyplot(fig)
           with col2:
               st.dataframe(new_df)
        #wordcloud
        st.title("Weekly Activity Map")
        user_heatmap=hp.heatmap(selected_user,df)
        fig,ax=plt.subplots()
        ax=sns.heatmap(user_heatmap)
        st.pyplot(fig)
        st.title("Wordcloud")
        wc=hp.create_wordcloud(selected_user,df)
        fig,ax=plt.subplots()
        ax.imshow(wc)
        st.pyplot(fig)
        #most common words
        st.title("Most Common Words")
        most_common_df=hp.most_common_words(selected_user,df)
        fig,ax=plt.subplots()
        ax.barh(most_common_df[0],most_common_df[1])
        plt.xticks(rotation='vertical')
        st.pyplot(fig)
        emoji_df=hp.emoji_helper(selected_user,df)
        st.title("Emoji Analysis")
        col1,col2=st.columns(2)
        with col1:
            if emoji_df is None or emoji_df.empty:
                st.write("No emojis found for the selected user.")
            else:
                st.dataframe(emoji_df)
        with col2:
            if emoji_df is None or emoji_df.empty:
                st.write("No emoji data to plot.")
            else:
                fig,ax=plt.subplots()
                ax.pie(emoji_df['count'].head(), labels=emoji_df['emoji'].head(), autopct="%0.2f")
                st.pyplot(fig)
        #sentiment analysis
        st.title("Sentiment Analysis")
        sent = df if selected_user=='Overall' else df[df['user']==selected_user]
        sent_counts = sent['sentiment'].value_counts().reindex(['positive','neutral','negative']).fillna(0)
        fig,ax = plt.subplots()
        ax.pie(sent_counts, labels=sent_counts.index, autopct='%0.1f%%', colors=['#2ca02c','#7f7f7f','#d62728'])
        st.pyplot(fig)
