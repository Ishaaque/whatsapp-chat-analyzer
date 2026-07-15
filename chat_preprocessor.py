#This file preprocesses the raw chat data into a structured format
import re
import pandas as pd

def preprocess(data: str) -> pd.DataFrame:
    # Regex pattern for WhatsApp date-time format
    pattern = r'\d{1,2}\/\d{1,2}\/\d{2},\s\d{1,2}:\d{2}\s(?:AM|PM)'
    
    # Split messages and extract dates
    messages = re.split(pattern, data)[1:]
    dates = re.findall(pattern, data)

    # Create DataFrame
    df = pd.DataFrame({'user_message': messages, "message_date": dates})

    # Clean and convert date: ensure string dtype before using .str accessor
    df['message_date'] = df['message_date'].fillna('').astype(str).str.replace('\u202f', ' ', regex=True)
    # Parse datetimes; coerce errors to NaT to avoid exceptions for malformed entries
    df['message_date'] = pd.to_datetime(df['message_date'], format='%m/%d/%y, %I:%M %p', errors='coerce')
    df.rename(columns={'message_date': 'date'}, inplace=True)

    # Clean user messages
    df['user_message'] = df['user_message'].fillna('').astype(str).str.lstrip('- ').str.strip()


    # Separate users and messages
    users = []
    messages_list = []
    for message in df['user_message']:
        # Capture any sender name before the first colon, including unicode and punctuation
        entry = re.split(r'^([^:]+):\s(.*)', message)
        if len(entry) > 2:
            users.append(entry[1].strip())
            messages_list.append(entry[2])
        else:
            users.append('group_notification')
            messages_list.append(entry[0])

    # Add to DataFrame
    df['user'] = users
    df['messages'] = messages_list
    df.drop(columns=['user_message'], inplace=True)

    # Extract datetime components
    df['year'] = df['date'].dt.year
    df['month_num'] = df['date'].dt.month
    df['month'] = df['date'].dt.month_name()
    df['day'] = df['date'].dt.day
    df['day_name'] = df['date'].dt.day_name()
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute
    period=[]
    for hour in df[['day_name','hour',]]['hour']:
        if hour==23:
            period.append(str(hour) + "=" + str('00'))
        elif hour==0:
            period.append(str('00')+ "-" + str(hour+1))
        else:
            period.append(str(hour) + "-" + str(hour+1))
    df['period'] = period

    return df
