import requests
import pandas as pd
from datetime import datetime, timedelta

end_date = datetime.now().date() - timedelta(days=1)
start_date = end_date - timedelta(days=180)

url = 'https://archive-api.open-meteo.com/v1/archive'
params = {
    'latitude': 35.55,
    'longitude': 139.78,
    'start_date': start_date.isoformat(),
    'end_date': end_date.isoformat(),
    'daily': 'temperature_2m_max',
    'timezone': 'Asia/Tokyo'
}

resp = requests.get(url, params=params, timeout=30)
data = resp.json()

df = pd.DataFrame({
    'date': data['daily']['time'],
    'actual_max_temp': data['daily']['temperature_2m_max']
})
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')

df.to_csv('/root/hermes_research/rjtt_project/historical_temps_6months.csv')
print(f'Saved historical temps: {len(df)} days')
print(df['actual_max_temp'].describe())
