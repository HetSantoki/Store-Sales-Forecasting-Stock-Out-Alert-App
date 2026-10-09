import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

st.set_page_config(page_title='Store Inventory & Stock Prediction Dashboard', layout='wide')


def load_data():
    df = pd.read_csv('Data/train.csv', usecols=['date', 'store_nbr', 'family', 'sales'])
    df['date'] = pd.to_datetime(df['date'])
    df = df.rename(columns={'family': 'Category'})
    df = df[df['store_nbr'].isin([44, 45, 47, 3, 49])
            & df['Category'].isin(['GROCERY I', 'BEVERAGES', 'PRODUCE', 'CLEANING', 'DAIRY'])]
    df = df.sort_values(by=['store_nbr', 'Category', 'date']).reset_index(drop=True)
    return df


df = load_data()
models_dict = joblib.load('model.pkl') 

st.title('Retail Store Inventory & Stock Prediction Dashboard')
st.markdown(
    'Forecast the next 7, 14 or 30 days of sales with a **Holt-Winters** model '
    'and check whether the current stock is enough.'
)

if not models_dict:
    st.error('model.pkl not found. Run section 9 of the notebook first to create it, '
             'and keep it in the same folder as app.py.')
    st.stop()

st.sidebar.header('Inventory Parameters')

top_stores = [44, 45, 47, 3, 49]
top_categories = ['GROCERY I', 'BEVERAGES', 'PRODUCE', 'CLEANING', 'DAIRY']

selected_store = st.sidebar.selectbox('Select Store Number', top_stores, index=0)
selected_category = st.sidebar.selectbox('Select Category', top_categories, index=0)
current_stock = st.sidebar.number_input('Enter Current Stock (Units)', min_value=0.0, value=200000.0, step=1000.0)
forecast_days = st.sidebar.selectbox('Select Future Prediction (Days)', [7, 14, 30], index=2)

predict_clicked = st.sidebar.button('Predict', type='primary')

if predict_clicked:
    model = models_dict[(selected_store, selected_category)]

    last_date = model.fittedvalues.index[-1]
    start = len(model.fittedvalues)
    result = model.get_prediction(start=start, end=start + forecast_days - 1)
    forecast = result.summary_frame(alpha=0.20)
    forecast.index = pd.date_range(last_date + pd.Timedelta(days=1), periods=forecast_days)
    forecast = forecast.rename(columns={'mean': 'Forecast', 'pi_lower': 'Lower', 'pi_upper': 'Upper'})
    forecast = forecast[['Forecast', 'Lower', 'Upper']].clip(lower=0)

    predicted_demand = float(forecast['Forecast'].sum())
    high_demand = float(forecast['Upper'].sum())
    stock_difference = current_stock - predicted_demand

    st.markdown(f'#### INVENTORY CHECK FOR STORE ({selected_store}) & CATEGORY ({selected_category})')
    st.caption(f'Forecast period: {forecast.index[0]:%d %b %Y} to {forecast.index[-1]:%d %b %Y}')

    col1, col2, col3 = st.columns(3)
    col1.metric('Current Stock', f'{current_stock:,.0f} units')
    col2.metric(f'{forecast_days}-Day Demand (Model Predicted)', f'{predicted_demand:,.0f} units')

    if stock_difference < 0:
        col3.metric('Stock Status', 'Deficit', delta=f'{stock_difference:,.0f} units', delta_color='inverse')
    else:
        col3.metric('Stock Status', 'Surplus', delta=f'+{stock_difference:,.0f} units', delta_color='normal')

    forecast['Cumulative'] = forecast['Forecast'].cumsum()
    run_out = forecast[forecast['Cumulative'] > current_stock]

    if stock_difference < 0:
        st.error(
            f'🚨 **ALERT - STOCK NOT SUFFICIENT:** deficit of **{abs(stock_difference):,.0f} units** '
            f'over the next {forecast_days} days. Restock required. '
            f'Stock is expected to run out around **{run_out.index[0]:%d %b %Y}**.')
    elif current_stock < high_demand:
        st.warning(
            f'⚠️ **STOCK IS SUFFICIENT, BUT AT RISK:** it covers the expected demand '
            f'(surplus {stock_difference:,.0f} units) but not the high-demand case of '
            f'{high_demand:,.0f} units (80% upper limit).')
    else:
        st.success(
            f'✅ **STOCK IS SUFFICIENT:** surplus of **{stock_difference:,.0f} units**, '
            f'enough even for the high-demand case ({high_demand:,.0f} units). No restock required.')

    st.markdown('---')
    st.subheader('📈 Sales Forecast with Confidence Interval')

    target_df = df[(df['store_nbr'] == selected_store) & (df['Category'] == selected_category)].copy()
    target_df['day_name'] = target_df['date'].dt.day_name()
    recent = target_df.set_index('date')['sales'].tail(60)

    tab1, tab2, tab3 = st.tabs(['Daily Forecast', 'Cumulative Demand vs Stock', 'Forecast Table'])

    with tab1:
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(recent.index, recent.values, color='blue', label='Actual (last 60 days)')
        ax.plot(forecast.index, forecast['Forecast'], '--', color='orange', label='Forecast')
        ax.fill_between(forecast.index, forecast['Lower'], forecast['Upper'],
                        color='orange', alpha=0.2, label='80% confidence interval')
        ax.set_title(f'Store {selected_store} - {selected_category}: {forecast_days}-Day Forecast')
        ax.set_xlabel('Date')
        ax.set_ylabel('Sales per day')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

    with tab2:
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(forecast.index, forecast['Cumulative'], color='orange', label='Expected cumulative demand')
        ax.fill_between(forecast.index, forecast['Lower'].cumsum(), forecast['Upper'].cumsum(),
                        color='orange', alpha=0.2, label='80% confidence interval')
        ax.axhline(current_stock, color='red', linestyle='--', label='Current stock')
        ax.set_title('Cumulative Forecast Demand vs Current Stock')
        ax.set_xlabel('Date')
        ax.set_ylabel('Units')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

    with tab3:
        table = forecast.round(0).reset_index().rename(columns={'index': 'Date'})
        table['Date'] = table['Date'].dt.strftime('%d %b %Y')
        st.dataframe(table, use_container_width=True, hide_index=True)

    # ---- historical analysis (same as reference app)
    st.markdown('---')
    st.subheader('🗂️ Historical Sales Analysis')
    h1, h2 = st.tabs(['Monthly Sales Trend', 'Day of Week Sales Pattern'])

    with h1:
        fig, ax = plt.subplots(figsize=(10, 4))
        monthly_sales = target_df.set_index('date')['sales'].resample('ME').sum()
        ax.plot(monthly_sales.index, monthly_sales.values, marker='o', color='green', linewidth=2)
        ax.set_title('Monthly Sales Trend')
        ax.set_xlabel('Date')
        ax.set_ylabel('Monthly Total Sales')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

    with h2:
        fig, ax = plt.subplots(figsize=(7, 3))
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_sales = target_df.groupby('day_name')['sales'].mean().reindex(day_order)
        day_sales.plot(kind='bar', color='skyblue', edgecolor='black', ax=ax)
        ax.set_title('Average Sales by Day')
        ax.set_xlabel('Day')
        ax.set_ylabel('Average Sales')
        plt.xticks(rotation=45)
        st.pyplot(fig)
        plt.close(fig)
else:
    st.info('👈 Choose a store, category, stock and number of days in the sidebar, then click **Predict**.')