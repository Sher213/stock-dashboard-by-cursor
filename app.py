import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
from prophet import Prophet
import openai
from pathlib import Path
import json
import os
# Configuration
st.set_page_config(
    page_title="Stock Market AI Dashboard",
    page_icon="📈",
    layout="wide"
)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []


openai.api_key = os.getenv("OPENAI_API_KEY")

class StockDashboard:
    def __init__(self):
        self.default_stock = "AAPL"
        self.stocks_data = {}
        
    def fetch_stock_data(self, symbol, period="1y"):
        """Fetch stock data using yfinance"""
        stock = yf.Ticker(symbol)
        hist = stock.history(period=period)
        return hist
    
    def create_candlestick_chart(self, df, symbol):
        """Create an interactive candlestick chart using plotly"""
        fig = go.Figure(data=[go.Candlestick(x=df.index,
                                            open=df['Open'],
                                            high=df['High'],
                                            low=df['Low'],
                                            close=df['Close'])])
        fig.update_layout(title=f'{symbol} Stock Price',
                         xaxis_title='Date',
                         yaxis_title='Price (USD)')
        return fig
    
    def predict_stock_price(self, df, days=30):
        """Predict future stock prices using Prophet"""
        # Prepare data for Prophet
        prophet_df = df.reset_index()[['Date', 'Close']].rename(
            columns={'Date': 'ds', 'Close': 'y'})
        
        # Remove timezone information
        prophet_df['ds'] = prophet_df['ds'].dt.tz_localize(None)
        
        # Create and fit the model
        model = Prophet(daily_seasonality=True)
        model.fit(prophet_df)
        
        # Create future dates for prediction
        future_dates = model.make_future_dataframe(periods=days)
        forecast = model.predict(future_dates)
        
        return forecast
    
    def create_prediction_chart(self, forecast, symbol):
        """Create prediction chart using plotly"""
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'],
                                name='Prediction'))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'],
                                fill=None, mode='lines', name='Upper Bound'))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'],
                                fill='tonexty', mode='lines', name='Lower Bound'))
        
        fig.update_layout(title=f'{symbol} Price Prediction',
                         xaxis_title='Date',
                         yaxis_title='Price (USD)')
        return fig

def process_query(query, context):
    """Process user query using OpenAI"""
    system_prompt = """You are a helpful financial analyst assistant. 
    Use the provided stock market data context to answer questions accurately and concisely."""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
    ]
    
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.7,
        max_tokens=150
    )
    
    return response.choices[0].message.content

def main():
    st.title("📈 Stock Market AI Dashboard")
    
    # Add clear history button in sidebar
    if st.sidebar.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()
    
    # Initialize dashboard
    dashboard = StockDashboard()
    
    # Sidebar
    st.sidebar.header("Settings")
    symbol = st.sidebar.text_input("Stock Symbol", dashboard.default_stock)
    
    # Fetch data
    if symbol:
        data = dashboard.fetch_stock_data(symbol)
        
        # Display current stock price
        current_price = data['Close'].iloc[-1]
        price_change = current_price - data['Close'].iloc[-2]
        price_change_pct = (price_change / data['Close'].iloc[-2]) * 100
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Current Price", f"${current_price:.2f}", 
                   f"{price_change:.2f} ({price_change_pct:.2f}%)")
        
        # Candlestick chart
        st.subheader("Price History")
        candlestick_chart = dashboard.create_candlestick_chart(data, symbol)
        st.plotly_chart(candlestick_chart, use_container_width=True)
        
        # Price prediction
        st.subheader("Price Prediction (Next 30 Days)")
        forecast = dashboard.predict_stock_price(data)
        prediction_chart = dashboard.create_prediction_chart(forecast, symbol)
        st.plotly_chart(prediction_chart, use_container_width=True)
        
        # AI Q&A Section
        st.subheader("Ask Questions About the Stock")
        
        # Create an inline input field and button using columns with custom CSS
        st.markdown("""
            <style>
            .stButton > button {
                margin-top: 24px;
            }
            </style>
            """, unsafe_allow_html=True)
            
        col1, col2 = st.columns([5, 1], gap="small")  # Reduced gap between columns
        
        with col1:
            user_query = st.text_input("Enter your question:", key="query_input", label_visibility="visible")
        with col2:
            submit_button = st.button("Enter", use_container_width=True)
        
        if submit_button and user_query:  # Only process if button is clicked and query exists
            # Prepare context
            context = f"""
            Stock Symbol: {symbol}
            Current Price: ${current_price:.2f}
            Price Change: ${price_change:.2f} ({price_change_pct:.2f}%)
            Last 5 days closing prices: {data['Close'].tail().to_dict()}
            """
            
            response = process_query(user_query, context)
            st.session_state.chat_history.append(("You", user_query))
            st.session_state.chat_history.append(("AI", response))
            
            # Clear the input field after submission
            if "query_input" not in st.session_state:
                st.session_state.query_input = ""
            
            st.rerun()
        
        # Display chat history
        st.subheader("Chat History")
        chat_container = st.container()
        
        with chat_container:
            for role, message in st.session_state.chat_history:
                if role == "You":
                    st.markdown(
                        f"""
                        <div style='background-color: #e6f3ff; padding: 10px; border-radius: 5px; margin: 5px 0; color: #1a1a1a;'>
                            <strong style='color: #004d99;'>You:</strong><br>{message}
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                else:  # AI response
                    st.markdown(
                        f"""
                        <div style='background-color: #f0f0f0; padding: 10px; border-radius: 5px; margin: 5px 0; color: #1a1a1a;'>
                            <strong style='color: #404040;'>AI:</strong><br>{message}
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )

if __name__ == "__main__":
    main()
