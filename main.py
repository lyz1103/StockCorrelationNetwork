import pandas as pd
import numpy as np
import Stock_MST_Classes # store the classes and functions in this module
import streamlit as st
import graphviz # used AI to search how to use this package to visualize trees
import datetime
import plotly.graph_objects as go

## Constants ##
medium_tickers = ["SPY", # represent the market, to be the root
    "MMM","AOS","ABT","ABBV","ACN","ATVI","ADM","ADBE","ADT","AAP",
    "AES","AFL","A","APD","AKAM","ALK","ALLE","LNT","ALL",
    "GOOG","MO","AMZN","AMCR","AEE","AAL","AEP","AXP","AIG","AMT",
    "AWK","AMP","ABC","AME","AMGN","APH","ADI","ANSS","ANTM","AON","GOOGL"]

small_tickers = ["SPY", # represent the market, to be the root
    "MMM","AOS","ABT","ABBV","ACN","ATVI","GOOGL"]

large_tickers = ["SPY",
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "GOOG", "META", "BRK.B", "TSLA", "UNH",
    "XOM", "JPM", "JNJ", "V", "MA", "HD", "PG", "LLY", "AVGO", "CVX",
    "MRK", "PEP", "COST", "WMT", "ABBV", "KO", "CRM", "PFE", "ACN", "TMO",
    "NFLX", "LIN", "DHR", "ABT", "WFC", "MCD", "CSCO", "TXN", "UPS", "PM",
    "MS", "ADBE", "BMY", "AMGN", "COP", "HON", "LOW", "NEE", "IBM", "QCOM",
    "SBUX", "CAT", "GE", "RTX", "INTU", "GS", "AMD", "CVS", "BLK", "MDT",
    "AXP", "SPGI", "SYK", "AMAT", "DE", "PLD", "USB", "BKNG", "LMT", "ADI",
    "T", "ELV", "NOW", "GILD", "ISRG", "FIS", "MO", "BDX", "CI", "MDLZ",
    "MMC", "TJX", "ZTS", "C", "APD", "SO", "ICE", "PNC", "MMM", "REGN",
    "AON", "MU", "CL", "EQIX", "FDX", "SCHW", "ITW", "NKE", "EW", "ORLY"]

financial_tickers = ["JPM", "BAC", "WFC", "C", "GS", "MS", "AXP", "V", "MA", "BLK",
    "SCHW", "PNC", "USB", "TFC", "COF", "AON", "MET", "PRU", "CME", "ICE"]

tech_tickers = ["AAPL", "MSFT", "GOOGL", "GOOG", "NVDA", "META", "AMZN", "TSLA", "AVGO", "ORCL",
    "INTC", "AMD", "ADBE", "CSCO", "CRM", "QCOM", "TXN", "IBM", "NOW", "AMAT"]

## Functions ##
## Used AI for codes of visualize_tree
def visualize_tree(root:str, children:dict, edge_weight:dict):
    dot = graphviz.Digraph()
    dot.attr("node", shape="circle")

    def dfs(node):
        for child in children.get(node, []):
            w = edge_weight.get(node, {}).get(child, "")
            dot.edge(str(node), str(child), label=str(round(w,2)))
            dfs(child)

    dfs(root)
    st.graphviz_chart(dot)

def display_ticker_info(interested_ticker:str, children:dict, \
                        parent:dict, degrees:dict, distance:dict, dist_to_parent:dict):
    
    st.text(f'Children of {interested_ticker} is {", ".join(children[interested_ticker])}')
    st.text(f'Parent of {interested_ticker} is {parent[interested_ticker]}')
    st.text(f'Degree of {interested_ticker} is {degrees[interested_ticker]}')
    st.text(f'Distance of {interested_ticker} to its parent is {round(dist_to_parent[interested_ticker],2)}')
    st.text(f'Distance of {interested_ticker} to root node is {round(distance[interested_ticker],2)}')

def process_port_info(port_str:str, weight_str:str) -> tuple[list,list]:
    ticker = [t.strip().upper() for t in port_str.split(",")]
    weight = [float(w.strip()) for w in weight_str.split(",")]
    if abs(sum(weight) - 1) > 0.05:
        st.error("Weights must sum to 1, please check input number and/or format.")
    return ticker, weight

## Visualizations
# streamlit run main.py
st.title('Stock Correlation Networks and Systemic Risk')
st.subheader("A Graph-Based Analyzer Using MST with Tree-DP Optimization")

# get ticker list
ticker_type = st.selectbox('Choose the ticker list:',
['small', 'medium', 'large','finance','tech'])

ticker_map = {"small" : small_tickers, "medium" : medium_tickers, "large" : large_tickers,
              "finance" : financial_tickers,"tech" : tech_tickers}
              
test_tickers = ticker_map[ticker_type]

# get date range
start, end = st.date_input("Select date range: ",
    value=(datetime.date(2024, 1, 1), datetime.date.today()))

# get stock information and process the data
# use OOP
data_load_state = st.text('Loading data...')
stocks = Stock_MST_Classes.StockInfo(test_tickers, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
stocks.loadPrice()
distance_df = stocks.CorrToDistance()
price_df = stocks.price_df
data_load_state.text('Loading data...done!')

st.subheader('Data Overview')
st.text('Price Data: ')
st.dataframe(price_df.style.format_index("{:%Y-%m-%d}", axis=0))
st.text('Distance Data: ')
st.write(distance_df)

# use the distance to represent correlation, and generate the MST
stock_graph = Stock_MST_Classes.stockMST(distance_df)

# get all related information
st.subheader('Tree info')
root = st.selectbox("Root of the tree: ", test_tickers)
K = st.number_input("Number of Stocks in the portfolio: ", value = 3)

parent, children, dist_to_parent, distance, depth = stock_graph.generateRootedTree(root)
degrees = stock_graph.computeDegrees()
subtreeSizes = stock_graph.computeSubtreeSize(root, parent)
st.text('Visualization of Rooted Tree: ')
visualize_tree(root, children, stock_graph.mst)
# searched the syntax online
st.text(f'The stock with highest degree is {max(degrees, key = degrees.get)}') 
st.text(f'The stock with largest subtree size is {max(subtreeSizes, key = subtreeSizes.get)}')

# can use it to compare when large numbers and see how it compares with other ticker
st.subheader('Node/Stock info')
ticker1, ticker2 = st.columns(2)
with ticker1:
    interested_ticker = st.selectbox("Ticker 1", test_tickers)
    display_ticker_info(interested_ticker, children, parent, degrees, distance, dist_to_parent)
    
with ticker2:
    interested_ticker = st.selectbox("Ticker 2", test_tickers)
    display_ticker_info(interested_ticker, children, parent, degrees, distance, dist_to_parent)

# apply DP to choose exactly K stocks that are well-diversified
# which is the maximum-weight set of size K on a tree
st.subheader('Optimizing Portfolio with longest distance in MST')
portfolioOptimizer = Stock_MST_Classes.Optimizer(parent, children, dist_to_parent, root)
res = portfolioOptimizer.GetminCorrPortfolio(K)
max_dist_sum = res['maximum pairwise distance sum']
selected_tickers = res['selected K tickers']
st.text("The max distance sum from the MST is " + str(round(max_dist_sum,2)))
st.text("The selected tickers from the market portfolio with min correlation are " + ", ".join(selected_tickers))
# backtesting using selected portfolio vs. market portfolio - to see if it has good risk management

# time-window analysis---------------------------
st.subheader('Time-window analysis: choose another time and compare results')
# get date range and price info
start_new, end_new = st.date_input("Select comparison date range: ",
    value=(datetime.date(2020, 1, 1), datetime.date(2021, 1, 1)))
stocks_new = Stock_MST_Classes.StockInfo(test_tickers, \
                start_new.strftime("%Y-%m-%d"), end_new.strftime("%Y-%m-%d"))

data_load_state = st.text('Loading data...')
stocks_new.loadPrice()
data_load_state.text('Loading data...done!')

distance_df_new = stocks_new.CorrToDistance()

# construct new MST for comparison
stock_graph_new = Stock_MST_Classes.stockMST(distance_df_new)
parent_new, children_new, dist_to_parent_new, distance_new, \
    depth_new = stock_graph_new.generateRootedTree(root)

degrees_new = stock_graph_new.computeDegrees()
subtreeSizes_new = stock_graph_new.computeSubtreeSize(root, parent_new)
st.text('Visualization of new rooted Tree: ')
visualize_tree(root, children_new, stock_graph_new.mst)

# compare same stock within two different time frames
st.subheader('Stock Time-window Comparison')
interested_ticker = st.selectbox("Ticker_new", test_tickers)
time1, time2 = st.columns(2)
with time1:
    st.text(f'{start.strftime("%Y-%m-%d")} to {end.strftime("%Y-%m-%d")}: ')
    display_ticker_info(interested_ticker, children, parent, degrees, distance, dist_to_parent)
    
with time2:
    st.text(f'{start_new.strftime("%Y-%m-%d")} to {end_new.strftime("%Y-%m-%d")}: ')
    display_ticker_info(interested_ticker, children_new, parent_new, degrees_new, \
                        distance_new, dist_to_parent_new)

st.subheader('Portfolio Backtesting')
# get date range
start_port, end_port = st.date_input("Select portfolio date range: ",
    value=(start, end))

perf_1 = None
perf_2 = None
port1, port2 = st.columns(2)
with port1:
    port1_str = st.text_input("Enter tickers for first portfolio (comma-separated):",'SPY')
    weight1_str = st.text_input("Enter weights 1 (comma-separated):",'1')
    

with port2:
    port2_str = st.text_input("Enter tickers for second portfolio (comma-separated):",'MMM,ACN,ABT')
    weight2_str = st.text_input("Enter weights 2 (comma-separated):",'0.333,0.333,0.333')

if st.button("Generate Results"):
    if port1_str and weight1_str:
        ticker1, weight1 = process_port_info(port1_str,weight1_str)
        portfolio_1 = Stock_MST_Classes.Backtester(ticker1, weight1, start_port.strftime("%Y-%m-%d"),\
                                                    end_port.strftime("%Y-%m-%d"))
        perf_1, cum_val_1 = portfolio_1.get_performance()
        
    if port2_str and weight2_str:
        ticker2, weight2 = process_port_info(port2_str,weight2_str)
        portfolio_2 = Stock_MST_Classes.Backtester(ticker2, weight2, start_port.strftime("%Y-%m-%d"),\
                                                    end_port.strftime("%Y-%m-%d"))
        perf_2, cum_val_2 = portfolio_2.get_performance()

# list out the performance
if perf_1 != None and perf_2 != None:
    summary = pd.DataFrame([perf_1, perf_2], index=["Portfolio 1", "Portfolio 2"])
    st.write(summary)

    # visualize using cumulative val: searched AI for syntax
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=cum_val_1.index,y=cum_val_1.values, mode="lines", name="Portfolio 1"))
    fig.add_trace(go.Scatter(x=cum_val_2.index,y=cum_val_2.values, mode="lines", name="Portfolio 2"))

    fig.update_layout(title="Cumulative Portfolio Value", xaxis_title="Date",
        yaxis_title="Portfolio Value", hovermode="x unified")

    st.plotly_chart(fig, use_container_width=True)

# the correlation plot does not imply anything on return, but for clustering
st.subheader('Clustering: K nearest neighbor of the target stock')
target = st.selectbox("Target stock: ", test_tickers)
num_neighbor = st.text_input("Number of nearest neighbors:", 3)
res = stock_graph.getKnearestneihgbor(target, num_neighbor)
res_df = pd.DataFrame(res, columns=["Distance", "Ticker"])
st.write(res_df)

