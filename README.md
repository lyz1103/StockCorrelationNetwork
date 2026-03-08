# Stock Correlation Networks and Systemic Risk  
### A Graph-Based Analyzer Using Minimum Spanning Trees with Tree-DP Optimization

## Project Overview
This project analyzes the structure of financial markets using graph algorithms.  
Stocks often move together due to shared economic factors, sector exposure, or market sentiment. These relationships can be quantified using **correlation**.

By transforming correlations into graph distances, we can construct a network of stocks and extract its **Minimum Spanning Tree (MST)** to reveal the dominant dependency structure of the market.

This project builds an end-to-end pipeline to:

- Construct stock correlation networks
- Extract and analyze Minimum Spanning Trees
- Identify systemic stocks and sector clusters
- Select diversified portfolios using **tree dynamic programming**

---

## Financial Background

In financial markets, stock relationships are commonly analyzed using **correlation**, a value between **−1 and 1** indicating how similarly two stocks move.

A common workflow in quantitative finance research:

1. Compute the **correlation matrix** between stock returns
2. Transform correlations into **distance measures**
3. Construct a **graph**  
   - Nodes = stocks  
   - Edge weights = distances
4. Extract the **Minimum Spanning Tree (MST)**

The MST captures the most important dependencies in the market while removing redundant connections.

Key interpretations:

- **High-degree nodes** → potential systemic hubs
- **Clusters** → sector or risk group structures
- **Tree structure** → simplified view of market connectivity

This project extends MST analysis by applying **tree dynamic programming** to study systemic importance and portfolio diversification.

---

## MVP Features

### Data Pipeline
- Load stock price data using financial APIs
- Compute **log returns** and **correlation matrices**

### Graph Construction
- Transform correlations into **pairwise distances**
- Build a **complete weighted graph**

### Minimum Spanning Tree
- Implement **Prim's Algorithm**
- Extract the MST representing market structure

### Systemic Risk Metrics
Analyze nodes using:
- Node degree
- Subtree size
- Distance from a root node (e.g., **SPY**)

These metrics help identify **systemically important stocks**.

### Portfolio Optimization (Tree DP)
Use **Tree Dynamic Programming** to select a diversified subset of stocks:

- Choose **k non-adjacent nodes**
- Equivalent to a **maximum-weight independent set on a tree**
- Ensures diversification across the network structure

---

## Additional Features

### Time-Window Analysis
Compare MST structures across different time periods to study market evolution.

### Interactive Queries
Users can input a ticker to retrieve:

- Neighbors in the MST
- Distance to SPY
- Whether the stock appears in selected portfolios

### Sector Visualization
Detect clusters using **BFS/DFS** traversal to reveal sector groupings within the MST.

---

## Algorithmic Components

### Graph Algorithms
- Correlation → distance transformation
- **Minimum Spanning Tree (Prim’s Algorithm)**

### Tree Dynamic Programming
- Portfolio selection via **maximum-weight independent set**
- Optimize diversified portfolio of **k stocks**

### System Design
- Data ingestion via API
- Input parsing and error handling
- Modular analysis pipeline

---

## Potential Challenges

### Scalability
If the dataset contains hundreds of stocks, tree-DP and graph operations may become computationally expensive.

### Portfolio Optimization
Ensuring the **dynamic programming solution correctly selects the optimal portfolio of size k** while respecting graph constraints.

---

## Technologies
- Python
- Graph algorithms
- Financial data APIs
- Network analysis
- Dynamic programming

---

## Research Inspiration
The MST-based approach to financial networks is widely used in quantitative finance to analyze systemic risk and market structure.

This project combines **financial network theory** with **algorithmic optimization techniques** to build a practical market analysis tool.
