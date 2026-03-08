import yfinance as yf
import pandas as pd
import numpy as np
import math

'''
AI Citation:
for class Optimizer DP logic: used AI to get the ideas & hints to start, 
and get the logic of tree DP. See details below.
'''
class StockInfo:
    def __init__(self, ticker_list:list[str], start_date : str, end_date : str):
        '''
        ticker_list: the list of tickers to get information
        start_date: the start date of the stock info, format is string, 'yyyy-mm-dd'
        end_date: the end date of the stock info, format is string, 'yyyy-mm-dd'
        '''
        self.ticker_list = ticker_list
        self.start_date = start_date
        self.end_date = end_date
        self.price_df = pd.DataFrame()
    
    def loadPrice(self): # ***loadprice syntax searched online ***
        '''
        load the close price of stock adjusted for dividends and splits
        '''
        print("Fetching data from yahoo finance...")
        df = yf.download(self.ticker_list, start = self.start_date, end = self.end_date, \
                         group_by="ticker", auto_adjust=True, threads=True)
        self.price_df = df.xs("Close", axis=1, level=1)
        print("Price loaded.")
    
    def getPriceDf(self) -> pd.DataFrame:
        return self.price_df

    def getLogRetDf(self) -> pd.DataFrame:
        if len(self.price_df) == 0:
            self.loadPrice()
            if len(self.price_df) == 0: # if still no value 
                raise Exception("Prices cannot be loaded. Please check your input.")
        print("calculate log return ...")
        log_ret_df = np.log(self.price_df / self.price_df.shift(1)).dropna(how='all')
        log_ret_df = log_ret_df.dropna(axis=1, how='all')
        return log_ret_df
    
    def getCorrMatrix(self) -> pd.DataFrame:
        print("calculate correlation matrix ...")
        log_ret_df = self.getLogRetDf()
        missed_tickers = list(set(self.ticker_list)- set(list(log_ret_df.columns)))
        print(f"Unable to fetch information for {missed_tickers}, skip in optimization processes.")
        self.ticker_list =  list(log_ret_df.columns)
        self.price_df = self.price_df[log_ret_df.columns]
        return log_ret_df.corr()

    def CorrToDistance(self) -> pd.DataFrame: 
        # MST requires positive weight - map correlation [-1, 1] to values [0,2]
        print("convert correlation matrix to distances in graph ...")
        corr_matrix = self.getCorrMatrix().values # returns N * N array
        dist = np.sqrt(2 * (1 - corr_matrix))
        dist_df = pd.DataFrame(dist, index = self.ticker_list, columns = self.ticker_list)
        return dist_df


class stockMST:
    def __init__(self, distanceDf:pd.DataFrame):
        self.minweight, self.mst = stockMST.generateMST(distanceDf)
    
    @staticmethod
    def generateMST(distanceDf:pd.DataFrame) -> tuple[float, dict]:
        '''
        Generate MST from the distance dataframe got from stock correlation information
        input: distanceDf
        output: min_weight, MST
        '''
        tickers = list(distanceDf.columns)
        edges = []
        n = len(tickers)
        for i in range(n):
            for j in range(i + 1, n):
                edges.append((distanceDf[tickers[i]][tickers[j]], tickers[i], tickers[j]))
        
        edges.sort()

        uf = UnionFind(tickers)
        mst = dict()
        min_weight = 0
        for edge in edges:
            weight, node1, node2 = edge
            
            # if not in the same forest, add edge to MST and connect the forests
            if uf.find(node1) != uf.find(node2):
                min_weight += weight
                mst[node1] = mst.get(node1, dict())
                mst[node1][node2] = weight
                mst[node2] = mst.get(node2, dict())
                mst[node2][node1] = weight
                uf.union(node1, node2)

        return min_weight, mst
    
    def getKnearestneihgbor(self, target:str, K:str|int) -> list | None: 
        '''
        calc the distance between target and other nodes in MST
        using a dictionary containing the key = node, value = distance of the node to target
        '''
        # get the distance from each node to target node
        dist_to_target = {target:0}
        tovisit = [target]
        mst = self.mst

        while len(tovisit) > 0:
            curr = tovisit.pop()
            for node,dist in mst[curr].items():
                if node not in dist_to_target:
                    dist_to_target[node] = dist_to_target[curr] + dist
                    tovisit.append(node)
        
        # check the cluster: K nearest neighbor
        res = [[float(d), node] for node, d in dist_to_target.items() if node != target]
        res.sort()
        return res[:int(K)]

    def generateRootedTree(self, root:str) -> tuple[dict, dict, dict, dict, dict]:
        '''
        generate rooted trees, usually set the root to be SPY, so we can analyze
        other nodes, to see how 'market-like' it is
        returns: the parent dictionary for MST, the distance (from root) dictionary for MST weights,
        the depth (from root) dictionary for MST
        '''
        graph = self.mst
        parent = {root : None}
        children = {node: [] for node in graph}
        dist_to_parent = {root:0} 


        distance = {root : 0}
        depth = {root : 0}
        tovisit = [root]

        # perform dfs to update the nodes in MST
        while len(tovisit) != 0:
            curr_node = tovisit.pop()
            for nbr, weight in graph[curr_node].items():
                if nbr not in parent:
                    parent[nbr] = curr_node
                    children[curr_node].append(nbr)
                    depth[nbr] = depth[curr_node] + 1
                    distance[nbr] = distance[curr_node] + weight
                    dist_to_parent[nbr] = weight
                    tovisit.append(nbr)
        
        return parent, children, dist_to_parent, distance, depth
    
    def computeDegrees(self) -> dict:
        '''
        Compute the degree of each nodes in MST, to see if it is clustered or isolated
        if one node has large degrees, meaning it is likely to be the information carrier 
        or systematically important (with many strong correlations)
        returns a dictionary containing all nodes with the value to be the degrees
        '''
        graph = self.mst
        degrees = {node: len(graph[node]) for node in graph}
        return degrees
    

    def computeSubtreeSize(self, root:str, parent:dict) -> dict: 
        # parent got from generateRootedTree function
        '''
        Compute the subtree size for each node, if it the root of large subtree, 
        and implies it is the root of a large correlated cluster
        it reflects the sector of the clusters, and the distance away from root
        also indicates whether it is market-like or no
        '''

        subtreeSizes = {}
        graph = self.mst

        def dfs(u): # apply dfs to calculate the subtree sizes
            curr_size = 1
            for v in graph[u]:
                if parent[v] == u: 
                    curr_size += dfs(v)
            subtreeSizes[u] = curr_size # update the dictionary
            return curr_size # return the current subtree size (including itself)

        dfs(root)
        return subtreeSizes

class Optimizer:
    def __init__(self, parent:dict, children:dict, dist_to_parent:dict, root:str):
        self.parent = parent
        self.children = children
        self.dist_to_parent = dist_to_parent
        self.root = root
    
    # choose k with largest distance and most diversified
    def GetminCorrPortfolio(self, K) -> dict:
        dp, choices = self.minCorrPortfolioCalc(K)
        max_corr_sum = dp[self.root][K]
        selected_tickers = Optimizer.backtrackingSelection(self.root, K, self.children, choices)
        return {'maximum pairwise distance sum': max_corr_sum, 
                "selected K tickers": selected_tickers}

    def minCorrPortfolioCalc(self, K) -> tuple[dict, dict]:
        '''
        dp[u][t]: min total distance using edges rooted on subtree u, 
        with exactly t stocks(nodes) selected
        base case: if leaf: dp[u][0]=dp[u][1]=0, o.w. infinity
        '''

        # create two dictionaries to store dp tables for each node, key is the ticker
        dp = {}
        choices = {}

        def dfs(node):
            for child in self.children[node]:
                dfs(child) # solve child first, bottom up
            # then compute dp[u] using children's dp
            dp[node], choices[node] = \
                Optimizer.compute_mincorr_node(node, self.children, dp, self.dist_to_parent, K)
        
        dfs(self.root)
        return dp, choices
     
    @staticmethod
    def compute_mincorr_node(node:str, children:dict, \
                             dp:dict, dist_to_parent:dict, K) -> tuple[list, list]:
        '''
        Citation: used AI to get some starting ideas and thoughts on how to run dp;
        Used AI to fix small bugs.
        '''
        child_list = children[node]
        n = len(child_list)

        # initialize dp for node
        dp_node = [-math.inf] * (K + 1) # from 0 to K selected from node
        dp_node[0] = 0 # to store the min weight, select 0 stock
        if K >= 1:
            dp_node[1] = 0 # select itself
        
        # choice dp to record the results
        # choice[node]-> [child_index][total] = (allocated, to_allocate) optimal choice
        choices_node = [ [None] * (K + 1) for _ in range(n + 1) ]

        # loop through children to combine the subtree result
        for i, child in enumerate(child_list, start = 1): # index start at 1
            dp_child = dp[child] # dp[child][*]: a list
            dist = dist_to_parent[child] # dist (corr) between child and parent node

            # create a new dp to store the updated result for dp_node
            merged_dp = [-math.inf] * (K + 1) # from 0 to K selected from node
            for allocated in range(0, K + 1): 
            # allocated is the number of nodes allocated on the rest of trees rooted at node
                if dp_node[allocated] == -math.inf: 
                    # total nodes in previous subtree is infeasible
                    continue
                for to_allocate in range(0, K - allocated + 1):
                    # the available number of nodes that can be allocate to this child
                    if dp_child[to_allocate] == -math.inf: 
                        #nodes in child subtree is infeasible
                        continue

                    total = allocated + to_allocate
                    # dist is the edge cost need to pass
                    # to_allocate * (K - to_allocate): 
                    # # of pairwise combination need to cross this edge
                    weight_edge = dist * to_allocate * (K - to_allocate) 
                    curr = dp_node[allocated] + dp_child[to_allocate] + weight_edge
                    if curr > merged_dp[total]:
                        merged_dp[total] = curr
                        choices_node[i][total] = (allocated, to_allocate)

            # update dp_node, and continue to rolling update for next child
            dp_node = merged_dp 
        
        # dp_node: 1D list for node u, the weight from selecting 0 to K stocks
        # choices_node: 2D list with shape (n+1, K+1) storing the splits at each step
        return dp_node, choices_node 

    @staticmethod
    def backtrackingSelection(node:str, t:int, children:dict, choices:dict) -> list[str]:
        '''
        Return a list of selected tickers in the subtree of node,
        given we have chosen exactly t nodes in that subtree.

        Citation: used AI to get the ideas and hints.
        '''
        selected = []
        child_list = children[node]
        choice_steps = choices[node] # (m+1) x (K+1) table for u
        n = len(child_list) # number of children & iterative steps for node
        
        # track how many nodes we assigned for each child
        child_counts = [0] * n
        to_track = t 

        # recurrence
        for i in range(n, 0, -1): # for choice, the index start at 1
            # allocated is the number of nodes assigned to other subtrees
            # to_allocate is the number of nodes assigned to the ith child
            allocated, to_allocate = choice_steps[i][to_track]
            child_counts[i-1] = to_allocate # index uses i-1 since dp start at 1
            to_track = allocated
        
        # after looping all children: left with whether node is selected or no
        if to_track == 1: # node selected
            selected.append(node)
        
        for i, child in enumerate(child_list):
            to_allocate = child_counts[i]
            if to_allocate > 0:
                selected.extend(Optimizer.backtrackingSelection(child, to_allocate, children, choices))

        return selected
    
class Backtester:
    def __init__(self, ticker_list:list, weights:list, start_date:str, end_date:str):
        '''
        ticker_list: the list of tickers to get information
        weights in a portfolio: corresponding to the ticker_list order
        start_date: the start date of the stock info, format is string, 'yyyy-mm-dd'
        end_date: the end date of the stock info, format is string, 'yyyy-mm-dd'
        '''
        self.ticker_list = ticker_list
        self.start_date = start_date
        self.end_date = end_date
        self.weights = weights
    
    def get_performance(self) -> dict:
        print("Fetching data from yahoo finance...")
        raw_data = yf.download(self.ticker_list, start = self.start_date, end = self.end_date, \
                         group_by="ticker", auto_adjust=True, threads=True)
        price_df = raw_data.xs("Close", axis=1, level=1)
        print("Price loaded.")

        # return-related
        stock_ret = price_df.pct_change().dropna()
        port_ret = stock_ret.dot(self.weights)
        cum_val = (1 + port_ret).cumprod()
        cum_ret = cum_val.iloc[-1] - 1
        years = (price_df.index[-1] - price_df.index[0]).days / 365
        annual_ret = (cum_val.iloc[-1])**(1/years) - 1

        # risk-related: searched online for syntax on cummmax()
        # 1. max drawdown
        running_max = cum_val.cummax()
        drawdown = (cum_val - running_max) / running_max
        max_dd = drawdown.min()

        # 2. Sharpe Ratio
        sharpe = (port_ret.mean() * 252) / (port_ret.std() * np.sqrt(252))

        return { "total_return": cum_ret,
                "annualized_return": annual_ret,
                "sharpe": sharpe,
                "max_drawdown": max_dd}, cum_val


# used for MST - Krustal - *** From Recitation codes
class UnionFind:
    def __init__(self, nodes):
        self.parents = {node : node for node in nodes}
        self.sizes = {node : 1 for node in nodes}
    
    def find(self, node):
        # With union by size and path compression,
        # find is expected to run in O(alpha(n))
        if self.parents[node] == node:
            return node
        else:
            parent = self.parents[node]
            self.parents[node] = self.find(parent)
            return self.parents[node]
    
    def union(self, node1, node2):
        root1, root2 = self.find(node1), self.find(node2)

        if root1 == root2: return

        size1, size2 = self.sizes[root1], self.sizes[root2]

        if size1 < size2:
            self.parents[root1] = root2
            self.sizes[root2] = size1 + size2
        else:
            self.parents[root2] = root1
            self.sizes[root1] = size1 + size2
    
    def getSize(self, node):
        return self.sizes[self.find(node)]







