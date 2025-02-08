import dgl
"""
    File to load dataset based on user control from main file
"""
def LoadData(DATASET_NAME):
    """
        This function is called in the main_xx.py file 
        returns:
        ; dataset object
    """    
    # handling for (ZINC) molecule dataset
    if DATASET_NAME == 'Cora':  
        graph = dgl.data.CoraGraphDataset()
        dataset = graph[0]        
        return dataset
