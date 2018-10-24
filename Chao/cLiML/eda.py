def findOutlier(dataframe):
    import numpy as np
    
    ## Isolate the numerical columns out of the data frame
    variables = dataframe.columns.values.tolist()
    
    num_vars = []
    for var in variables:
        if np.issubdtype(dataframe[var].dtype, np.number):
           num_vars.append(var)
        
    # for each column
    outlier_dict = {}
    for column in num_vars:
        # get the Q1 and Q3 for that column
        Q1 = dataframe[column].quantile(0.25)
        Q3 = dataframe[column].quantile(0.75)
        # calculate the IQR
        IQR = Q3 - Q1
        # get the outliers from that column
        outliers = dataframe[[column]][(dataframe[column] < (Q1 - 1.5*IQR)) | 
                                       (dataframe[column] > (Q3 + 1.5*IQR))]
        
        value = outliers[column].unique().tolist()
        if (len(value) > 0):
            key = column
            outlier_dict[key] = value
        
    return outlier_dict



### CONDUCTS EDA ON CATEGORICAL VARIABLES
def edaCategorical(df, target, aggre = 'median', max_y_axis = None):
    
    import numpy as np
    import math 
    import matplotlib.pyplot as plt
    import pandas as pd
    
    ncols = 2
    num_of_plots = df.shape[1] - 1
    num_of_row_needed = num_of_plots/ncols
    
    nrows = 0
    
    if num_of_row_needed.is_integer():
        nrows = num_of_row_needed
    else:
        nrows = math.floor(num_of_row_needed) + 1
        
        
    # range limit
    y_max = df[target].max()
    y_min = df[target].min()
    
    if max_y_axis != None:
        y_max = max_y_axis

    # fig size
    width = 16
    height = width*int(nrows)*0.5

    # plotting each variables against price
    fig, ax = plt.subplots(nrows = int(nrows), ncols = int(ncols), figsize = (width, height))
    x_columns = df.columns.values.tolist()
    x_columns.remove(target)

    count = 0
    i = 0
    j = 0
    for col in x_columns:
        i = math.floor(count/2)
        if count % 2 == 0:
            j = 0
        else:
            j = 1

    #     print ('i', i)
    #     print ('j', j)
    #     print(col)
        if aggre == 'mean':
            df.groupby(col)[[target]].mean().plot(kind = 'bar', ax = ax[i,j])
        else:
            df.groupby(col)[[target]].median().plot(kind = 'bar', ax = ax[i,j])
            
        ax[i,j].set_ylim([y_min, y_max])
        
        count+=1
       
    
### CONDUCTS EDA ON CONTINUOUS VARIABLES
def edaContinuous(df, target, max_y_axis = None):
    
    import numpy as np
    import math 
    import matplotlib.pyplot as plt
    import pandas as pd
    
    ## Isolate the numerical columns out of the data frame
    variables = df.columns.values.tolist()
    
    num_vars = []
    for var in variables:
        if np.issubdtype(df[var].dtype, np.number):
           num_vars.append(var)
        
    # if target got remove, re add it.
    if target not in num_vars:
        num_vars = num_vars.append(target)
        
    # remove all categorical variables
    df = df[num_vars]
    
    # get subplot rows and cols
    ncols = 2
    num_of_plots = df.shape[1] - 1
    num_of_row_needed = num_of_plots/ncols
    
    nrows = 0
    
    if num_of_row_needed.is_integer():
        nrows = num_of_row_needed
    else:
        nrows = math.floor(num_of_row_needed) + 1
        
    
    
    # range limit
    y_max = df[target].max()
    y_min = df[target].min()
    
    if max_y_axis != None:
        y_max = max_y_axis

        
    # fig size
    width = 16
    height = width*int(nrows)*0.5

    # plotting each variables against price
    fig, ax = plt.subplots(nrows = int(nrows), ncols = int(ncols), figsize = (width, height))
    x_columns = df.columns.values.tolist()
    x_columns.remove(target)
    

    count = 0
    i = 0
    j = 0
    
    for col in x_columns:
        i = math.floor(count/2)
        if count % 2 == 0:
            j = 0
        else:
            j = 1

        #print ('i', i)
        #print ('j', j)
        #print(col)

        #df.groupby(col)[[target]].median().plot(kind = 'bar', ax = ax[i,j])
        df.plot(kind = 'scatter', x = col, y = target, ax = ax[i,j])
        ax[i,j].set_ylim([y_min, y_max])
        
        count+=1