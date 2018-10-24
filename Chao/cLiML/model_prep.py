def standardize_X_train(df):
    return (df - df.mean())/df.std()


def standardize_X_test(df_test, df_train):
    return (df_test - df_train.mean())/df_train.std()


def match_test_to_train_columns(df_test, df_train):
    columns_missing = set(df_train.columns) - set(df_test.columns)
    
    for col in columns_missing:
        df_test[col] = 0
    
    df_test = df_test[df_train.columns]
    return df_test



def feature_select_linear_reg(X, y, cv = 10):
    import pandas as pd
    import numpy as np
    # RFECV
    from sklearn.feature_selection import RFECV
    from sklearn.linear_model import LinearRegression

    lr = LinearRegression()

    selector = RFECV(lr, step = 1, cv = cv)
    selector = selector.fit(X, y)

    df_RFECV = pd.DataFrame({'variable_RFECV': X.columns.values.tolist(),
                            'support': selector.support_,
                            'ranking': selector.ranking_}).sort_values('ranking', ascending = True)
    df_RFECV.reset_index(drop = True, inplace = True)
    
    # SelectKbest
    from sklearn.feature_selection import SelectKBest, f_regression
    skb = SelectKBest(f_regression)

    skb.fit(X, y)

    df_skb = pd.DataFrame({'variable_skb': X.columns.values.tolist(),
                          'score': skb.scores_.tolist()}).sort_values('score', ascending = False)
    df_skb.reset_index(drop = True, inplace = True)
    
    # Lasso
    from sklearn.preprocessing import StandardScaler

    ss = StandardScaler()
    Xs = ss.fit_transform(X)
    
    from sklearn.linear_model import Lasso, LassoCV
    optimal_lasso = LassoCV(n_alphas = 500, cv = cv, verbose = 1)
    optimal_lasso.fit(Xs, y)

    optimal_alpha = optimal_lasso.alpha_
    lasso_coef = optimal_lasso.coef_
    
    df_lasso = pd.DataFrame({'Variable_Lasso': X.columns.values.tolist(),
                            'Coefficients': lasso_coef.tolist(),
                            'CoeffAbs': abs(lasso_coef).tolist()}).sort_values('CoeffAbs', ascending = False)
    df_lasso.reset_index(drop = True, inplace = True)
    
    
    # combine all 3 dataframe
    df = pd.concat([df_RFECV, df_skb], axis = 1)
    df = pd.concat([df, df_lasso], axis = 1)
    
    return df


def feature_select_logistic_reg(X, y, cv = 10):
    import pandas as pd
    import numpy as np
    # RFECV
    from sklearn.feature_selection import RFECV
    from sklearn.linear_model import LogisticRegression

    lr = LogisticRegression()

    selector = RFECV(lr, step = 1, cv = cv)
    selector = selector.fit(X, y)

    df_RFECV = pd.DataFrame({'variable_RFECV': X.columns.values.tolist(),
                            'support': selector.support_,
                            'ranking': selector.ranking_}).sort_values('ranking', ascending = True)
    df_RFECV.reset_index(drop = True, inplace = True)
    
    # SelectKbest
    from sklearn.feature_selection import SelectKBest, f_classif
    skb = SelectKBest(f_classif)

    skb.fit(X, y)

    df_skb = pd.DataFrame({'variable_skb': X.columns.values.tolist(),
                          'score': skb.scores_.tolist()}).sort_values('score', ascending = False)
    df_skb.reset_index(drop = True, inplace = True)
    
    
    # combine all 3 dataframe
    df = pd.concat([df_RFECV, df_skb], axis = 1)
    
    
    return df


def dummify(df, categorical_var = None):
    import numpy as np
    
    if categorical_var != None:
        # loop through each categorical variable
        for category in categorical_var:
            # dummify this category
            df = __dummy(df, category)

        return df
    else:
        variables = df.columns.values.tolist()
        
        cate_vars = []
        for var in variables:
            if np.issubdtype(df[var].dtype, np.number) == False:
                cate_vars.append(var)
        
        
        for category in cate_vars:
            # dummify this category
            df = __dummy(df, category)
            
        return df
        
        
     
def __dummy(df, category):
    import pandas as pd
    # dummify the categorical variable
    dummy = pd.get_dummies(df[category], prefix = str(category))

    # concatenate this to the original dataframe
    df = pd.concat([df, dummy], axis = 1)

    # delete the original categorical variable, 
    df.drop(columns = [category], inplace = True)

    # get the name of the last column in dummies, and delete the last column of dummified variable
    del_col = dummy.columns.values.tolist()[-1]
    df.drop(columns = [del_col], inplace = True)
    
    return df



def dummify_test(df, categorical_var = None):
    import numpy as np
    
    if categorical_var != None:
        # loop through each categorical variable
        for category in categorical_var:
            # dummify this category
            df = __dummy_test(df, category)

        return df
    else:
        variables = df.columns.values.tolist()
        
        cate_vars = []
        for var in variables:
            if np.issubdtype(df[var].dtype, np.number) == False:
                cate_vars.append(var)
        
        
        for category in cate_vars:
            # dummify this category
            df = __dummy_test(df, category)
            
        return df
        
        
     
def __dummy_test(df, category):
    import pandas as pd
    # dummify the categorical variable
    dummy = pd.get_dummies(df[category], prefix = str(category))

    # concatenate this to the original dataframe
    df = pd.concat([df, dummy], axis = 1)

    # delete the original categorical variable, 
    df.drop(columns = [category], inplace = True)
    
    return df