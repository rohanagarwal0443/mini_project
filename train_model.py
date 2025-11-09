import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder

data=pd.read_csv(r"mini_project\properties_5000_modified.csv")
if data.empty:
    print("data has not loaded successfully")
else:
    
    #filling the missing value
    def filling_missing(data:pd.DataFrame):
        if data.isnull().values.any():
            for i in data.columns.values:
                if pd.isnull(data[i]).any():
                    if data[i].dtype in ['object','category']:
                        data[i]=data[i].fillna(data[i].mode())
                    else:
                        data[i]=data[i].fillna(data[i].mean())
        return data
    
    #encoding the category column
    def encoding_column(data:pd.DataFrame,ohe=None,user=None):
        col=data.select_dtypes(include=['object','category']).columns
        if ohe is None:
            ohe=OneHotEncoder(sparse_output=False)
            encode=ohe.fit_transform(data[col])
        else:
            encode=ohe.transform(data[col])
        encode_df=pd.DataFrame(encode,columns=ohe.get_feature_names_out(col))
        data.drop(columns=col,axis=1,inplace=True)
        df1=pd.concat([data,encode_df],axis=1)
        
        if user is None:
            return df1,ohe
        else:
            return df1

    data_filling=filling_missing(data)
    data_filling_encoding,ohe=encoding_column(data_filling)
    
    x=data_filling_encoding.drop(columns="price",axis=1)
    y=data_filling_encoding['price']
    
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=40)
    
    lr=LinearRegression()
    lr.fit(x_train,y_train)
    
    print(lr.score(x_train,y_train)*100)
    print(lr.score(x_test,y_test)*100)
    # get current script folder path
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # save inside same folder as script
    joblib.dump(lr, os.path.join(current_dir, "model.pkl"))
    joblib.dump(ohe, os.path.join(current_dir, "encoder.pkl"))

    print("Model and Encoder saved successfully!")