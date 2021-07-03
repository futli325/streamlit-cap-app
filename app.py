import streamlit as st
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from category_encoders import TargetEncoder
import time

def load_data():
   df = pd.read_csv("ModelData_product.csv")
   df = df.loc[:, ~df.columns.str.contains('Unnamed: 0')]
   return df
def preprocessing(df):
   encoder = TargetEncoder()
   df['MainColorGroupDesc_en'] = encoder.fit_transform(df['MainColorGroupDesc'], df['ReturnRate'])
   encoder3 = TargetEncoder()
   df['USSize_en'] = encoder3.fit_transform(df['USSize'], df['ReturnRate'])
   ProductDivision_dummy = pd.get_dummies(df.ProductDivision)
   df = pd.concat([df, ProductDivision_dummy], axis=1)
   ERPMasterGender_dummy = pd.get_dummies(df.ERPMasterGender)
   df = pd.concat([df, ERPMasterGender_dummy], axis=1)
   x = df.drop(columns='ReturnRate')
   y = df[['ReturnRate']]
   y = y.values.ravel()
   x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2021)
   x_train = x_train.drop(columns=['MainColorGroupDesc',
                                    'ProductDivision',
                                    'ERPMasterGender',
                                    'USSize', 'ReturnQuantity',
                                    'ComfortRating',
                                    'SizeRating',
                                    'WidthRating'])
   x_test_output = x_test.drop(columns=[
        'ProductDivision',
        'ERPMasterGender',
        'ReturnQuantity',
        'ComfortRating',
        'SizeRating',
        'WidthRating'])
   x_test = x_test.drop(columns=[
        'MainColorGroupDesc',
        'ProductDivision',
        'ERPMasterGender',
        'USSize', 'ReturnQuantity',
        'ComfortRating',
        'SizeRating',
        'WidthRating'])
   return x_train, x_test, y_train, y_test, x_test_output

def dic(df):
   Product_Type_name = df["ProductDivision"].unique()
   Product_Gender_name = df["ERPMasterGender"].unique()
   Product_Color_name = df["MainColorGroupDesc"].unique()
   Product_Size_name = df["USSize"].unique()
   MainColorGroupDesc_dic = df[['MainColorGroupDesc', 'MainColorGroupDesc_en']]
   MainColorGroupDesc_dic = MainColorGroupDesc_dic.drop_duplicates()
   color_dic = pd.Series(MainColorGroupDesc_dic.MainColorGroupDesc_en.values,
                         index=MainColorGroupDesc_dic.MainColorGroupDesc).to_dict()
   USSize_dic = df[['USSize', 'USSize_en']]
   USSize_dic = USSize_dic.drop_duplicates()
   USSize_dic.head()
   Size_dic = pd.Series(USSize_dic.USSize_en.values, index=USSize_dic.USSize).to_dict()
   return color_dic, Size_dic, Product_Type_name, Product_Gender_name, Product_Color_name, Product_Size_name

@st.cache(allow_output_mutation=True)
def RandomForest(x_train, x_test, y_train, y_test):
    # Train the model

    RF = RandomForestRegressor(n_estimators=1000,
                                min_samples_split= 10,
                                min_samples_leaf= 4,
                                max_features= 'auto',
                                max_depth= 10,
                                bootstrap= True)
    RF.fit(x_train, y_train)
    RF_pred = RF.predict(x_test)
    MAE = metrics.mean_absolute_error(y_test, RF_pred)
    MSE = metrics.mean_squared_error(y_test, RF_pred)
    RMSE = metrics.mean_squared_error(y_test, RF_pred)
    Rsquare = r2_score(y_test, RF_pred)
    return MAE, MSE, RMSE, Rsquare, RF

def Calculate_New_Transaction_Return(color_dic, Size_dic, ProductLine,
                                     Gender,
                                     Color,
                                     Size,
                                     UnitPrice,
                                     SalesQuantity,
                                     ReviewsCount,
                                     OverallRating):
    # set Footwear variable
    if ProductLine == "Footwear":
        Footwear = 1
    else:
        Footwear = 0

    # set KIDS variable
    if Gender == "KIDS":
        KIDS = 1
    else:
        KIDS = 0
    if ProductLine == "Apparel":
        Apparel = 1
    else:
        Apparel = 0

    # set KIDS variable
    if Gender == "WNS":
        WNS = 1
    else:
        WNS = 0
    if ProductLine == "Accessories":
        Accessories = 1
    else:
        Accessories = 0

    # set KIDS variable
    if Gender == "MNS":
        MNS = 1
    else:
        MNS = 0
    Color = color_dic[Color]
    # set USSize_en variable
    Size = Size_dic[Size]

    # new test data
    new_test_data = np.array([[UnitPrice, SalesQuantity, OverallRating, ReviewsCount, Color, Size, Accessories, Apparel,
                               Footwear, KIDS, MNS, WNS]])
    return new_test_data


def main():
    st.title("Prediction Product Return Rate")
    st.subheader("Using Machine Learning Regression Algorithms")
    data = load_data()
    x_train, x_test, y_train, y_test, x_test_output = preprocessing(data)
    # check-box to show the merged clean data

    if st.checkbox('Show Raw Data'):
        st.write("This Dataset was Merged and Cleaned")
        st.write(data.head())


    if st.checkbox('Show Model Performance'):
        st.write("This model is build by Random forest Algorithm")
        MAE, MSE, RMSE, Rsquare, RF = RandomForest(x_train, x_test, y_train, y_test)
        st.text("Mean Absolute Error MAE: ")
        st.write(MAE)
        st.text("Mean Squared Error MSE: ")
        st.write(MSE)
        st.text("Root Mean Squared Error RMSE:")
        st.write(RMSE)
        st.text("R^2:")
        st.write(Rsquare)


    if (st.checkbox("Want to predict on your selected Input? ")):

        color_dic, Size_dic, Product_Type_name, Product_Gender_name, Product_Color_name, Product_Size_name = dic(data)
        number_UnitPrice = st.slider('UnitPrice', step=1, min_value=0, max_value=600)
        number_SalesQuanity = st.number_input('SalesQuanity', step=2, min_value=0, max_value=1904)
        number_ReviewCount = st.slider('Product Reviews Count', step=1, min_value=0, max_value=50)
        number_OverallRating = st.slider('Online Overall Rating', step=1, min_value=0, max_value=5)
        number_MainColorGroupDesc = st.selectbox('Product MainColor', Product_Color_name)
        number_USSize = st.selectbox('Product Size', Product_Size_name)
        option_ERPMasterGender = st.selectbox('Gender Category?', Product_Gender_name)
        option_ProductDivision = st.selectbox('Product Line?', Product_Type_name)

        new_test_data = Calculate_New_Transaction_Return(color_dic, Size_dic, option_ProductDivision,
                                                                 option_ERPMasterGender, number_MainColorGroupDesc,
                                                                 number_USSize, number_UnitPrice, number_SalesQuanity,
                                                                 number_ReviewCount, number_OverallRating)
        if st.button("Predict"):
            with st.spinner('Wait for it...'):
               time.sleep(2)
            result = RF.predict(new_test_data)
            st.success('The Reture Rate is {}'.format(result))
if __name__ == "__main__":
    main()
