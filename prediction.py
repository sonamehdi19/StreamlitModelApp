#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 22:38:36 2021

@author: sonamehdizade
"""
import matplotlit.pylot as plt
import seaborn as sns 
# adding some plots for visual analysis
import pandas as pd
import streamlit as st
import plotly.express as px
import os 
import numpy as np
from sklearn.impute import SimpleImputer
from PIL import Image
from imblearn.under_sampling import RandomUnderSampler 

icon = Image.open("icon.png")
st.set_page_config(layout = "wide", page_title = "Prediction Application", page_icon=icon)

# front end elements of the web page 
html_temp = """ 
    <div style ="background-color:darkblue;padding:px"> 
    <h1 style ="color:white;text-align:center;">Streamlit Case Study: Prediction ML App</h1> 
    </div> 
    """

# display the front end aspect
st.markdown(html_temp, unsafe_allow_html = True) 


ban = Image.open("banner.jpg")
lg= Image.open("logo.png")


st.sidebar.image(lg, use_column_width="always")
pg = st.sidebar.selectbox("", ["Homepage", "EDA", "Modeling"])


if pg == "Homepage":
    
    st.header("Homepage")
    st.image(ban, use_column_width="always")
    
    dataset = st.selectbox("Select dataset", ["Loan Prediction", "Water Probability"])
    st.markdown("Selected: {} dataset".format(dataset))
            
    if dataset == "Loan Prediction":
        st.subheader("""Given Data""")
        df = pd.read_csv("loan_prediction.csv")
        st.dataframe(df)

        
        st.header("Task")
        st.info("""Dream Housing Finance company deals in all home loans. 
                   They have presence across all urban, semi urban and rural areas. 
                   Customer first apply for home loan after that company validates the customer eligibility for loan. 
                   Company wants to automate the loan eligibility process (real time) based on customer detail provided while 
                   filling online application form. These details are Gender, Marital Status, Education, Number of Dependents, Income, Loan Amount, Credit History and others. To automate this process, they have given a problem to identify the customers segments, 
                   those are eligible for loan amount so that they can specifically target these customers.""")

        st.header("Data understanding: Variables")

        st.info("""
            Loan_ID : Unique Loan ID

            Gender : Male/ Female

            Married : Applicant married (Y/N)

            Dependents : Number of dependents 

            Education : Applicant Education (Graduate/ Under Graduate)  

            Self_Employed : Self employed (Y/N)

            ApplicantIncome : Applicant income

            CoapplicantIncome : Coapplicant income

            LoanAmount : Loan amount in thousands of dollars

            Loan_Amount_Term : Term of loan in months

            Credit_History : credit history meets guidelines yes or no

            Property_Area : Urban/ Semi Urban/ Rural

            Loan_Status : Loan approved (Y/N) this is the target variable""")     

    else:
        st.header("Task")
        st.subheader("""Given Data""")
        df = pd.read_csv("water_potability.csv")
        st.dataframe(df)
        
        st.info("""Access to safe drinking-water is essential to health,
                   a basic human right and a component of effective policy for health protection. 
                   This is important as a health and development issue at a national, regional and local level. 
                   In some regions, it has been shown that investments in water supply and sanitation can yield a net economic benefit, 
                   since the reductions in adverse health effects and health care costs outweigh the costs of undertaking the interventions. 
                   The task is to predict water quality based on given parameters""")
                   
        st.header("Data understanding: Variables")
        st.info("""
                pH value: PH is an important parameter in evaluating the acid–base balance of water. The current investigation ranges were 6.52–6.83 which are in the range of WHO standards.

                Hardness: Hardness is mainly caused by calcium and magnesium salts.Hardness was originally defined as the capacity of water to precipitate soap caused by Calcium and Magnesium.

                Solids (Total dissolved solids - TDS): Water has the ability to dissolve a wide range of inorganic and some organic minerals or salts such as potassium, calcium, sodium, bicarbonates, chlorides, magnesium, sulfates etc. Desirable limit for TDS is 500 mg/l and maximum limit is 1000 mg/l which prescribed for drinking purpose.

                Chloramines: Chlorine and chloramine are the major disinfectants used in public water systems. Chlorine levels up to 4 milligrams per liter (mg/L or 4 parts per million (ppm)) are considered safe in drinking water.

                Sulfate: Sulfates are naturally occurring substances that are found in minerals, soil, and rocks. They are present in ambient air, groundwater, plants, and food.

                Conductivity: Pure water is not a good conductor of electric current rather’s a good insulator. According to WHO standards, EC value should not exceeded 400 μS/cm.

                Organic_carbon: Total Organic Carbon (TOC) in source waters comes from decaying natural organic matter (NOM) as well as synthetic sources. TOC is a measure of the total amount of carbon in organic compounds in pure water. According to US EPA < 2 mg/L as TOC in treated / drinking water, and < 4 mg/Lit in source water which is use for treatment.

                Trihalomethanes: THMs are chemicals which may be found in water treated with chlorine. The concentration of THMs in drinking water varies according to the level of organic material in the water, the amount of chlorine required to treat the water, and the temperature of the water that is being treated. THM levels up to 80 ppm is considered safe in drinking water.

                Turbidity: The turbidity of water depends on the quantity of solid matter present in the suspended state. The mean turbidity value obtained for Wondo Genet Campus (0.98 NTU) is lower than the WHO recommended value of 5.00 NTU.

                Potability: Indicates if water is safe for human consumption where 1 means Potable and 0 means Not potable.""")        

elif pg== "EDA":
    st.header("Explatory Data Analysis")
    dataset = st.selectbox("Select a dataset", ["Loan Prediction", "Water Probability"])
    st.markdown("Selected: {} dataset".format(dataset))


    def outlier_treatment(datacol):
        sorted(datacol)
        q1, q3 = np.percentile(datacol,  [25,75])
        iqr = q3-q1
        lower_range = q1 - (1.5 * iqr)
        upper_range = q3 + (1.5 * iqr)
        return lower_range, upper_range
    
    def data_describtion(df):

        st.subheader("Initial Target balance")
        st.bar_chart(df.iloc[:,-1].value_counts())
        
        null_values_count= df.isnull().sum().to_frame().reset_index()
        null_values_count.columns = ["Columns", "Counts"]
        
        c1, c2, c3 = st.columns([4,5,4,])
        
        c1.subheader("Null values")
        c1.dataframe(null_values_count)
        
        c2.subheader("Missing values imputation")
        cat_method = c2.radio("Categorical", ["Mode", "Backfill", "Ffill"])
        num_method = c2.radio("Numerical", ["Mode", "Median"])
        
        c2.subheader("Feature engineering")
        balance = c2.checkbox("Handle imbalance")
        outlier = c2.checkbox("Coarse Outliers")
        
        if c2.button("Data Preprocessing"):
            
            cat_array = df.iloc[:, :-1].select_dtypes(include="object").columns
            num_array = df.iloc[:, :-1].select_dtypes(exclude="object").columns
            
            
            if cat_array.size > 0:
                if cat_method =="Mode":
                    imp_cat = SimpleImputer(missing_values = np.nan, strategy = "most_frequent")
                    df[cat_array] = imp_cat.fit_transform(df[cat_array])
                elif cat_method == "Backfill":
                    df[cat_array].fillna(method = "backfill", inplace = True)
                else:
                    df[cat_array].fillna(method = "ffill", inplace = True)

            if num_array.size > 0:
                if num_method == "Mode":
                    imp_cat = SimpleImputer(missing_values = np.nan, strategy = "most_frequent")
                else:
                    pass

            if balance:
                rs = RandomUnderSampler()
                X = df.iloc[:, :-1]
                Y = df.iloc[:, [-1]]
                
                X, Y = rs.fit_resample(X, Y)
                df = pd.concat([X, Y ], axis= 1)
                
            if outlier:
                for col in num_array:
                    lowb, uppb = outlier_treatment(df[col])
                    df[col] = np.clip(df[col], a_min=lowb, a_max=uppb)

                    
        null_df = df.isnull().sum().to_frame().reset_index()
        null_df.columns = ["Column", "Count"]
        
        c3.subheader("Null value count after")
        c3.dataframe(null_df)
        st.subheader("Target balance after preprocessing")
        st.bar_chart(df.iloc[:, -1].value_counts())
        
        st.subheader("Conrrelation Map")
        heatmap = px.imshow(df.corr())
        st.plotly_chart(heatmap)

   

        html_temp = """ 
    <div style ="background-color:darkblue;padding:10px"> 
    <h4 style ="color:yellow;text-align:center;">Final version of data after preparation </h4> 
    </div> 
    """

    # display the front end aspect
        st.markdown(html_temp, unsafe_allow_html = True) 

        st.dataframe(df)
        
        if os.path.exists("model.csv"):
            os.remove("model.csv")
        df.to_csv("model.csv", index= False)


    if dataset == "Loan Prediction":
        df = pd.read_csv("loan_prediction.csv")
        st.subheader("""Initial Data""")
        st.dataframe(df)
        data_describtion(df)
        
    else:
        df = pd.read_csv("water_potability.csv")
        st.subheader("""Initial Data""")
        st.dataframe(df)
        data_describtion(df)
        
    
    
else:
    st.header("Modeling")
    if not os.path.exists("model.csv"):
        st.header("Run Preprocessing Please")
        
    else:
        
        df = pd.read_csv("model.csv")
        st.dataframe(df)
        
        c1, c2 = st.columns([3, 3])
        
        c1.subheader("Scaling")
        scaler = c1.radio("", ["Standart", "Robust", "MinMax"])

        c2.subheader("Encoding")
        en_mod = c2.radio("", ["Label", "One-Hot"])
        
        st.header("Train Test Split")
        c11, c22 = st.columns([3, 3])
        X = df.iloc[:, :-1]
        Y = df.iloc[:, [-1]]
                
        cat_array = df.iloc[:, :-1].select_dtypes(include="object").columns
        num_array = df.iloc[:, :-1].select_dtypes(exclude="object").columns
        
        if num_array.size > 0:
            if scaler == "Standart":
                from sklearn.preprocessing import StandardScaler
                sc = StandardScaler()

            if scaler == "Robust":
                from sklearn.preprocessing import RobustScaler
                sc = RobustScaler()

            if scaler == "MinMax":
                from sklearn.preprocessing import MinMaxScaler
                sc = MinMaxScaler()
                            
        if cat_array.size > 0:
            if en_mod == "Label":
                from sklearn.preprocessing import LabelEncoder
                lb = LabelEncoder()
                for col in cat_array:
                    df[col] = lb.fit_transform(df[col])
                    
                    
            else:
                df.drop(df.iloc[:, [-1]], axis= 1, inplace=True)
                d_df = df[cat_array]
                d_df = pd.get_dummies(d_df, drop_first = True)
                df_ = df.drop(cat_array, axis=1)
                df = pd.concat([df_, d_df, Y], axis = 1)
                
        st.dataframe(df)

        
        test_size, random_state = st.columns([2,2])
        test_size = test_size.number_input("Test size: ", min_value=0.1)
        random_state = random_state.number_input('Random state: ')
        random_state = round(int(random_state))
        X = df.iloc[:, :-1]
        Y = df.iloc[:, [-1]]
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=float(test_size), random_state = random_state)
        
        st.markdown("X_train size = {}".format(X_train.shape))
        st.markdown("X_test size = {}".format(X_test.shape))
        
        st.subheader("Model is working")
        
        model = st.selectbox("Select a model", ["XGBoost", "CatBoost"])
        
        if model == "XGBoost":
            import xgboost as xgb
            model = xgb.XGBClassifier().fit(X_train, y_train)
   
        else:
            from catboost import CatBoostClassifier
            model = CatBoostClassifier().fit(X_train, y_train)
#add more models for selection        
        preds = model.predict(X_test)
        from sklearn.metrics import accuracy_score
        from sklearn.metrics import confusion_matrix
        
        confusion = confusion_matrix(y_test, preds)
         
        accuracy = accuracy_score(y_test, preds)

        html_temp = """ 
    <div style ="background-color:darkblue;padding:px"> 
    <h3 style ="color:white;text-align:center;">Evaluation of the model </h3> 
    </div> 
    """

    # display the front end aspect
        st.markdown(html_temp, unsafe_allow_html = True) 

        st.subheader("Accuracy score is: {}".format(round(accuracy, 2)))
        ss = pd.DataFrame(confusion)
        st.subheader("Confusion Matrix")
        st.dataframe(ss)
