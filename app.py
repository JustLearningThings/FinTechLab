import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.models import load_model

from sklearn.preprocessing import StandardScaler

class Dataset:
    def __init__(self, location: str = None) -> None:
        if location:
            self.df = pd.read_excel(location, header=2)    

        self.__AGE_BINS = [0, 18, 30, 40, 50, 60, 100]
        self.__AGE_LABELS = ['0-18', '19-30', '31-40', '41-50', '51-60', '61+']

        self.class_weights = None
        self.__scaler = StandardScaler()

        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None

    def __preprocess_income(self, precision: int = 8) -> 'pd.DataFrame':
        EPS = .1**(precision)
        
        self.df['Income'] = np.abs((self.df['DeclIncome'] - self.df['ANAFIncome']) / (self.df['ANAFIncome'] + EPS))

        return self.df
    
    def __create_age_bins(self) -> 'pd.DataFrame':
        self.df['Age'] = pd.cut(self.df['Age'], bins=self.__AGE_BINS, labels=self.__AGE_LABELS, right=False)
        self.df['Age'] = self.df['Age'].map({'0-18': 0, '19-30': 2, '31-40': 5, '41-50': 4, '51-60': 3, '61+': 1})

        return self.df
    
    def __create_contract_duration(self) -> 'pd.DataFrame':
        self.df['ContractDuration'] = self.df['DataInchis'] - self.df['DataSemnarii']
        current_date = datetime.now()
        self.df['ContractDuration'] = self.df['ContractDuration'].fillna(current_date - self.df['DataSemnarii'])

        return self.df
    
    def __encode_gender(self) -> 'pd.DataFrame':
        self.df['Gender'] = self.df['Gender'].map({'F': 0, 'M': 1})

        return self.df
    
    def __one_hot_product_type(self) -> 'pd.DataFrame':    
        encoded = pd.get_dummies(self.df['Produs'], prefix='Product')

        for col in encoded.columns:
            self.df[col] = pd.Series([0] * encoded.shape[0])
        
        self.df.update(encoded)

        return self.df
    
    def __encode_credit_status(self) -> 'pd.DataFrame':
        self.df['State'] = self.df['State'].map({'Executare': 0, 'Moneysend': 2, 'Activ': 1, 'Inchis': 3})

        return self.df
    
    # note: used ANAFIncome instead of DeclIncome to avoid correlation between this feature and Income
    def __create_income_to_credit_limit_ratio(self) -> 'pd.DataFrame':
        self.df['IncomeToCreditLimit'] = self.df['ANAFIncome'] / self.df['CreditLimit']

        return self.df
    
    def __create_total_loan_payments_to_income_ratio(self) -> 'pd.DataFrame':
        self.df['TotalLoanPaymentsToIncome'] = self.df['TotalLoanPayments'] / self.df['DeclIncome']

        return self.df
    
    # amount needed to pay / amount paid
    def __create_debt_ratio(self, precision: int = 8) -> 'pd.DataFrame':
        EPS = .1**(precision)

        safe_paid_total = self.df['PaidTotal'].replace(to_replace=0, value=EPS)
        self.df['DebtRatio'] = np.abs(self.df['DpdTotal'] / safe_paid_total)

        return self.df
    
    def __encode_derrogation(self) -> 'pd.DataFrame':
        self.df['IsDerrogationBNR'].fillna(0, inplace=True)

        return self.df

    # select days from dates
    def __transform_dates(self) -> 'pd.DataFrame':
        self.df['ContractDuration'] = pd.Series([date.days for date in self.df['ContractDuration']])

        return self.df
    
    def __revert_diff_days(self) -> 'pd.DataFrame':
        self.df['DpdDiffDaysMax'] = -self.df['DpdDiffDaysMax']

        return self.df
    
    def __encode_target(self) -> 'pd.DataFrame':
        self.df.loc[self.df['ClientCategory'] == 3, 'ClientCategory'] = 0
        self.df.loc[(self.df['ClientCategory'] == 2) & (self.df['IsDerrogationBNR'] == 1.0) & (self.df['score'] > 450), 'ClientCategory'] = 1
        self.df.loc[self.df['ClientCategory'] == 2, 'ClientCategory'] = 0

        return self.df
    
    def __normalize(self) -> 'pd.DataFrame':
        df_columns = list(self.df.columns)
        df_columns.remove('ClientCategory')
        X, y = self.df.drop(columns=['ClientCategory']), self.df['ClientCategory']

        self.df = self.__scaler.fit_transform(X.values, y.values)
        self.df = pd.DataFrame(self.df, columns=df_columns)
        self.df['ClientCategory'] = pd.Series(y)

        return self.df
    
    def __normalize_for_predictions(self) -> 'pd.DataFrame':
        self.df = pd.DataFrame(self.__scaler.fit_transform(self.df.values), columns=self.df.columns)

        return self.df
    
    def preprocess(self, remove_columns: bool = True, normalize: bool = True, ignore_target: bool = False) -> None:
        self.__create_age_bins()
        self.__create_contract_duration()
        self.__encode_gender()
        self.__one_hot_product_type()
        self.__encode_credit_status()
        self.__create_income_to_credit_limit_ratio()
        self.__create_total_loan_payments_to_income_ratio()
        self.__create_debt_ratio() # DPDTotal / PaidTotal
        self.__preprocess_income()
        self.__encode_derrogation()
        self.__transform_dates()
        self.__revert_diff_days() # to establish positive relationship between target and this feature
        
        if not ignore_target:
            self.__encode_target()

        if remove_columns:
            self.df.drop(columns=['CNP', 'DataSemnarii', 'DataInchis', 'Number', 'Produs', 'CreditLimit', 'DeclIncome', 'scoringdate', 'TotalLoanPayments', 'PaidTotal', 'DpdTotal'], inplace=True)    
        
        if normalize:
            if ignore_target:
                self.__normalize_for_predictions()
            else:
                self.__normalize()
    

    def split(self, create_class_weights: bool = True) -> None:        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.df.drop(columns=['ClientCategory']), 
            self.df['ClientCategory'], 
            test_size=.2, 
            random_state=42)
        
        if create_class_weights:
            self.class_weights = {
                0: len(self.y_train[self.y_train == 0]) / len(self.y_train[self.y_train == 1]),
                1: 1
            }


def get_inputs():
    cnp = st.text_input("CNP")
    gender = st.selectbox("Gender", ['M', 'F'])
    age = st.number_input("Age")

    number = st.text_input("Number")
    produs = st.selectbox("Product", ['Penguin', 'Dolphin', 'Crab'])
    credit_limit = st.number_input("Credit Limit")
    state = st.selectbox("Credit State", ['Activ', 'Executare', 'Moneysend', 'Inchis'])

    scoring_date = st.date_input("Scoring Date")
    scoring_date = datetime.combine(scoring_date, datetime.min.time())
    scoring_date.strftime('%d/%m/%Y')
    scoring_date = pd.to_datetime(scoring_date, format='%d/%m/%Y')

    contract_duration = st.number_input('Contract duration (days)')

    score = st.number_input("Score")
    probability_of_model = st.number_input("Probability of Model")
    DeclIncome = st.number_input("Declared Income")
    ANAFIncome = st.number_input("ANAF Income")
    tlp = st.number_input("Total Loan Payments")
    BNR40Available = ANAFIncome * .4
    credits_before = st.number_input("Credits Before")
    offer_crab = st.number_input("Offer Crab")
    offer_dolphin = st.number_input("Offer Dolphin")
    offer_penguin = st.number_input("Offer Penguin")
    crab_ignoring = st.number_input("Crab Ignoring")
    dolphin_ignoring = st.number_input("Dolpin Ignoring")
    penguin_ignoring = st.number_input("Penguin Ignoring")
    commission = st.selectbox("Comission", [0, 5, 7, 9])
    withdrawed = st.number_input("Withdrawn")
    withdrawed = -withdrawed
    dpd = st.number_input("DpdDiffDaysMax")
    dpd_total = st.number_input("DpdTotal")
    paid = st.number_input("Paid Total")
    future = st.number_input("Future Total")
    derrogation = st.checkbox('Derrogation')

    dataframe = pd.DataFrame({
        'Gender': [gender],
        'Age': [age],
        'State': [state],
        'score': [score],
        'ProbabilityOfModel': [probability_of_model],
        'ANAFIncome': [ANAFIncome],
        'BNR40Available': [BNR40Available],
        'CreditsBefore': [credits_before],
        'OfferCrab': [offer_crab],
        'OfferPenguin': [offer_penguin],
        'OfferDolphin': [offer_dolphin],
        'CrabIgnoringBNR': [crab_ignoring],
        'PenguinIgnoringBNR': [penguin_ignoring],
        'DolphinIgnoringBNR': [dolphin_ignoring],
        'Comission': [commission],
        'Withdrawed': [withdrawed],
        'DpdDiffDaysMax': [dpd],
        'FutureTotal': [dpd_total],
        'IsDerrogationBNR': [derrogation],
        'ContractDuration': [contract_duration],
        'Product_Crab': [None],
        'Product_Dolphin': [None],
        'Product_Penguin': [None],
        'IncomeToCreditLimit': [None],
        'TotalLoanPaymentsToIncome': [None],
        'DebtRatio': [None],
        'Income': [None]
    })

    if produs == 'Dolphin':
        dataframe['Product_Dolphin'] = [1.]
        dataframe['Product_Penguin'] = [0.]
        dataframe['Product_Crab'] = [0.]
    elif produs == 'Penguin':
        dataframe['Product_Dolphin'] = [0.]
        dataframe['Product_Penguin'] = [1.]
        dataframe['Product_Crab'] = [0.]
    else:
        dataframe['Product_Dolphin'] = [0.]
        dataframe['Product_Penguin'] = [0.]
        dataframe['Product_Crab'] = [1.]
    
    dataframe['IncomeToCreditLimit'] = ANAFIncome / credit_limit
    dataframe['TotalLoanPaymentsToIncome'] = tlp / DeclIncome
    
    EPS = .1**(8)

    if paid == 0:
        paid = EPS

    dataframe['DebtRatio'] = np.abs(dpd_total / paid)
    
    dataframe['Income'] = np.abs((DeclIncome - ANAFIncome) / (ANAFIncome + EPS))

    return dataframe


@st.cache
def process_input(dataframe, ds):
    # preprocess
    AGE_BINS = [0, 18, 30, 40, 50, 60, 100]
    AGE_LABELS = ['0-18', '19-30', '31-40', '41-50', '51-60', '61+']

    dataframe['Age'] = pd.cut(dataframe['Age'], bins=AGE_BINS, labels=AGE_LABELS, right=False)
    dataframe['Age'] = dataframe['Age'].map({'0-18': 0., '19-30': 2., '31-40': 5., '41-50': 4., '51-60': 3., '61+': 1.})
    dataframe['Gender'] = dataframe['Gender'].map({'F': 0., 'M': 1.})
    dataframe['State'] =dataframe['State'].map({'Executare': 0., 'Moneysend': 2., 'Activ': 1., 'Inchis': 3.})
    dataframe['IsDerrogationBNR'] = int(dataframe['IsDerrogationBNR'] == True)
    dataframe['DpdDiffDaysMax'] = -dataframe['DpdDiffDaysMax']

    for col in dataframe:
        dataframe[col] = dataframe[col].astype('float64')

    for col in ds:
        ds[col] = ds[col].astype('float64')

    # standardize    
    ds = ds.reindex(columns=dataframe.columns)
    std = pd.DataFrame(ds.std()).T
    mean = pd.DataFrame(ds.mean()).T

    print('======================================================')
    print(ds['ProbabilityOfModel'])
    print(((dataframe - mean) / std)['ProbabilityOfModel'])
    print(dataframe['ProbabilityOfModel'])
    print(std['ProbabilityOfModel'])
    print(mean['ProbabilityOfModel'])

    dataframe = (dataframe - mean) / std
    dataframe['ProbabilityOfModel'] = ds['ProbabilityOfModel']

    return dataframe


def main():
    model_dataset = Dataset('./data.xlsx')
    model = load_model('./model.h5')

    model_dataset.preprocess(normalize=False)

    st.title('FinTech Lab')

    df = get_inputs()
    df = process_input(df, model_dataset.df.drop(columns=['ClientCategory']))
    df = df.reindex(columns=model_dataset.df.columns)
    df = df.drop(columns=['ClientCategory'])
    df
    
    
    prediction = model.predict(df.values)[0]
    
    st.write(f'Prediction: {"Risky client" if prediction[0] >= .6 else "Not risky client"}')
    

if __name__ == '__main__':    
    main()