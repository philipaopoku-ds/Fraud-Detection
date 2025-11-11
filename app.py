import streamlit as st
import pandas as pd
import numpy as np
import pickle

ALL_FEATURES = ['BatchId', 'AccountId', 'SubscriptionId', 'CustomerId', 'ProviderId',
       'ProductId', 'ProductCategory', 'ChannelId', 'Value', 'PricingStrategy',
       'Month', 'Day', 'Credit_or_Debit']

CATEGORICAL_COLS = ['BatchId',
 'AccountId',
 'SubscriptionId',
 'CustomerId',
 'ProviderId',
 'ProductId',
 'ProductCategory',
 'ChannelId']

@st.cache_resource #tells Streamlit to load the model once and resuse them
def load_assets():
    model = pickle.load(open("fraud_model.pkl", "rb"))
    encoder = pickle.load(open("fraud_encoder.pkl", "rb"))
    scaler = pickle.load(open("fraud_scaler.pkl", "rb"))

    return model, encoder, scaler

model, encoder, scaler = load_assets()


def main():
    st.title("Fraud Detection App")
    #Organize inputs in a simple sidebar
    with st.sidebar:
        st.header("Input Transaction Details")
        value = st.number_input("Transaction Value", min_value=0.0, value=100.0)
        month =st.slider("Transaction Month", min_value=1, max_value=12, value=6)
        day = st.slider("Transaction Day", min_value=1, max_value=31, value=15)
        batch_id = st.selectbox('BatchId', ("BatchId_36123"))
        account_id = st.selectbox('AccountId', ("AccountId_3957"))
        subscription_id = st.selectbox('SubscriptionId', ("SubscriptionId_12345"))
        customer_id = st.selectbox('CustomerId', ("CustomerId_67890"))
        provider_id = st.selectbox('ProviderId', ("ProviderId_54321"))
        product_id = st.selectbox('ProductId', ("ProductId_98765"))
        product_category = st.selectbox('ProductCategory', ('airtime', 'financial_services', 'utility_bill', 'data_bundles',
       'tv', 'transport', 'ticket', 'movies', 'other'))
        channel_id = st.selectbox('ChannelId', ('ChannelId_3', 'ChannelId_2', 'ChannelId_1', 'ChannelId_5'))
        pricing_strategy = st.selectbox('PricingStrategy (where 2 is online)', (2, 4, 1, 0))
        credit_or_debit = st.selectbox('Credit_or_Debit (where 0 is Credit, 1 is Debit)', (0, 1))


    if st.button("Predict Fraud"):
        input_dict = {
            'BatchId': batch_id,
            'AccountId': account_id,
            'SubscriptionId': subscription_id,
            'CustomerId': customer_id,
            'ProviderId': provider_id,
            'ProductId': product_id,
            'ProductCategory': product_category,
            'ChannelId': channel_id,
            'Value': value,
            'PricingStrategy': pricing_strategy,
            'Month': month,
            'Day': day,
            'Credit_or_Debit': credit_or_debit
        }
        input_df = pd.DataFrame([input_dict])
        df_processed = input_df.copy()
        for col in CATEGORICAL_COLS:
            df_processed[col] = encoder.fit_transform(df_processed[[col]])
        
        all_input = df_processed[ALL_FEATURES]

        #extract the values into an array
        all_input_array = all_input.values

        scaled_input_array = scaler.transform(all_input_array)

        prediction = model.predict(scaled_input_array)

        st.subheader("Prediction Result")

        if prediction[0] == 1:
            st.error("The transaction is predicted to be FRAUDULENT.")
        else:
            st.success("The transaction is predicted to be LEGITIMATE.")






if __name__ == "__main__":
    main()