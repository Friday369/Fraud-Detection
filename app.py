import pandas as pd
import requests
import streamlit as st
import yaml

from src.lookup import get_account


def load_config():
    with open("config.yaml") as file:
        return yaml.safe_load(file)


@st.cache_data
def load_table(path):
    return pd.read_csv(path)


def call_api(api_url, account):
    return requests.post(api_url + "/score", json=account, timeout=5)


def show_result(result):
    st.metric("Suspicion score", f"{result['suspicion_score']:.4f}")
    if result["flagged"]:
        st.error("FLAGGED")
    else:
        st.success("Not flagged")
    st.write("Reason: " + result["explanation"])

def score_account(account_id, table, api_url):
    if account_id.strip() == "":
        st.warning("Please type an account ID.")
        return

    account = get_account(account_id, table)
    if account is None:
        st.error(f"Account '{account_id.strip().upper()}' was not found.")
        return

    try:
        response = call_api(api_url, account)
    except requests.exceptions.RequestException:
        st.error("Cannot reach the scoring API. Start it with: uvicorn src.api:app")
        return

    if response.status_code != 200:
        st.error(f"The API replied with an error (status {response.status_code}).")
        return

    show_result(response.json())


config = load_config()
table = load_table(config["paths"]["features"])
api_url = config["api"]["url"]

st.title("Trading Account Review")
st.title("Trading Account Review")

with st.form("score_form"):
    account_id = st.text_input("Account ID", placeholder="ACC00267")
    submitted = st.form_submit_button("Score")

if submitted:
    score_account(account_id, table, api_url)