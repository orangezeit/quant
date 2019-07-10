from selenium import webdriver
import pandas as pd
import numpy as np
import time
import datetime


def get_tickers():

    browser = webdriver.Chrome()

    try:
        browser.get('https://finance.yahoo.com/most-active?offset=50&count=50')
        tickers = [element.text for element in browser.find_elements_by_css_selector("[class='Fw(600)']")]
        return tickers
    finally:
        browser.close()


def download_crawler(tickers, ts1, ts2):

    browser = webdriver.Chrome()

    try:
        for ticker in tickers:
            url = 'https://finance.yahoo.com/quote/{:s}/history?period1={:d}&period2={:d}'.format(ticker, ts1, ts2)
            browser.get(url)
            time.sleep(10)
            browser.find_element_by_css_selector("[class='Fl(end) Mt(3px) Cur(p)']").click()
            time.sleep(10)
    finally:
        browser.close()


if __name__ == '__main__':
    tks = ['BA', 'CSCO', 'DHI', 'DIS', 'JNJ', 'JPM', 'KO', 'MSFT', 'NEE', 'XOM', 'SPY']
    # download_crawler(tickers)
    start_ts = datetime.datetime.strptime('01/01/1998', "%d/%m/%Y").timestamp()
    end_ts = datetime.datetime.strptime('31/12/2001', "%d/%m/%Y").timestamp()
    download_crawler(tks, int(start_ts), int(end_ts))
