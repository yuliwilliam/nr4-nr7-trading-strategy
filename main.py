import pandas as pd
import matplotlib.pyplot as plt


def load_data(data_file_path):
    data = pd.read_csv(data_file_path,
                       usecols=['<TICKER>', '<DTYYYYMMDD>', '<TIME>', '<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>', '<VOL>'])
    return data


def get_everyday_data(data):
    data_list = []
    # data_size = data.shape[0]
    # smaller data size for testing purpose
    data_size = int(data.shape[0] * 0.01)
    for i in range(data_size):
        curr_day_ticker = data['<TICKER>'][i]
        curr_day_date = data['<DTYYYYMMDD>'][i]
        curr_day_high = data['<HIGH>'][i]
        curr_day_low = data['<LOW>'][i]
        if len(data_list) != 0 and data_list[-1]['DATE'] == curr_day_date:
            continue
        for j in range(i, data_size):
            curr_entry_date = data['<DTYYYYMMDD>'][j]
            curr_entry_high = data['<HIGH>'][j]
            curr_entry_low = data['<LOW>'][j]
            if curr_entry_date != curr_day_date:
                break
            if curr_entry_high > curr_day_high:
                curr_day_high = curr_entry_high
            if curr_entry_low < curr_day_low:
                curr_day_low = curr_entry_low
        curr_day = {'TICKER': curr_day_ticker, 'DATE': curr_day_date, 'HIGH': curr_day_high, 'LOW': curr_day_low}
        data_list.append(curr_day)
        print("processed ticker: {}, date: {}, high: {}, low: {}".format(curr_day_ticker, curr_day_date, curr_day_high,
                                                                         curr_day_low))

    return data_list


def test_result(data, buy_sell_prices, cash, stop_loss_pip, stop_profit_pip):
    profit_loss_data = {'DATE': [], 'PROFIT LOSS PERCENTAGE': [], 'PROFIT LOSS AMOUNT': []}
    for buy_sell_price in buy_sell_prices:
        initial_cash = cash
        position = 0
        holdings = 0
        indicator = ''
        buy_price = buy_sell_price['BUY']
        short_price = buy_sell_price['SHORT']
        curr_date = buy_sell_price['DATE']
        curr_day_data = data.loc[data['<DTYYYYMMDD>'] == curr_date]
        index = 0
        print("date: {}".format(curr_date))
        for i, curr_entry in curr_day_data.iterrows():
            curr_price = curr_entry['<OPEN>']

            # buy sell
            if indicator == '' and curr_price >= buy_price:
                position = curr_price
                holdings = int(cash / curr_price)
                cash = cash - curr_price * holdings
                indicator = 'b'
                print("    bought at {}".format(curr_price))
            if indicator == '' and curr_price <= short_price:
                position = curr_price
                holdings = int(cash / curr_price)
                cash = cash + curr_price * holdings
                indicator = 's'
                print("    shorted at {}".format(curr_price))

            # holdings in hand
            if position != 0 or holdings != 0:
                # check loss
                if indicator == 'b' and curr_price <= position - position * stop_loss_pip:
                    cash = cash + curr_price * holdings
                    position = 0
                    holdings = 0
                    print("    stop loss bought at {}".format(curr_price))

                elif indicator == 's' and curr_price >= position + position * stop_loss_pip:
                    cash = cash - curr_price * holdings
                    position = 0
                    holdings = 0
                    print("    stop loss shorted at {}".format(curr_price))

                # check profit
                elif indicator == 'b' and curr_price >= position + position * stop_profit_pip:
                    cash = cash + curr_price * holdings
                    position = 0
                    holdings = 0
                    print("    stop profit bought at {}".format(curr_price))

                elif indicator == 's' and curr_price <= position - position * stop_profit_pip:
                    cash = cash - curr_price * holdings
                    position = 0
                    holdings = 0
                    print("    stop profit shorted at {}".format(curr_price))

                # clean up, if last day
                elif index == curr_day_data.shape[0] - 1:
                    if indicator == 'b':
                        cash = cash + curr_price * holdings
                        position = 0
                        holdings = 0
                        print("    cleaned bought at {}".format(curr_price))
                    if indicator == 's':
                        cash = cash - curr_price * holdings
                        position = 0
                        holdings = 0
                        print("    cleaned shorted at {}".format(curr_price))
            index += 1
        profit_loss = cash - initial_cash
        profit_loss_percentage = profit_loss / initial_cash
        profit_loss_data['DATE'].append(str(curr_date))
        profit_loss_data['PROFIT LOSS PERCENTAGE'].append(profit_loss_percentage)
        profit_loss_data['PROFIT LOSS AMOUNT'].append(profit_loss)
        print("    cash on open: {}, cash on close: {}".format(initial_cash, cash))
        print("    profit/loss amount: {}, profit/loss rate: {}".format(profit_loss, profit_loss_percentage))

    df = pd.DataFrame(profit_loss_data, columns=['DATE', 'PROFIT LOSS PERCENTAGE', 'PROFIT LOSS AMOUNT'])
    print('overview:')
    print(df)
    ax = plt.gca()
    df.plot(kind='line', x='DATE', y='PROFIT LOSS PERCENTAGE', ax=ax)
    df.plot(kind='line', x='DATE', y='PROFIT LOSS AMOUNT', color='red', ax=ax)
    df.plot(kind='line', x='DATE', y='PROFIT LOSS PERCENTAGE')
    df.plot(kind='line', x='DATE', y='PROFIT LOSS AMOUNT')

    plt.show()
    return cash


if __name__ == '__main__':
    data_file_path = "./data/USD/USDMXN-1M-2008.1.1-2020.5.22.csv"
    cash = 10000
    pip = 0.0001
    stop_loss_pip = 10 * pip
    stop_profit_pip = 40 * pip
    data = load_data(data_file_path)
    everyday_data = get_everyday_data(data)
    buy_sell_prices = []

    # exclude last day
    for i in range(3, len(everyday_data) - 1):
        first_price_range = everyday_data[i - 3]['HIGH'] - everyday_data[i - 3]['LOW']
        second_price_range = everyday_data[i - 2]['HIGH'] - everyday_data[i - 2]['LOW']
        third_price_range = everyday_data[i - 1]['HIGH'] - everyday_data[i - 1]['LOW']
        fourth_price_range = everyday_data[i]['HIGH'] - everyday_data[i]['LOW']
        if min(first_price_range, second_price_range, third_price_range, fourth_price_range) == fourth_price_range \
                and everyday_data[i]['LOW'] >= everyday_data[i - 1]['LOW'] \
                and everyday_data[i]['HIGH'] <= everyday_data[i - 1]['HIGH']:
            transaction_date = everyday_data[i + 1]['DATE']
            buy_price = everyday_data[i]['HIGH'] + everyday_data[i]['HIGH'] * pip
            short_price = everyday_data[i]['LOW'] - everyday_data[i]['LOW'] * pip

            buy_sell_price = {'DATE': transaction_date, 'BUY': buy_price, 'SHORT': short_price}
            buy_sell_prices.append(buy_sell_price)
            print("NR4 day, data: {}, high: {}, low: {}, price range: {}".format(everyday_data[i]['DATE'],
                                                                                 everyday_data[i]['HIGH'],
                                                                                 everyday_data[i]['LOW'],
                                                                                 fourth_price_range))
    print("started testing with {} cash".format(cash))
    cash = test_result(data, buy_sell_prices, cash, stop_loss_pip, stop_profit_pip)
    print("ended testing with {} cash".format(cash))
