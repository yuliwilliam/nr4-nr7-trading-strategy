import pandas as pd
import matplotlib.pyplot as plt
import threading
from pathlib import Path


def load_data(data_file_path):
    data = pd.read_csv(data_file_path,
                       usecols=['<DTYYYYMMDD>', '<TIME>', '<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>'])
    return data


def get_everyday_data(data):
    data_list = []
    data_size = data.shape[0]
    # smaller data size for testing purpose
    # data_size = int(data.shape[0] * 0.01)
    for i in range(data_size):
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
        curr_day = {'DATE': curr_day_date, 'HIGH': curr_day_high, 'LOW': curr_day_low}
        data_list.append(curr_day)
        print("processed date: {}, high: {}, low: {}".format(curr_day_date, curr_day_high, curr_day_low))

    return data_list


def get_nr4_days(everyday_data, pip):
    buy_sell_prices = []
    # exclude last day
    for i in range(3, len(everyday_data) - 3):
        first_price_range = everyday_data[i - 3]['HIGH'] - everyday_data[i - 3]['LOW']
        second_price_range = everyday_data[i - 2]['HIGH'] - everyday_data[i - 2]['LOW']
        third_price_range = everyday_data[i - 1]['HIGH'] - everyday_data[i - 1]['LOW']
        fourth_price_range = everyday_data[i]['HIGH'] - everyday_data[i]['LOW']
        if min(first_price_range, second_price_range, third_price_range, fourth_price_range) == fourth_price_range \
                and everyday_data[i]['LOW'] >= everyday_data[i - 1]['LOW'] \
                and everyday_data[i]['HIGH'] <= everyday_data[i - 1]['HIGH']:
            transaction_date = [everyday_data[i + 1]['DATE'], everyday_data[i + 2]['DATE'],
                                everyday_data[i + 3]['DATE']]
            buy_price = everyday_data[i]['HIGH'] + everyday_data[i]['HIGH'] * pip
            short_price = everyday_data[i]['LOW'] - everyday_data[i]['LOW'] * pip

            buy_sell_price = {'DATE': transaction_date, 'BUY': buy_price, 'SHORT': short_price}
            buy_sell_prices.append(buy_sell_price)
            print("NR4 day, data: {}, high: {}, low: {}, price range: {}".format(everyday_data[i]['DATE'],
                                                                                 everyday_data[i]['HIGH'],
                                                                                 everyday_data[i]['LOW'],
                                                                                 fourth_price_range))
    return buy_sell_prices


def get_nr7_days(everyday_data, pip):
    buy_sell_prices = []
    # exclude last day
    for i in range(7, len(everyday_data) - 3):

        first_price_range = everyday_data[i - 6]['HIGH'] - everyday_data[i - 6]['LOW']
        second_price_range = everyday_data[i - 5]['HIGH'] - everyday_data[i - 5]['LOW']
        third_price_range = everyday_data[i - 4]['HIGH'] - everyday_data[i - 4]['LOW']
        fourth_price_range = everyday_data[i - 3]['HIGH'] - everyday_data[i - 3]['LOW']
        fifth_price_range = everyday_data[i - 2]['HIGH'] - everyday_data[i - 2]['LOW']
        sixth_price_range = everyday_data[i - 1]['HIGH'] - everyday_data[i - 1]['LOW']
        seventh_price_range = everyday_data[i]['HIGH'] - everyday_data[i]['LOW']
        if min(first_price_range, second_price_range, third_price_range, fourth_price_range, fifth_price_range,
               sixth_price_range, seventh_price_range) == seventh_price_range \
                and everyday_data[i]['LOW'] >= everyday_data[i - 1]['LOW'] \
                and everyday_data[i]['HIGH'] <= everyday_data[i - 1]['HIGH']:
            transaction_date = [everyday_data[i + 1]['DATE'], everyday_data[i + 2]['DATE'],
                                everyday_data[i + 3]['DATE']]
            buy_price = everyday_data[i]['HIGH'] + everyday_data[i]['HIGH'] * pip
            short_price = everyday_data[i]['LOW'] - everyday_data[i]['LOW'] * pip

            buy_sell_price = {'DATE': transaction_date, 'BUY': buy_price, 'SHORT': short_price}
            buy_sell_prices.append(buy_sell_price)
            print("NR7 day, data: {}, high: {}, low: {}, price range: {}".format(everyday_data[i]['DATE'],
                                                                                 everyday_data[i]['HIGH'],
                                                                                 everyday_data[i]['LOW'],
                                                                                 seventh_price_range))
    return buy_sell_prices


def test_result(data, buy_sell_prices, cash, stop_loss_pip, stop_profit_pip):
    start_cash = cash
    print("started testing with {} cash".format(start_cash))
    profit_loss_data_by_day = {'DATE': [], 'PROFIT LOSS PERCENTAGE': [], 'PROFIT LOSS AMOUNT': []}
    profit_loss_data_by_month = {'MONTH': [], 'PROFIT LOSS PERCENTAGE': [], 'PROFIT LOSS AMOUNT': []}

    for buy_sell_price in buy_sell_prices:
        initial_cash = cash
        position = 0
        holdings = 0
        indicator = ''
        buy_price = buy_sell_price['BUY']
        short_price = buy_sell_price['SHORT']
        print("dates: {}".format(buy_sell_price['DATE']))
        for day_number, curr_date in enumerate(buy_sell_price['DATE']):
            curr_day_data = data.loc[data['<DTYYYYMMDD>'] == curr_date]
            entry_number = 0
            print("    date: {}".format(curr_date))

            for i, curr_entry in curr_day_data.iterrows():
                curr_price = curr_entry['<OPEN>']

                # buy sell
                if indicator == '' and curr_price >= buy_price:
                    position = curr_price
                    holdings = int(cash / curr_price)
                    cash = cash - curr_price * holdings
                    indicator = 'b'
                    print("        bought at {}".format(curr_price))
                if indicator == '' and curr_price <= short_price:
                    position = curr_price
                    holdings = int(cash / curr_price)
                    cash = cash + curr_price * holdings
                    indicator = 's'
                    print("        shorted at {}".format(curr_price))

                # holdings in hand
                if position != 0 or holdings != 0:
                    # check loss
                    if indicator == 'b' and curr_price <= position - position * stop_loss_pip:
                        cash = cash + curr_price * holdings
                        position = 0
                        holdings = 0
                        print("        stop loss bought at {}".format(curr_price))

                    elif indicator == 's' and curr_price >= position + position * stop_loss_pip:
                        cash = cash - curr_price * holdings
                        position = 0
                        holdings = 0
                        print("        stop loss shorted at {}".format(curr_price))

                    # check profit
                    elif indicator == 'b' and curr_price >= position + position * stop_profit_pip:
                        cash = cash + curr_price * holdings
                        position = 0
                        holdings = 0
                        print("        stop profit bought at {}".format(curr_price))

                    elif indicator == 's' and curr_price <= position - position * stop_profit_pip:
                        cash = cash - curr_price * holdings
                        position = 0
                        holdings = 0
                        print("        stop profit shorted at {}".format(curr_price))

                    # clean up, if last day last entry
                    elif entry_number == curr_day_data.shape[0] - 1 and day_number == (len(buy_sell_price['DATE']) - 1):
                        if indicator == 'b':
                            cash = cash + curr_price * holdings
                            position = 0
                            holdings = 0
                            print("        cleaned bought at {}".format(curr_price))
                        if indicator == 's':
                            cash = cash - curr_price * holdings
                            position = 0
                            holdings = 0
                            print("        cleaned shorted at {}".format(curr_price))
                entry_number += 1

        profit_loss = cash - initial_cash
        profit_loss_percentage = profit_loss / initial_cash
        profit_loss_data_by_day['DATE'].append(str(curr_date))
        profit_loss_data_by_day['PROFIT LOSS PERCENTAGE'].append(profit_loss_percentage)
        profit_loss_data_by_day['PROFIT LOSS AMOUNT'].append(profit_loss)

        curr_month = (str(curr_date))[0:6]
        # new month
        if curr_month not in profit_loss_data_by_month['MONTH']:
            profit_loss_data_by_month['MONTH'].append(curr_month)
            profit_loss_data_by_month['PROFIT LOSS PERCENTAGE'].append(profit_loss_percentage)
            profit_loss_data_by_month['PROFIT LOSS AMOUNT'].append(profit_loss)
        else:
            if profit_loss_data_by_month['PROFIT LOSS AMOUNT'][-1] == 0:
                profit_loss_data_by_month['PROFIT LOSS PERCENTAGE'][-1] += profit_loss_percentage
            else:
                profit_loss_data_by_month['PROFIT LOSS PERCENTAGE'][-1] \
                    += (profit_loss / profit_loss_data_by_month['PROFIT LOSS AMOUNT'][-1]) \
                       * profit_loss_data_by_month['PROFIT LOSS PERCENTAGE'][-1]
            profit_loss_data_by_month['PROFIT LOSS AMOUNT'][-1] += profit_loss

        print("    cash on open: {}, cash on close: {}".format(initial_cash, cash))
        print("    profit/loss amount: {}, profit/loss rate: {}".format(profit_loss, profit_loss_percentage))

    profit_loss_df_by_day = pd.DataFrame(profit_loss_data_by_day,
                                         columns=['DATE', 'PROFIT LOSS PERCENTAGE', 'PROFIT LOSS AMOUNT'])
    profit_loss_df_by_month = pd.DataFrame(profit_loss_data_by_month,
                                           columns=['MONTH', 'PROFIT LOSS PERCENTAGE', 'PROFIT LOSS AMOUNT'])

    print("trading overview by day:")
    print(profit_loss_df_by_day)
    print("trading overview by month:")
    print(profit_loss_df_by_month)
    print("started testing with {} cash".format(start_cash))
    print("ended testing with {} cash".format(cash))
    print("profit/loss amount: {}, profit/loss rate: {}".format(cash - start_cash, (cash - start_cash) / start_cash))

    return profit_loss_df_by_day, profit_loss_df_by_month


def save_test_result(profit_loss_df_by_day, profit_loss_df_by_month, save_path):
    Path("./output/" + save_path).mkdir(parents=True, exist_ok=True)
    profit_loss_df_by_day.to_csv("./output/" + save_path + "/profit_loss_df_by_day.csv", index=False)
    profit_loss_df_by_month.to_csv("./output/" + save_path + "/profit_loss_df_by_month.csv", index=False)

    if not profit_loss_df_by_day.empty:
        ax = plt.gca()
        profit_loss_df_by_day.plot(kind='line', x='DATE', y='PROFIT LOSS PERCENTAGE', ax=ax)
        profit_loss_df_by_day.plot(kind='line', x='DATE', y='PROFIT LOSS AMOUNT', color='red', ax=ax)
        plt.suptitle("PROFIT LOSS PERCENTAGE AND AMOUNT vs DATE LINE GRAPH")
        plt.savefig("./output/" + save_path + "/PROFIT LOSS PERCENTAGE AND AMOUNT vs DATE LINE GRAPH.png")

        profit_loss_df_by_day.plot(kind='line', x='DATE', y='PROFIT LOSS PERCENTAGE')
        plt.suptitle("PROFIT LOSS PERCENTAGE vs DATE LINE GRAPH")
        plt.savefig("./output/" + save_path + "/PROFIT LOSS PERCENTAGE vs DATE LINE GRAPH.png")

        profit_loss_df_by_day.plot(kind='line', x='DATE', y='PROFIT LOSS AMOUNT')
        plt.suptitle("PROFIT LOSS AMOUNT vs DATE LINE GRAPH")
        plt.savefig("./output/" + save_path + "/PROFIT LOSS AMOUNT vs DATE LINE GRAPH.png")

        profit_loss_df_by_day.plot(kind='bar', x='DATE', y='PROFIT LOSS PERCENTAGE')
        plt.suptitle("PROFIT LOSS PERCENTAGE vs DATE BAR GRAPH")
        plt.savefig("./output/" + save_path + "/PROFIT LOSS PERCENTAGE vs DATE BAR GRAPH.png")

        profit_loss_df_by_day.plot(kind='bar', x='DATE', y='PROFIT LOSS AMOUNT')
        plt.suptitle("PROFIT LOSS AMOUNT vs DATE BAR GRAPH")
        plt.savefig("./output/" + save_path + "/PROFIT LOSS AMOUNT vs DATE BAR GRAPH.png")
        plt.show()


def start_simulate(option):
    data_file_path = option["data_file_path"]
    cash = option["cash"]
    pip = option["pip"]
    stop_loss_pip = option["stop_loss_pip"] * pip
    stop_profit_pip = option["stop_profit_pip"] * pip

    data = load_data(data_file_path)
    everyday_data = get_everyday_data(data)
    buy_sell_prices = []
    if option["type"] == "nr4":
        buy_sell_prices = get_nr4_days(everyday_data, pip)
    elif option["type"] == "nr7":
        buy_sell_prices = get_nr7_days(everyday_data, pip)
    profit_loss_df_by_day, profit_loss_df_by_month = test_result(data, buy_sell_prices, cash, stop_loss_pip,
                                                                 stop_profit_pip)
    save_path = (data_file_path.split("/"))[-1] + "-" + option["type"] + "-" + str(pip) + "-" + str(
        stop_loss_pip) + "-" + str(stop_profit_pip)
    save_test_result(profit_loss_df_by_day, profit_loss_df_by_month, save_path)


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)  # or 1000
    pd.set_option('display.max_rows', None)

    options = [
        {"data_file_path": "./data/USD/AUDCNH2015-2020.csv", "cash": 10000, "pip": 0.0001,
         "stop_loss_pip": 100, "stop_profit_pip": 400, "type": "nr7"},
        {"data_file_path": "./data/USD/AUDCNH2015-2020.csv", "cash": 10000, "pip": 0.0001,
         "stop_loss_pip": 100, "stop_profit_pip": 400, "type": "nr4"}
    ]
    threads = []
    for option in options:
        if len(option.keys()) == 6:
            thread = threading.Thread(target=start_simulate, args=(option,))
            threads.append(thread)
            thread.start()

    for thread in threads:
        thread.join()
