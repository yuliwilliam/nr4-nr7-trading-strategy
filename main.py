import time
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing
from pathlib import Path


def load_data(data_file_path):
    data = pd.read_csv(data_file_path,
                       usecols=['<DTYYYYMMDD>', '<TIME>', '<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>'])
    return data


def get_data_by_date(data):
    data_list = []
    data_size = data.shape[0]
    # smaller data size for testing purpose
    # data_size = int(data.shape[0] * 0.05)
    for i in range(data_size):
        curr_day_date = data['<DTYYYYMMDD>'][i]
        curr_day_high = data['<HIGH>'][i]
        curr_day_low = data['<LOW>'][i]
        if len(data_list) > 0 and curr_day_date == data_list[-1]['DATE']:
            data_list[-1]['HIGH'] = max(data_list[-1]['HIGH'], curr_day_high)
            data_list[-1]['LOW'] = min(data_list[-1]['LOW'], curr_day_low)
            data_list[-1]['RANGE'] = data_list[-1]['HIGH'] - data_list[-1]['LOW']
        else:
            curr_day = {'DATE': curr_day_date, 'HIGH': curr_day_high, 'LOW': curr_day_low,
                        'RANGE': curr_day_high - curr_day_low}
            data_list.append(curr_day)
            print("processed date: {}, high: {}, low: {}".format(curr_day_date, curr_day_high, curr_day_low))

    return data_list


def get_nr4_days(everyday_data, pip):
    buy_sell_prices = []
    # exclude last day
    for i in range(3, len(everyday_data) - 3):
        if min(everyday_data[i - 3]['RANGE'], everyday_data[i - 2]['RANGE'], everyday_data[i - 1]['RANGE'],
               everyday_data[i]['RANGE']) == everyday_data[i]['RANGE'] \
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
                                                                                 everyday_data[i]['RANGE']))
    return buy_sell_prices


def get_nr7_days(everyday_data, pip):
    buy_sell_prices = []
    # exclude last day
    for i in range(7, len(everyday_data) - 3):
        if min(everyday_data[i - 6]['RANGE'], everyday_data[i - 5]['RANGE'], everyday_data[i - 4]['RANGE'],
               everyday_data[i - 3]['RANGE'], everyday_data[i - 2]['RANGE'],
               everyday_data[i - 1]['RANGE'], everyday_data[i]['RANGE']) == everyday_data[i]['RANGE'] \
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
                                                                                 everyday_data[i]['RANGE']))
    return buy_sell_prices


def open_position(indicator, cash, curr_price, holdings):
    # indicator = 1 - buy, indicator = -1 - sell
    position = curr_price
    holdings = indicator * int(cash / curr_price)
    cash = cash - curr_price * holdings
    return cash, holdings, position


def close_position(cash, curr_price, holdings):
    cash = cash + curr_price * holdings
    position = 0
    holdings = 0
    return cash, holdings, position


def trade(cash, holdings, position, curr_price, buy_price, short_price, stop_loss_pip, stop_profit_pip,
          is_last_trading_period):
    # buy sell when no holdings
    if position == 0 and holdings == 0:
        if curr_price >= buy_price:
            cash, holdings, position = open_position(1, cash, curr_price, holdings)
            print("        bought at {}".format(curr_price))
        if curr_price <= short_price:
            cash, holdings, position = open_position(-1, cash, curr_price, holdings)
            print("        short at {}".format(curr_price))

    # holdings in hand
    if position != 0 or holdings != 0:
        # check loss
        if holdings > 0 and curr_price <= position - position * stop_loss_pip:
            cash, holdings, position = close_position(cash, curr_price, holdings)
            print("        stop loss bought at {}".format(curr_price))

        elif holdings < 0 and curr_price >= position + position * stop_loss_pip:
            cash, holdings, position = close_position(cash, curr_price, holdings)
            print("        stop loss shorted at {}".format(curr_price))

        # check profit
        elif holdings > 0 and curr_price >= position + position * stop_profit_pip:
            cash, holdings, position = close_position(cash, curr_price, holdings)
            print("        stop profit bought at {}".format(curr_price))

        elif holdings < 0 and curr_price <= position - position * stop_profit_pip:
            cash, holdings, position = close_position(cash, curr_price, holdings)
            print("        stop profit shorted at {}".format(curr_price))

        # clean up, if last day last entry
        elif is_last_trading_period:
            if holdings > 0:
                print("        cleaned bought at {}".format(curr_price))
            if holdings < 0:
                print("        cleaned shorted at {}".format(curr_price))
            cash, holdings, position = close_position(cash, curr_price, holdings)
    return cash, holdings, position


def backtest(data, buy_sell_prices, cash, stop_loss_pip, stop_profit_pip):
    start_cash = cash
    print("started testing with {} cash".format(start_cash))
    profit_loss_data_by_day = {'DATE': [], 'PROFIT LOSS PERCENTAGE': [], 'PROFIT LOSS AMOUNT': []}
    profit_loss_data_by_month = {'MONTH': [], 'PROFIT LOSS PERCENTAGE': [], 'PROFIT LOSS AMOUNT': []}

    for buy_sell_price in buy_sell_prices:
        curr_period_initial_cash, holdings, position, = cash, 0, 0
        buy_price, short_price = buy_sell_price['BUY'], buy_sell_price['SHORT']
        print("dates: {}".format(buy_sell_price['DATE']))
        for day_number, curr_date in enumerate(buy_sell_price['DATE']):
            curr_day_data = data.loc[data['<DTYYYYMMDD>'] == curr_date]
            print("    date: {}".format(curr_date))
            for i in range(curr_day_data.shape[0]):
                curr_price = curr_day_data['<CLOSE>'].iloc[i]
                # clean up, if last day last entry
                is_last_trading_period = i == curr_day_data.shape[0] - 1 and day_number == len(
                    buy_sell_price['DATE']) - 1
                cash, holdings, position = trade(cash, holdings, position, curr_price, buy_price, short_price,
                                                 stop_loss_pip, stop_profit_pip, is_last_trading_period)

        profit_loss = cash - curr_period_initial_cash
        profit_loss_percentage = profit_loss / curr_period_initial_cash
        profit_loss_data_by_day['DATE'].append(str(buy_sell_price['DATE'][0]))
        profit_loss_data_by_day['PROFIT LOSS PERCENTAGE'].append(profit_loss_percentage)
        profit_loss_data_by_day['PROFIT LOSS AMOUNT'].append(profit_loss)

        curr_month = str(buy_sell_price['DATE'][0])[0:6]
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

        print("    cash on open: {}, cash on close: {}".format(curr_period_initial_cash, cash))
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
        # plt.show()


def start_simulation(option):
    start = time.time()

    data_file_path = option["data_file_path"]
    cash = option["cash"]
    pip = option["pip"]
    stop_loss_pip = option["stop_loss_pip"] * pip
    stop_profit_pip = option["stop_profit_pip"] * pip

    data = load_data(data_file_path)
    everyday_data = get_data_by_date(data)
    buy_sell_prices = []
    if option["type"] == "nr4":
        buy_sell_prices = get_nr4_days(everyday_data, pip)
    elif option["type"] == "nr7":
        buy_sell_prices = get_nr7_days(everyday_data, pip)
    profit_loss_df_by_day, profit_loss_df_by_month = backtest(data, buy_sell_prices, cash, stop_loss_pip,
                                                              stop_profit_pip)
    save_path = (data_file_path.split("/"))[-1] + "-" + option["type"] + "-" + str(pip) + "-" + str(
        stop_loss_pip) + "-" + str(stop_profit_pip)
    save_test_result(profit_loss_df_by_day, profit_loss_df_by_month, save_path)
    end = time.time()
    print('time elapsed: {} seconds'.format(end - start))


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)  # or 1000
    pd.set_option('display.max_rows', None)

    options = [
        {"data_file_path": "./data/USD/AUDCNH2015-2020.csv", "cash": 10000, "pip": 0.0001,
         "stop_loss_pip": 100, "stop_profit_pip": 400, "type": "nr7"},
        {"data_file_path": "./data/USD/AUDCNH2015-2020.csv", "cash": 10000, "pip": 0.0001,
         "stop_loss_pip": 100, "stop_profit_pip": 400, "type": "nr4"}
    ]

    # use process instead of thread since matplotlib is not thread safe
    processes = []
    for option in options:
        if len(option.keys()) == 6:
            process = multiprocessing.Process(target=start_simulation, args=(option,))
            processes.append(process)
            process.start()

    for process in processes:
        process.join()
