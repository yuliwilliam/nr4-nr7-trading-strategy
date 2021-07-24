# NR4-NR7-Trading-Strategy
A python project that implements the NR4 and NR7 trading strategy, and tests the strategy based on the past foreign exchange rates between major currencies.

## Sample Input
A snippet of AUDCNH2015-2020.csv which contains the foreign exchange rates between AUD and CHN.
|\<TICKER\>|\<DTYYYYMMDD\>|\<TIME\>|\<OPEN\>|\<HIGH\>|\<LOW\>|\<CLOSE\>|
|----------|--------------|--------|--------|--------|-------|---------|
|0         |20150101      |22:00:00|5.08729 |5.08736 |5.08623|5.0863   |
|1         |20150101      |22:01:00|5.08629 |5.08671 |5.08502|5.08556  |
|2         |20150101      |22:02:00|5.08597 |5.08597 |5.08597|5.08597  |
|3         |20150101      |22:03:00|5.08517 |5.08597 |5.08516|5.08584  |
|4         |20150101      |22:04:00|5.08581 |5.08581 |5.08544|5.08544  |
|5         |20150101      |22:05:00|5.08538 |5.08538 |5.08511|5.08511  |


## Sample Outputs
### Example 1: AUD - CHN with NR4
Source Currencies: AUD - CHN  
Data Range: 20150101 - 20171218  
Trading strategy: NR4  
Initial Cash: 10000  

![](./output/AUDCNH2015-2020.csv-nr4-0.0001-0.01-0.04/PROFIT%20LOSS%20AMOUNT%20vs%20DATE%20LINE%20GRAPH.png)
![](./output/AUDCNH2015-2020.csv-nr4-0.0001-0.01-0.04/PROFIT%20LOSS%20AMOUNT%20vs%20DATE%20BAR%20GRAPH.png)
![](./output/AUDCNH2015-2020.csv-nr4-0.0001-0.01-0.04/PROFIT%20LOSS%20PERCENTAGE%20vs%20DATE%20LINE%20GRAPH.png)
![](./output/AUDCNH2015-2020.csv-nr4-0.0001-0.01-0.04/PROFIT%20LOSS%20PERCENTAGE%20vs%20DATE%20BAR%20GRAPH.png)

### Example 2: AUD - CHN with NR7
Source Currencies: AUD - CHN  
Data Range: 20150101 - 20171218  
Trading strategy: NR7  
Initial Cash: 10000  

![](./output/AUDCNH2015-2020.csv-nr7-0.0001-0.01-0.04/PROFIT%20LOSS%20AMOUNT%20vs%20DATE%20LINE%20GRAPH.png)
![](./output/AUDCNH2015-2020.csv-nr7-0.0001-0.01-0.04/PROFIT%20LOSS%20AMOUNT%20vs%20DATE%20BAR%20GRAPH.png)
![](./output/AUDCNH2015-2020.csv-nr7-0.0001-0.01-0.04/PROFIT%20LOSS%20PERCENTAGE%20vs%20DATE%20LINE%20GRAPH.png)
![](./output/AUDCNH2015-2020.csv-nr7-0.0001-0.01-0.04/PROFIT%20LOSS%20PERCENTAGE%20vs%20DATE%20BAR%20GRAPH.png)



