# 2021 Spring CS474 Term Project

## Run main function
```
./run.sh
```

## Dataset information
- Targeted data : The Korea Herald, National Section news
- Period : 2015 - 2017
- Crawled Date : 2018-10-26
- Data Header :  [title, author, time, description, body, section]
- Data Size : 23769 news
- Data Format : json

## Description

### Issue Trend Analysis
![IssueTrendAnalysis](unnamed.jpg)
This task is to find the top ten most significant issues for each year and rank them, from the news articles over the period of three years. It can be seen as a task for Topic Modeling, and LDA is known as the most common algorithm in Topic Modeling. Therefore, we intend to apply LDA as well as other algorithms to compare performance and use it to create the most optimal scoring function or criteria. 

### On-Issue Event Tracking
We extract the result which should describe a sequence of the events specifically tied to the issue on a temporal line. We divided this task into three small steps and devised specific methods for each step in detail.
1. Choose Two Issues
2. Identify the Events related to the Issues.
3. Extract the Events related to the Issues
4. Extract Detailed Facts for each of the Two Events and Relevant Events

On identifying the events related to the issues, we use our novel approach of consecutive Information Retrieval, which is to use keywords from the current article as an additional query to identify next event. 

### Related-Issue Event Tracking
We extract and describe related events which are not directly tied to the particular issue. We apply a method which is composed of 3 steps.
1. Filter related articles by query
2. Identify events related to the issue
3. Get details of the extracted events.

## Contributors
- Minseon Hwang
- Changhun Kim
- Sookyung Han