# 2021 Spring CS474 Term Project
Group project (Team 1) for 2021 Spring CS474 TextMining course

## 🏠 [Github page](https://github.com/comafj/CS489-Team-14-repository)

## :pushpin: Motivation and Objectives
Text mining is an area where various analytical methods and approaches are being developed with the development of big data. Because most of the data around us are in text form, collecting and analyzing appropriate text is a very important area of data science.  Among them, news data is generated periodically and deals with various categories, making it a very useful form of data for text mining. Therefore, we would like to analyze Korean Herald news articles written from 2015 to 2017 to find the top 10 Issue trends for each year and analyze more detailed events. We will largely divide this into Issue Trend Analysis and Issue Tracking parts and explain each task as follows.


## :newspaper: Dataset information
- Targeted data : The Korea Herald, National Section news
- Period : 2015 - 2017
- Crawled Date : 2018-10-26
- Data Header :  [title, author, time, description, body, section]
- Data Size : 23769 news
- Data Format : Json
- Repository : [CS474_Term_Project/dataset](https://github.com/drumpt/CS474_Term_Project/tree/main/dataset)

## :pencil2: Pre-learning
### LDA to find Issues
```
python3 ./src/lda_test.py
```
This command divides dataset into articles from 2015, 2016, and 2017 and identifies the Issues for each via LDA and stores them in a txt file. Created txt files are stored in the [output repository](https://github.com/drumpt/CS474_Term_Project/tree/main/output).

## :computer: Usage
### 1. Download repository
```
git clone https://github.com/drumpt/CS474_Term_Project
```
This command will install the uploaded version of the file on github.

### 2. Configure parameters
```
{
    "data_dir": "dataset",
    "lda": {
        "num_trends": 10
    },
    "doc2vec": {
        "part_weight": {
            "title": 0,
            "body": 1,
            "section": 0
        },
        "epoch": 50,
        "embedding_dim": 256,
        ....
}
```
We manage the variables used in the overall process through [config.json](https://github.com/drumpt/CS474_Term_Project/blob/main/config.json). Various settings can be applied by changing the value of each variable in this file.

### 3. Run main function
```
./run.sh
```
This command will automatically install the required libraries recorded in [requirments.txt](https://github.com/drumpt/CS474_Term_Project/blob/main/requirements.txt) and then implement the main function.

## :chart_with_upwards_trend: Paperwork
For more detailed implementation methods and results, you can refer to the report at the following link: [Report](https://docs.google.com/document/d/1oLPT07ocqV7-SmED2deSL15W9U-yrZMRxlp2oX1V8R0/edit)

## 👤 Author
* Github: Changhun Kim [@drumpt](https://github.com/drumpt)
* Github: Sookyung Han [@suplookie](https://github.com/suplookie)
* Github: Minseon Hwang [@comafj](https://github.com/comafj)
