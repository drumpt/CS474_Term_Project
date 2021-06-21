# 2021 Spring CS474 Term Project
Term project for 2021 Spring CS474 Text Mining course in [KAIST](https://kaist.ac.kr)

## 🏠 [Github page](https://github.com/comafj/CS489-Team-14-repository)

## :pushpin: Motivation and Objectives
Text mining is an area where various analytical methods and approaches are being developed with the development of big data. Because most of the data around us are in text form, collecting and analyzing appropriate text is a very important area of data science.  Among them, news data is generated periodically and deals with various categories, making it a very useful form of data for text mining. Therefore, we would like to analyze Korean Herald news articles written from 2015 to 2017 to find the top 10 Issue trends for each year and analyze more detailed events. We will largely divide this into Issue Trend Analysis and Issue Tracking parts and explain each task as follows.


## :newspaper: Dataset information
- Targeted data : The Korea Herald, National Section news
- Period : 2015 - 2017
- Crawled Date : 2018-10-26
- Data Header :  [title, author, time, description, body, section]
- Data Size : 23769 news
- Data Format : json

## :scroll: Description

### Issue Trend Analysis
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

## :pencil2: Pre-learning
### LDA to find Issues
```
python3 src/lda_test.py
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
        "num_trends": 100,
	    "num_passes" : 30
    },
    "issue_file_list": ["output/2015_Issues.txt", "output/2016_Issues.txt", "output/2017_Issues.txt"],
    "doc2vec": {
        "part_weight": {
            "title": 0,
            "body": 1,
            "section": 0
        },
        "epoch": 30,
        "embedding_dim": 256,
}
```
#### Possible parameters

Parameter | value
--- | ---
data_dir | "dataset", "sample_dataset"
lda | "num_trends", "num_passes"
doc2vec/partial_weight | {"title": x, "body": y, "section": z}, where x + y + z = 1 and x, y, z >= 0
clustering/method | "hierarchical", "DBSCAN", "OPTICS"
on_issue_event_tracking/method | "normal", "consecutive"
detailed_info_extractor/summary_target | "title", "description", "body"


We manage the variables used in the overall process through [config.json](https://github.com/drumpt/CS474_Term_Project/blob/main/config.json). Various settings can be applied by changing the value of each variable in this file.

### 3. Run main function
```
./run.sh
```

## :clipboard: Build and run on docker
### 1. Install docker and download repository
First, install docker on your computer and clone this repository. Below is the comand for cloning current version of the file on github.
```
git clone https://github.com/drumpt/CS474_Term_Project
```

### 2. Build docker image
Change directory to downloaded repository and build docker image
```
cd "path_to_CS474_Term_Project"
docker build . -t "cs474_term_project"
```

### 3. Run docker image
First identify "IMAGE_ID" for built docker image by typing "docker images" and type the command below.
```
docker run -it "IMAGE_ID" ./run_on_docker.sh
```

## :bulb: Show on-issue event tracking evaluation result
This includes the number of relevant documents for each issue and clustering evaluation result.
```
python3 src/on_issue evaluation.py
```

## :chart_with_upwards_trend: Paperwork
For more detailed implementation methods and results, you can refer to the report at the following link: [Report](https://docs.google.com/document/d/1oLPT07ocqV7-SmED2deSL15W9U-yrZMRxlp2oX1V8R0/edit)

## 👤 Author
* Github: Changhun Kim [@drumpt](https://github.com/drumpt)
* Github: Sookyung Han [@suplookie](https://github.com/suplookie)
* Github: Minseon Hwang [@comafj](https://github.com/comafj)
