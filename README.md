# Web-service for predicting apartment prices
Flask web-service for predicting prices for renting an apartment in Saint-Petersburg.
## Source data and some statistics
Service uses database from [Yandex.Realty classified](https://realty.yandex.ru) which contains real estate listings for apartments in St. Petersburg and Leningrad Oblast from 2016 till the middle of August 2018. 

The model uses only listings that are available for rent and located in St.Petersburg.

Heatmap shows correlation between last price and other features:

![Heatmap](https://github.com/irinapendryak/final_predictor/blob/main/pictures/heatmap.png) 

Relations between last price and other features showed by scatter plots and box plots:

![Open_plan](https://github.com/irinapendryak/final_predictor/blob/main/pictures/open_plan.png) 
![Studio](https://github.com/irinapendryak/final_predictor/blob/main/pictures/studio.png)

![Area](https://github.com/irinapendryak/final_predictor/blob/main/pictures/area.png) 
![Rooms](https://github.com/irinapendryak/final_predictor/blob/main/pictures/rooms.png) 
![Kitchen_area](https://github.com/irinapendryak/final_predictor/blob/main/pictures/kitchen_area.png) 
![Living_area](https://github.com/irinapendryak/final_predictor/blob/main/pictures/living_area.png) 
![Floor](https://github.com/irinapendryak/final_predictor/blob/main/pictures/floor.png) 
![Renovation](https://github.com/irinapendryak/final_predictor/blob/main/pictures/renovation.png) 

## Model, choosen framework, hyperparams
Both created machine learning models are **CatBoost models** but with different features used.

Feautures for the models:
| Model 1      | Model 2       |
| :----------- | :-----------: |
| open_plan    | open_plan     |
| rooms        | rooms         |
| area         | area          |
| renovation   | renovation    |
|              | floor         |
|              | studio        |
|              | kitchen_area  |
|              | living_area   |

Hyperparameteres were chosen based on **GridSearch** with 3 folds.

Hyperparameters used in **the first model**:
* Depth: 7
* Iterations: 1000
* Learning rate: 0.01

Hyperparameters used in **the second model**:
* Depth: 9
* Iterations: 1000
* Learning rate: 0.01

Test results:
|              | Model 1      | Model 2      |
| :----------- | :----------- | :----------- |
| MAE          | 0.345        | 0.330        |
| MSE          | 0.369        | 0.353        |
| RMSE         | 0.607        | 0.594        |



## How to install instructions and run your app with virtual environment
1. Connect to your virtual machine.
2. Create a virtual ebvironment with these commands:
```bash
sudo apt install python3.8-venv
python3 -m venv env
```
3. Activate your virtual environemnt:
```bash
source env/bin/activate
```
4. Install required packages:
```bash
pip install -r requirements.txt
```
5. Run the code:
```bash
python app.py
```
## Information about Dockerfile
Content of the Dokcerfile:
```Dockerfile
from ubuntu:20.04
MAINTAINER Irina Pendriak
RUN apt-get update -y
COPY . /opt/final_predictor
WORKDIR /opt/final_predictor
RUN apt install -y python3-pip
RUN pip3 install -r requirements.txt
CMD python3 app.py
```
The solution is based on `ubuntu` image. We start the container, copy the source code to the docker image, change working directory, install `pip3`, install required packages from `requirements.txt` and run the app.
## How to open the port in remote VM
1. Connect to your remote virtual machine.
2. Run the following command
```bash
sudo ufw allow 5444
```
## How to run app using docker and which port it uses
1. Pull the docker image:
```bash
docker pull irinapendriak/final_predictor:v.0.4
```
2. Run the docker image:
```bash
docker run --network host -d irinapendriak/final_predictor:v.0.4
```

It uses port **5444**.

Link to my docker: https://hub.docker.com/repository/docker/irinapendriak/final_predictor
