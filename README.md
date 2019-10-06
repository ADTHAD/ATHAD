<p align="center">
	<a href="https://circleci.com/gh/badges/shields/tree/master">
        <img src="https://img.shields.io/circleci/project/github/badges/shields/master" alt="build status"></a>
    <a href="https://github.com/badges/shields/graphs/contributors" alt="Contributors 3">
        <img src="https://img.shields.io/github/contributors/badges/shields" /></a>
</p>

### Stable release version: ![version](https://img.shields.io/badge/version-1.2-blue)

# ATHAD
Advanced Tools For Healthcare and Diagnostics.
Picture this-- your patient is unwell and he/she needs to get a  battery of tests done. He/She spends the next few days going to different labs and then few more days waiting for the results. Once again he/she is back to you for a diagnosis. Now, here is a tool that lets you predict all the diseases and gives the results instantly.It is a tool developed for the convenience of clinicians for detecting various types of diseases especially subclinical illness.This web application deals with the prediction of potential sickness using machine learning techniques.The dataset used for this procedure are experimental datas.

## Demo App
Link: https://athad.herokuapp.com/ 

## Run locally

1. To start this up, clone the repository by the command given below :

```
$ git clone https://github.com/ADTHAD/ATHAD.git
```
2. Next,move to ATHAD directory:
```
cd ATHAD/
```
3. Make a virtual environment:
```
 $ virtualenv -p python3 athad-app
```
4. Activate environment:

```
$ source athad-app/bin/activate
```
5. Install all of the dependencies

```
$  pip install -r requirements.txt
```
6. Run the app

```
$  python3 app.py
```
