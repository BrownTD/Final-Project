# Final-Project "Housing Prices Prediction - Investment Decision"

![image](https://highworthcitizen.com/wp-content/uploads/2019/12/real-estate-market.jpg)

Team Members:

Nevyn Brown

Priya Sivaraj

Ros Tiamzon

==========================================================================

## PROJECT GOAL

“Our aim is to create a predictive analysis of house prices so we can advise potential buyers or investors on sales prices and where to buy (neighborhood) based on the home features that they value most.”


==========================================================================

## MODEL, TECHNOLOGIES, LIBRARIES

numpy: provides comprehensive mathematical functions, used for working with arrays. 

pandas:  provides a plethora of useful functions that make it easy to express, analyze, and manipulate data.

matplotlib:  a comprehensive library for creating static, animated, and interactive visualizations in python.

hvplot:  allows for users to easily generate a wide array of plot types and interactive visualizations 

seaborn: a library mostly used for statistical plotting in Python, built on top of matplotlib and provides beautiful default styles and color palettes to make statistical plots more attractive.

geoviews:  library that makes it easy to explore and visualize geographical, meteorological, and oceanographic datasets

sklearn:  a machine learning library for the python programming language that allows for the use of multiple machine learning models, tools, and algorithms.

cartopy: allows for georeferencing matplotlib axes objects. Cartopy’s crs class supports a variety of map projections.

xgboost:  is an implementation of gradient-boosting decision trees. It has been used by data scientists and researchers worldwide to optimize their machine-learning models.

pickle:   used in serializing and deserializing a Python object structure. 

# Installation Guide

Before running the application first install the following dependencies.
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

==========================================================================

## ABOUT THE PROJECT
Iowa House Prices dataset / prediction

Relevant House Features Used:

    SalePrice - the property's sale price in dollars. This is the target variable to be predicted.
    MSSubClass: The building class
    MSZoning: The general zoning classification:
                C for Commercial
                FV for Floating Village
                RH for Residential High Density
                RL for Residential Low Density 
                RM for Residential Medium Density 
    LotFrontage: Linear feet of street connected to property
    LotArea: Lot size in square feet
    Street: Type of road access
    Alley: Type of alley access
    LotShape: General shape of property
    LandContour: Flatness of the property
    Utilities: Type of utilities available
    LotConfig: Lot configuration
    LandSlope: Slope of property
    Neighborhood: Physical locations within Ames city limits
    Condition1: Proximity to main road or railroad
    Condition2: Proximity to main road or railroad (if a second is present)
    BldgType: Type of dwelling
    HouseStyle: Style of dwelling
    OverallQual: Overall material and finish quality
    OverallCond: Overall condition rating
    YearBuilt: Original construction date
    YearRemodAdd: Remodel date
    RoofStyle: Type of roof
    RoofMatl: Roof material
    Exterior1st: Exterior covering on house
    Exterior2nd: Exterior covering on house (if more than one material)
    MasVnrType: Masonry veneer type
    MasVnrArea: Masonry veneer area in square feet
    ExterQual: Exterior material quality
    ExterCond: Present condition of the material on the exterior
    Foundation: Type of foundation
    BsmtQual: Height of the basement
    BsmtCond: General condition of the basement
    BsmtExposure: Walkout or garden level basement walls
    BsmtFinType1: Quality of basement finished area
    BsmtFinSF1: Type 1 finished square feet
    BsmtFinType2: Quality of second finished area (if present)
    BsmtFinSF2: Type 2 finished square feet
    BsmtUnfSF: Unfinished square feet of basement area
    TotalBsmtSF: Total square feet of basement area
    Heating: Type of heating
    HeatingQC: Heating quality and condition
    CentralAir: Central air conditioning
    Electrical: Electrical system
    1stFlrSF: First Floor square feet
    2ndFlrSF: Second floor square feet
    LowQualFinSF: Low quality finished square feet (all floors)
    GrLivArea: Above grade (ground) living area square feet
    BsmtFullBath: Basement full bathrooms
    BsmtHalfBath: Basement half bathrooms
    FullBath: Full bathrooms above grade
    HalfBath: Half baths above grade
    Bedroom: Number of bedrooms above basement level
    Kitchen: Number of kitchens
    KitchenQual: Kitchen quality
    TotRmsAbvGrd: Total rooms above grade (does not include bathrooms)
    Functional: Home functionality rating
    Fireplaces: Number of fireplaces
    FireplaceQu: Fireplace quality
    GarageType: Garage location
    GarageYrBlt: Year garage was built
    GarageFinish: Interior finish of the garage
    GarageCars: Size of garage in car capacity
    GarageArea: Size of garage in square feet
    GarageQual: Garage quality
    GarageCond: Garage condition
    PavedDrive: Paved driveway
    WoodDeckSF: Wood deck area in square feet
    OpenPorchSF: Open porch area in square feet
    EnclosedPorch: Enclosed porch area in square feet
    3SsnPorch: Three season porch area in square feet
    ScreenPorch: Screen porch area in square feet
    PoolArea: Pool area in square feet
    PoolQC: Pool quality
    Fence: Fence quality
    MiscFeature: Miscellaneous feature not covered in other categories
    MiscVal: $Value of miscellaneous feature
    MoSold: Month Sold
    YrSold: Year Sold
    SaleType: Type of sale
    SaleCondition: Condition of sale

==========================================================================

## DATA PROCESSING STEPS

1. DATA COLLECTION: Collected public information on housing data 
2. DATA EXPLORATION: Analyzed data, house features, correlation of house features to sales prices, analyzed numerical variables (discrete vs continuous), analyzed categorical features, distribution of sales prices
3. DATA CLEANING: Analyzed/cleansed missing data, analyzed outliers
4. HOME FEATURES ENGINEERING: Analyzed rare categorical features (less than 1% removed since not adding value to sales price)
5. DATA TRANSFORMATION: Converted categorical variables to one hot encoding, removed duplicates, prepared data for feeding into the model

==========================================================================

## RESULTS / GRAPHS

Sales price correlation with each feature - summary

![image](https://user-images.githubusercontent.com/108433370/206251684-030933c8-761b-4043-8b33-c3cde5014c63.png)

Distribution of sales price

![image](https://user-images.githubusercontent.com/108433370/206251801-3ad8e660-9dda-4d81-a4a4-539902f663dc.png)

A. discrete variables - correlation to sales price

![image](https://user-images.githubusercontent.com/108433370/206247953-83955e6c-7231-42cf-82f2-a5554023df9c.png)

![image](https://user-images.githubusercontent.com/108433370/206247992-57beafef-23e7-4dda-8950-62e1f1881d12.png)

![image](https://user-images.githubusercontent.com/108433370/206248089-a6808dc9-1c1e-41c4-b8de-69d34936e68e.png)

![image](https://user-images.githubusercontent.com/108433370/206248234-9ae5f706-83f3-4e92-9663-685e9cc3639a.png)

![image](https://user-images.githubusercontent.com/108433370/206248320-26a87024-9fb7-401e-ae65-8eae31b496dd.png)

![image](https://user-images.githubusercontent.com/108433370/206248369-20bc8cf5-a37a-4d68-b58d-6c531220d96a.png)

![image](https://user-images.githubusercontent.com/108433370/206248427-5b32f767-d2af-4ddb-9a89-3d99a553c7d5.png)

![image](https://user-images.githubusercontent.com/108433370/206248503-c15ccb86-07e7-4613-8590-b53a3a17be06.png)


B. Continous numerical variables - correlation to sales price

![image](https://user-images.githubusercontent.com/108433370/206248707-57916b77-775c-4ee0-b941-a0eec3c65871.png)

![image](https://user-images.githubusercontent.com/108433370/206248806-3d07f8d7-ecfa-4e7f-9094-17eb4a4034eb.png)

![image](https://user-images.githubusercontent.com/108433370/206248879-5fcc5541-6d4c-4eb3-8e5f-6666387fa777.png)

![image](https://user-images.githubusercontent.com/108433370/206248935-e9def54b-056a-4390-9a34-94b977b1cd1d.png)

![image](https://user-images.githubusercontent.com/108433370/206249013-54465165-cc7e-48a1-8ee6-64bc176eb788.png)


C. Categorical variables - correlation to sales price

![image](https://user-images.githubusercontent.com/108433370/206249215-179bc287-66b6-4ab7-9871-bba77b2d9a2d.png)

![image](https://user-images.githubusercontent.com/108433370/206249312-c63f1bae-f19f-4fd8-8baf-2e3a27988ed3.png)

![image](https://user-images.githubusercontent.com/108433370/206249406-b69827bf-6aad-48df-b975-5a44ff1ce54c.png)

![image](https://user-images.githubusercontent.com/108433370/206249522-d882d85f-390a-44a2-854f-4050b34208c8.png)

![image](https://user-images.githubusercontent.com/108433370/206249636-907605d4-5173-4d06-b847-0956c691276b.png)

![image](https://user-images.githubusercontent.com/108433370/206249758-3891c625-1d09-4729-a5fc-520a00bd4e2e.png)

![image](https://user-images.githubusercontent.com/108433370/206249843-20592acd-26b5-4df1-be25-1875a20865dc.png)

![image](https://user-images.githubusercontent.com/108433370/206249941-0136bf27-572b-4f19-bf23-5c7a0f295000.png)

![image](https://user-images.githubusercontent.com/108433370/206250100-d44641a3-dc57-4062-99b9-464b9c4a3aa0.png)

![image](https://user-images.githubusercontent.com/108433370/206250201-9e0eb7b2-f31b-4857-b995-e0f17444002f.png)

![image](https://user-images.githubusercontent.com/108433370/206250330-c5692f7d-5f33-498a-952c-8c66d73cbe7b.png)

![image](https://user-images.githubusercontent.com/108433370/206250444-6b185910-50eb-4ead-805e-f8e7208df4e7.png)

![image](https://user-images.githubusercontent.com/108433370/206250571-64de79ff-c8bd-4f41-878b-cac6f800e2f3.png)

![image](https://user-images.githubusercontent.com/108433370/206250674-dc10104d-f0d8-4b09-88cb-52e6cd9a7031.png)

![image](https://user-images.githubusercontent.com/108433370/206250774-eb87ab2e-6b70-4442-b21d-aa63c91f143f.png)

![image](https://user-images.githubusercontent.com/108433370/206250878-bd5f55cb-ed80-4d5b-9459-f8e577558ab5.png)

![image](https://user-images.githubusercontent.com/108433370/206250988-71d75b3f-f8ce-4342-87b0-e04c6031ea7f.png)

![image](https://user-images.githubusercontent.com/108433370/206251076-9d0e798b-951a-4a90-bc64-2eb921849d4f.png)

![image](https://user-images.githubusercontent.com/108433370/206251176-75a14516-3c49-4fde-99fd-dbfdf2e4f2c2.png)

![image](https://user-images.githubusercontent.com/108433370/206251278-646250b2-3f97-4b78-b139-0d53f97bf6c9.png)

![image](https://user-images.githubusercontent.com/108433370/206251375-eddc9c5e-731d-4623-8f2b-b56600872619.png)


==========================================================================

## RESULTS / MODEL TRAINING RESULTS

r-score

![image](https://user-images.githubusercontent.com/108433370/206318974-5569dca0-2e8a-47c8-a77a-1b6bcb430e72.png)

actual vs predicted prices

![image](https://user-images.githubusercontent.com/108433370/206319008-5575b691-fa4c-4f58-b710-0a6ffabee679.png)

RMSE score - base model
![image](https://user-images.githubusercontent.com/108433370/206319038-14f94171-0a97-4639-8bd8-49a97dbeb568.png)

RMSE score - after fine tuning

![image](https://user-images.githubusercontent.com/108433370/206319092-b8846708-badd-4e51-bb68-272a46f888d4.png)









