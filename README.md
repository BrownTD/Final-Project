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







==========================================================================

## RESULTS / MODEL TRAINING RESULTS













==========================================================================

## RESULTS
