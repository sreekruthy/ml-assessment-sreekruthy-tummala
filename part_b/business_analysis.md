# B1. Problem Formulation

## (a) 
- Target variable: items_sold (number of items sold per store per month)
- Input features: store_size, location_type, competition_density, promotion_type, month, is_weekend, is_festival, Store ID 

This is a supervised machine learning regression problem 

Justification: The goal is to predict a continuous numerical value(items sold). It builds relationship between dependent(target) and independent(feature) variables. Here, model will learn patterns between store characteristics, promotions and time factors to estimate sales volume. By predicting the volume for each of the five promotion types, the retailer can choose the one that gives highest predicted value.

## (b)
Items sold is a more reliable target than total revenue because revenue can be affected by price changes, discounts or product mix, which may not reflect true customer demand. In some cases, high discount may increase items sold but reduce revenue, making revenue an incorrect performance measure.

This illustrates the principle that the target variable should directly reflect the business objective. In the given scenario, the goal is to maximize sales volume, not total sales revenue.

## (c)
An alternative modelling strategy is to use a segmented or clustered model. We can build separate models for urban, semi-urban and rural stores, or store specific features can be included.

Justification: This is because different stores respond differently to promotions due to location, customer behavior, store size and competition. A segmented approach improves model accuracy by capturing these differences. We can group stores into clusters using K-means and train a separate model for each cluster. This allows the model to learn specific patterns.



# B2. Data and EDA Strategy

## (a)
The four tables (transactions, store attributes, promotions and calendar) would be joined using common keys such as store_id, promotion_id and transaction_date.

The grain of final dataset is one row per store per month, where each row contains the promotion applied in that month along with store and temporal features.

Aggregations include:
- Total items sold per store per month
- Average footfall
- Promotion applied in that month
- Calendar flags (weekend, festival)
This helps for monthly prediction at the store level.

## (b)
1. Promotion vs Sales (bar plot):
   Compare average items sold across promotion types to see which performs best

2. Sales over Time (line plot):
   Identify trends, seasonality and peak months

3. Sales by Store type (box plot):
   Compare urban, semi-urban and rural store performance

4. Correlation Heatmap:
   Identify relationships between numerical features like competition density and sales.

These analyses help in feature engineering, such as creating seasonal features or identifying important predictors.

## (c)
The model may become biased toward predicting outcomes without promotions and can give incorrect answer when predicting sales with promotions.
To address this:
- use downsampling to reduce rows without promotions
- SMOTE to create new rows with promotions
- ensure that there is sufficient promotion data 
- train separate models for promotion vs no promotion



# B3. Model Evaluation and Deployment

## (a)
A time-based split should be used, where earlier data is used for training and the most recent months for testing.

A random split is inappropriate because it mixes past and future data, causing data leakage and unrealistic evaluation.

Evaluation metrics:
- RMSE: measures large errors more heavily by penalising
- MAE: measures average prediction error for items sold count

Lower values indicate better performance. These metrics show how accurately the model predicts items sold.

## (b)
Feature importance plots can be used to understand why different promotions are recommended.
December may have festivals or higher demand(more festival flags), making loyalty points more effective while March may have lower demand(low footfall), where discounts work better. By analyzing feature importance and input values (like month, festival flags), we can explain how seasonal factors influence the model’s decisions. I will tell the team that in December, customers are already shopping for gifts, so Loyalty Points reward their high spend and in March, shopping is slow, so we need the Flat Discount as a hook to get them to stores. 

## (c)
1. Save the model:
   Use joblib or pickle to save the trained model.

2. Prepare new data:
   Each month, collect new store, promotion and calendar data and transforms it into the Grain format and gives it to saved model.

3. Generate predictions:
   Load the model and predict items sold for each store and promotion.

4. Monitoring:
   Compare the model’s predicted sales vs. actual sales at the end of each month. RMSE and MAE can be used to track model performance. If the error (MAE) increases significantly over 2-3 months(performance of model drops), it’s a signal that customer behavior has changed and the model needs retraining with updated data.
