import numpy as np
import pandas as pd
from scipy.stats import shapiro
from ydata_profiling import ProfileReport
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import emoji
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from io import StringIO


class ESD:
    def __init__(self):
        self.data = pd.read_csv('data_sample/sample.csv')
        self.columns_intial_int = []
        self.columns_names = list(self.data.columns)
        self.feature_importance_results = {}
        self.scaler = StandardScaler()

    def reading(self):
        return self.data.sample(5)

    def data_describe(self):
        return self.data.describe()

    def data_profiling(self):
        profile = ProfileReport(self.data)
        profile.to_file("data_sample/esd.html")

    def find_data_type_int(self):
        self.columns_intial_int = [column for column in self.data.columns if
                                   self.data[column].dtype in ['float64', 'int64']]

    def find_datatypes(self):
        return {col: str(self.data[col].dtype) for col in self.data.columns}

    def shapiro_test(self):
        shapiro_results = {}
        for column in self.data.columns:
            d1 = self.data[column].dropna()
            if not d1.empty:
                stat, p_value = shapiro(d1)
                shapiro_results[column] = (stat, p_value)
        return shapiro_results

    def convert_outliers_to_null(self, threshold=3):
        for column in self.data.columns:
            if self.data[column].dtype in ['float64', 'int64']:
                z_scores = np.abs((self.data[column] - self.data[column].mean()) / self.data[column].std())
                self.data[column] = self.data[column].mask(z_scores > threshold)

    def apply_yeo_johnson_transformation(self, column):
        pt = PowerTransformer(method='yeo-johnson')
        self.data[column] = pt.fit_transform(self.data[[column]])
        return self.data

    def fill_missing_categorical(self):
        for column in self.data.columns:
            if self.data[column].isnull().sum() > 0 and self.data[column].dtype == 'object':
                mode_value = self.data[column].mode()[0]
                self.data[column] = self.data[column].fillna(mode_value)
        return self.data

    def data_type_with_float(self):
        for column in self.data.columns:
            if self.data[column].dtype == 'float64':
                median = self.data[column].median()
                self.data[column] = self.data[column].fillna(median)
                self.data[column] = self.data[column].astype('int64')
        return self.data

    def data_type_with_int(self):
        self.data = self.data_type_with_float()
        for column in self.data.columns:
            if self.data[column].isnull().sum() > 0 and self.data[column].dtype == 'int64':
                self.convert_outliers_to_null()
                stat, p_value = self.shapiro_test().get(column, (None, None))
                if p_value is not None:
                    if p_value > 0.05:
                        self.data[column].fillna(self.data[column].mean(), inplace=True)
                    else:
                        self.data[column].fillna(self.data[column].mean(), inplace=True)
                        self.data = self.apply_yeo_johnson_transformation(column)
        return self.data

    def emojis_to_text(self, text):
        return emoji.demojize(text)

    def apply_emoji_functions(self):
        for column in self.data.columns:
            if self.data[column].dtype == 'object':
                self.data[column] = self.data[column].apply(self.emojis_to_text)

    def object_text_analysis(self):
        self.data = self.fill_missing_categorical()
        self.apply_emoji_functions()
        for column in self.data.columns:
            if self.data[column].dtype == 'object':
                unique_percentage = (self.data[column].nunique() / len(self.data)) * 100
                if unique_percentage <= 20:
                    frequency_encoding = self.data[column].value_counts().to_dict()
                    self.data[column] = self.data[column].map(frequency_encoding)
                else:
                    sentiment_scores = self.data[column].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
                    self.data[column] = sentiment_scores.astype('int64')
        return self.data

    def calculating_iqrs_qd(self):
        result = {}
        for column in self.data.columns:
            q1 = self.data[column].quantile(0.25)
            q3 = self.data[column].quantile(0.75)
            iqr = q3 - q1
            qd = iqr / 2
            result[column] = {'IQR': iqr, 'QD': qd}
        return result

    def skewness(self):
        result_skewness = {}
        for column in self.data.columns:
            skewness = self.data[column].skew()
            result_skewness[column] = {'Skewness': skewness}
        return result_skewness

    def cross_tabulation(self):
        column_names = self.columns_intial_int
        if len(column_names) < 2:
            raise ValueError("Not enough integer columns for cross-tabulation.")
        return pd.crosstab(index=self.data[column_names[0]], columns=[self.data[col] for col in column_names[1:]])

    def calculate_pearson_correlation(self):
        return self.data.corr(method='pearson')

    def zscore_normalization(self):
        self.data[self.data.columns] = self.scaler.fit_transform(self.data[self.data.columns])
        return self.data

    def feature_importance(self):
        for target_column in self.data.columns:
            if self.data[target_column].dtype in ['float64', 'int64', 'object']:
                X = self.data.drop(columns=[target_column])
                y = self.data[target_column]

                model = RandomForestClassifier() if y.dtype == 'object' else RandomForestRegressor()
                model.fit(X, y)

                self.feature_importance_results[target_column] = pd.DataFrame({
                    'Feature': X.columns,
                    'Importance': model.feature_importances_
                }).sort_values(by='Importance', ascending=False)

        return self.feature_importance_results

    def save_transformed_data(self):
        self.data.to_csv('data_sample/sample1.csv', index=False)

    def data_Preprocessing(self):
        self.data = self.data_type_with_int()
        self.data = self.object_text_analysis()
        self.save_transformed_data()
        return self.data


# Usage
esd = ESD()
print(esd.data_Preprocessing())
