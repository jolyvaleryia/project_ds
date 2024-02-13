import dill
import math
from datetime import datetime

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer


def filter_data_1(df):
    columns_to_drop = [
        'session_id',
        'client_id',
        'device_model',
        'utm_keyword',
        'device_os',
        'geo_city'
    ]
    return df.drop(columns_to_drop, axis=1)


def change_types(df):
    import pandas as pd
    df = df.copy()
    df['visit_number'] = df['visit_number'].astype(int)

    df['date'] = df.apply(lambda x: 'T'.join([x['visit_date'], x['visit_time']]), axis=1)
    df['date'] = pd.to_datetime(df['date'], utc=True)
    return df


def remove_outliers(df):
    def calculate_outliers(data):
        q25 = data.quantile(0.25)
        q75 = data.quantile(0.75)
        iqr = q75 - q25
        bounds = (q25 - 1.5 * iqr, q75 + 1.5 * iqr)
        return bounds

    df = df.copy()
    boundaries = calculate_outliers(df['visit_number'])
    df.loc[df['visit_number'] > boundaries[1], 'visit_number'] = int(math.ceil(boundaries[1]))
    return df


def create_features(df):
    df = df.copy()

    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['hour'] = df['date'].dt.hour
    return df


def filter_data_2(df):
    columns_to_drop = [
        'visit_date',
        'visit_time',
        'date'
    ]
    return df.drop(columns_to_drop, axis=1)


def change_data(df):
    df = df.copy()
    df['device_brand'] = df['device_brand'].apply(lambda x: 'other' if x == '(not set)' else x)

    categorical = df.select_dtypes(include=['object']).columns
    for feat in categorical:
        if df[feat].nunique() > 10:
            feat_list = list(dict(df[feat].value_counts()).keys())[0:10]
            df[feat] = df[feat].apply(lambda x: x if x in feat_list else 'other')
    return df


def main():
    import pandas as pd
    print('Prediction Pipeline')

    df = pd.read_csv('data/ga_sessions_target.csv')

    X = df.drop('target', axis=1)
    y = df['target']

    numerical_features = make_column_selector(dtype_include=['int64', 'float64', 'int32'])
    categorical_features = make_column_selector(dtype_include='object')

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    column_transformer = ColumnTransformer(transformers=[
        ('numerical', numerical_transformer, numerical_features),
        ('categorical', categorical_transformer, categorical_features)
    ])

    preprocessor = Pipeline(steps=[
        ('filter_1', FunctionTransformer(filter_data_1)),
        ('types_change', FunctionTransformer(change_types)),
        ('outlier_remover', FunctionTransformer(remove_outliers)),
        ('feature_creator', FunctionTransformer(create_features)),
        ('filter_2', FunctionTransformer(filter_data_2)),
        ('data_change', FunctionTransformer(change_data)),
        ('column_transformer', column_transformer)
    ])

    models = (
        RandomForestClassifier(),
        LogisticRegression(C=2, penalty='l2', solver='liblinear', max_iter=1500, random_state=12,
                           class_weight='balanced'),
        MLPClassifier(hidden_layer_sizes=(50,), activation='logistic')
    )

    best_score = .0
    best_pipe = None
    for model in models:

        pipe = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])

        score = cross_val_score(pipe, X, y, cv=4, scoring='roc_auc')
        print(f'model: {type(model).__name__}, roc_auc_mean: {score.mean():.4f}, roc_auc_std: {score.std():.4f}')

        if score.mean() > best_score:
            best_score = score.mean()
            best_pipe = pipe

    best_pipe.fit(X, y)
    print(f'best model: {type(best_pipe.named_steps["classifier"]).__name__}, roc_auc: {best_score:.4f}')

    date = datetime.now()

    with open(f'model_pipe_{date.strftime("%m%d%Y_%H%M%S")}.pkl', 'wb') as file:
        dill.dump({
            'model': best_pipe,
            'metadata': {
                'name': 'Prediction Pipeline',
                'author': 'Valeryia Joly',
                'version': 1,
                'date': date,
                'type': type(best_pipe.named_steps["classifier"]).__name__,
                'roc_auc': best_score
            }
        }, file, recurse=True)


if __name__ == '__main__':
    main()