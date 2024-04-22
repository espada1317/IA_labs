import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from scipy import stats


def retrieve_and_filter_dataset_from_file(file_path):
    # add names for columns
    columns = ['data1', 'data2', 'complexAge', 'totalRooms', 'totalBedrooms', 'complexInhabitants',
               'apartmentsNr', 'data8', 'medianCompexValue']

    # read csv file associated
    df = pd.read_csv(file_path, names=columns)

    # filter only columns needed for regression
    selected_fields = df[['complexAge', 'totalRooms', 'totalBedrooms', 'complexInhabitants',
                          'apartmentsNr', 'medianCompexValue']]

    return selected_fields


def analyze_with_common_statistical_approach(dataset):
    # display all rows and columns
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    # describe dataset displaying stat metrics - mean, min, quantiles (25, 50, 75), max
    print("\nDataset statistics:")
    print(dataset.describe())

    # check for null fields in each column
    print("\nCalculate number of null values across columns:")
    print(dataset.isnull().sum())

    # calculate Pearson correlation coefficient
    print("\nPearson correlation matrix:")
    print(dataset.corr())

    # histogram for each column
    dataset.hist(bins=20, figsize=(10, 8))
    plt.tight_layout()
    plt.show()


def z_score_cleaning(dataset):
    # calculate Z-scores for each column
    z_scores = stats.zscore(dataset.dropna())
    # define threshold for Z-score
    threshold = 3
    # create a mask for outliers
    outlier_mask = (abs(z_scores) > threshold).any(axis=1)
    # filter the dataset to remove outliers
    filtered_data_zscore = dataset[~outlier_mask]

    return filtered_data_zscore


def iqr_cleaning(dataset):
    # calculate the first quartile (Q1) and third quartile (Q3) for each column
    Q1 = dataset.quantile(0.25)
    Q3 = dataset.quantile(0.75)
    IQR = Q3 - Q1
    # create a mask for outliers using IQR
    outlier_mask_iqr = ((dataset < (Q1 - 1.5 * IQR)) | (dataset > (Q3 + 1.5 * IQR))).any(axis=1)
    # filter the dataset to remove outliers detected by IQR
    filtered_data_iqr = dataset[~outlier_mask_iqr]

    return filtered_data_iqr


def mixed_cleaning(dataset):
    z_score_cleaned_dataset = z_score_cleaning(dataset)
    mixed_cleaned_dataset = iqr_cleaning(z_score_cleaned_dataset)
    return mixed_cleaned_dataset


def retrieve_linear_regression_model(dataset):
    # split dataset in independent and dependent value - cols 3,4,5,6,7 are independent
    x = dataset[['complexAge', 'totalRooms', 'totalBedrooms', 'complexInhabitants', 'apartmentsNr']]
    # col 9 is dependent
    y = dataset['medianCompexValue']

    # split data in training 80% and test 20% with random seed 0
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    model = LinearRegression()
    model.fit(x_train, y_train)

    return model, x_train, x_test, y_train, y_test


def get_stats_for_model(model, x_train, x_test, y_train, y_test):
    y_pred_test = model.predict(x_test)
    y_pred_train = model.predict(x_train)

    test_mse = mean_squared_error(y_test, y_pred_test)
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)

    return test_mse, train_mse, test_r2, train_r2


def display_stats(test_mse, train_mse, test_r2, train_r2):
    print(f'Train Mean Squared Error (MSE): {train_mse}')
    print(f'Test Mean Squared Error (MSE): {test_mse}')
    print(f'Train R-squared (R2) Score: {train_r2}')
    print(f'Test R-squared (R2) Score: {test_r2}')

    metrics_mse = ['Train MSE', 'Test MSE']
    values_mse = [train_mse, test_mse]
    metrics_r2 = ['Train R2', 'Test R2']
    values_r2 = [train_r2, test_r2]

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].bar(metrics_mse, values_mse, color=['blue', 'red'])
    ax[0].set_title('MSE')

    ax[1].bar(metrics_r2, values_r2, color=['green', 'orange'])
    ax[1].set_title('R2')
    ax[1].set_ylim(0, 1)

    plt.show()


def predict_price(model):
    new_house = {
        'complexAge': [10],
        'totalRooms': [3000],
        'totalBedrooms': [500],
        'complexInhabitants': [800],
        'apartmentsNr': [400]
    }

    new_house_df = pd.DataFrame(new_house, columns=['complexAge', 'totalRooms', 'totalBedrooms', 'complexInhabitants',
                                                    'apartmentsNr'])
    predicted_price = model.predict(new_house_df)
    print(f'Predicted Price for the New House: $ {predicted_price[0]}')
    return predicted_price[0]


def retrieve_ridge_model(dataset):
    # split dataset in independent and dependent value - cols 3,4,5,6,7 are independent
    x = dataset[['complexAge', 'totalRooms', 'totalBedrooms', 'complexInhabitants', 'apartmentsNr']]
    # col 9 is dependent
    y = dataset['medianCompexValue']

    # split data in training 80% and test 20% with random seed 0
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    model = Ridge(alpha=1.0)
    model.fit(x_train, y_train)

    return model, x_train, x_test, y_train, y_test


def retrieve_lasso_model(dataset):
    # split dataset in independent and dependent value - cols 3,4,5,6,7 are independent
    x = dataset[['complexAge', 'totalRooms', 'totalBedrooms', 'complexInhabitants', 'apartmentsNr']]
    # col 9 is dependent
    y = dataset['medianCompexValue']

    # split data in training 80% and test 20% with random seed 0
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    model = Lasso(alpha=0.1)
    model.fit(x_train, y_train)

    return model, x_train, x_test, y_train, y_test


def retrieve_elastic_net_model(dataset):
    # split dataset in independent and dependent value - cols 3,4,5,6,7 are independent
    x = dataset[['complexAge', 'totalRooms', 'totalBedrooms', 'complexInhabitants', 'apartmentsNr']]
    # col 9 is dependent
    y = dataset['medianCompexValue']

    # split data in training 80% and test 20% with random seed 0
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    model = ElasticNet(alpha=0.1, l1_ratio=0.5)
    model.fit(x_train, y_train)

    return model, x_train, x_test, y_train, y_test


if __name__ == '__main__':
    # TASK 1.1 - Import your data
    file_location = 'apartmentComplexData.txt'
    retrieved_columns = retrieve_and_filter_dataset_from_file(file_location)
    print(retrieved_columns)

    # TASK 1.2 - Analyze it via common statistical approaches
    analyze_with_common_statistical_approach(retrieved_columns)

    # TASK 1.3 - Cleaned dataset
    mixed_cleaned = mixed_cleaning(retrieved_columns)
    print("\nIQR after Z-score cleaning dataset:")
    print(mixed_cleaned.describe())

    # histogram for each column
    mixed_cleaned.hist(bins=20, figsize=(10, 8))
    plt.tight_layout()
    plt.show()

    # TASK 2
    mixed_cleaned = mixed_cleaning(retrieved_columns)
    model, x_train, x_test, y_train, y_test = retrieve_linear_regression_model(mixed_cleaned)
    test_mse, train_mse, test_r2, train_r2 = get_stats_for_model(model, x_train, x_test, y_train, y_test)
    display_stats(test_mse, train_mse, test_r2, train_r2)

    # TASK 3
    mixed_cleaned = mixed_cleaning(retrieved_columns)
    model, _, _, _, _ = retrieve_linear_regression_model(mixed_cleaned)
    predicted_price = predict_price(model)

    plt.figure(figsize=(10, 6))
    sns.histplot(mixed_cleaned['medianCompexValue'], kde=True)
    plt.axvline(x=predicted_price, color='red')
    plt.show()

    # TASK 4.1
    mixed_cleaned = mixed_cleaning(retrieved_columns)
    model, x_train, x_test, y_train, y_test = retrieve_ridge_model(mixed_cleaned)
    test_mse, train_mse, test_r2, train_r2 = get_stats_for_model(model, x_train, x_test, y_train, y_test)
    display_stats(test_mse, train_mse, test_r2, train_r2)

    # TASK 4.2
    mixed_cleaned = mixed_cleaning(retrieved_columns)
    model, x_train, x_test, y_train, y_test = retrieve_lasso_model(mixed_cleaned)
    test_mse, train_mse, test_r2, train_r2 = get_stats_for_model(model, x_train, x_test, y_train, y_test)
    display_stats(test_mse, train_mse, test_r2, train_r2)

    # TASK 4.3
    mixed_cleaned = mixed_cleaning(retrieved_columns)
    model, x_train, x_test, y_train, y_test = retrieve_elastic_net_model(mixed_cleaned)
    test_mse, train_mse, test_r2, train_r2 = get_stats_for_model(model, x_train, x_test, y_train, y_test)
    display_stats(test_mse, train_mse, test_r2, train_r2)

    # TASK 5
    models = ['Linear', 'Ridge', 'Lasso', 'Elastic Net']

    mixed_cleaned = mixed_cleaning(retrieved_columns)

    model_linear, x_train_linear, x_test_linear, y_train_linear, y_test_linear = retrieve_linear_regression_model(
        mixed_cleaned)
    test_mse_linear, train_mse_linear, test_r2_linear, train_r2_linear = get_stats_for_model(model_linear,
                                                                                             x_train_linear,
                                                                                             x_test_linear,
                                                                                             y_train_linear,
                                                                                             y_test_linear)

    model_ridge, x_train_ridge, x_test_ridge, y_train_ridge, y_test_ridge = retrieve_ridge_model(mixed_cleaned)
    test_mse_ridge, train_mse_ridge, test_r2_ridge, train_r2_ridge = get_stats_for_model(model_ridge, x_train_ridge,
                                                                                         x_test_ridge, y_train_ridge,
                                                                                         y_test_ridge)

    model_lasso, x_train_lasso, x_test_lasso, y_train_lasso, y_test_lasso = retrieve_lasso_model(mixed_cleaned)
    test_mse_lasso, train_mse_lasso, test_r2_lasso, train_r2_lasso = get_stats_for_model(model_lasso, x_train_lasso,
                                                                                         x_test_lasso, y_train_lasso,
                                                                                         y_test_lasso)

    model_elastic, x_train_elastic, x_test_elastic, y_train_elastic, y_test_elastic = retrieve_elastic_net_model(
        mixed_cleaned)
    test_mse_elastic, train_mse_elastic, test_r2_elastic, train_r2_elastic = get_stats_for_model(model_elastic,
                                                                                                 x_train_elastic,
                                                                                                 x_test_elastic,
                                                                                                 y_train_elastic,
                                                                                                 y_test_elastic)

    train_mse_scores = [train_mse_linear, train_mse_ridge, train_mse_lasso, train_mse_elastic]
    test_mse_scores = [test_mse_linear, test_mse_ridge, test_mse_lasso, test_mse_elastic]
    train_r2_scores = [train_r2_linear, train_r2_ridge, train_r2_lasso, train_r2_elastic]
    test_r2_scores = [test_r2_linear, test_r2_ridge, test_r2_lasso, test_r2_elastic]

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    index = range(len(models))
    plt.bar(index, train_mse_scores, 0.35, alpha=0.5, label='Train MSE')
    plt.bar([i + 0.35 for i in index], test_mse_scores, 0.35, alpha=0.5, label='Test MSE')
    plt.title('comparison of MSE scores')

    plt.subplot(1, 2, 2)
    index = range(len(models))
    plt.bar(index, train_r2_scores, 0.35, alpha=0.5, label='Train R2')
    plt.bar([i + 0.35 for i in index], test_r2_scores, 0.35, alpha=0.5, label='Test R2')
    plt.title('comparison of R2 scores')

    plt.tight_layout()
    plt.show()
