import matplotlib.pyplot as plt
import pandas as pd


def histogram(df, column_name):
    column = df[column_name]

    if not pd.api.types.is_numeric_dtype(column):
        print("Error: Column should be numeric.")
        return

    # Calculating statistics
    stats = {
        'Count': column.count(),
        'Mean': column.mean(),
        'Median': column.median(),
        'Minimum': column.min(),
        'Maximum': column.max(),
        'Standard Deviation': column.std()
    }

    print('Statistics for column {}:'.format(column_name))
    for key, value in stats.items():
        print('{}: {}'.format(key, value))

    # Plotting histogram
    plt.hist(column, bins='auto')
    plt.xlabel(column_name)
    plt.ylabel('Frequency')
    plt.title('Histogram of {}'.format(column_name))
    plt.show()


def correlation(df, column1, column2):
    if not (pd.api.types.is_numeric_dtype(df[column1]) and pd.api.types.is_numeric_dtype(df[column2])):
        print("Error: Both columns should be numeric.")
        return

    # Plotting scatter plot
    plt.scatter(df[column1], df[column2])
    plt.xlabel(column1)
    plt.ylabel(column2)
    plt.title('Scatter Plot of {} vs {}'.format(column1, column2))
    plt.show()

    # Calculating correlation coefficient
    correlation_coefficient = df[[column1, column2]].corr().iloc[0, 1]

    print('Correlation coefficient between {} and {}: {}'.format(column1, column2, correlation_coefficient))

    if correlation_coefficient > 0.8:
        explanation = 'There is a strong positive correlation between {} and {}. As {} increases, {} tends to increase as well.'.format(
            column1, column2, column1, column2)
    elif correlation_coefficient < -0.8:
        explanation = 'There is a strong negative correlation between {} and {}. As {} increases, {} tends to decrease.'.format(
            column1, column2, column1, column2)
    elif correlation_coefficient > 0.2:
        explanation = 'There is a moderate positive correlation between {} and {}. As {} increases, {} tends to increase as well, but the relationship is not very strong.'.format(
            column1, column2, column1, column2)
    elif correlation_coefficient < -0.2:
        explanation = 'There is a moderate negative correlation between {} and {}. As {} increases, {} tends to decrease, but the relationship is not very strong.'.format(
            column1, column2, column1, column2)
    else:
        explanation = 'There is little to no linear relationship between {} and {}. The correlation coefficient suggests a weak or no correlation.'.format(
            column1, column2)

    print(explanation)