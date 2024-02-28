from IPython.display import display
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random

pd.options.display.float_format = '{:,.3f}'.format


def run_eda(df) -> None:
    '''Perform an Exploratory Data Analysis on a given dataframe'''
    # intro, shape of df
    # print('Hello, my sweet puppy!\U0001F436 \nIn your dataframe:')
    print(f'\t{df.shape[0]} rows, {df.shape[1]} columns/variables')
    print()

    # dtypes sort
    factor_treshold = 6
    factor_columns = []
    num_columns = []
    str_columns = []
    for colname in df.columns:
        if df[colname].nunique() <= factor_treshold:
            factor_columns.append(colname)
        else:
            if df[colname].dtype.kind in 'iufc':  # signed int, uint, float, complex
                num_columns.append(colname)
            # elif df[colname].dtype.kind == 'O': # 'O' - object
            else:
                str_columns.append(colname)
    print(f'\tNumeric categorical variables (number of unique values is less or equal '
          f'{factor_treshold}):\n', factor_columns)
    print('\tNumeric variables: \n', num_columns)
    print('\tObject type variables:\n',  str_columns)
    
    print()

    # factors stats
    print('Statictics for categorical variables:')
    for colname in factor_columns:
        fstat = pd.DataFrame()
        fstat[colname] = df[colname].value_counts(dropna=False).index
        fstat['Counts'] = df[colname].value_counts(dropna=False).values
        fstat['Frequency'] = df[colname].value_counts(dropna=False,
                                                      normalize=True).values
        display(fstat)
    print()

    # numeric stats
    nstat = df[num_columns].describe()
    nstat.rename(index={'50%': 'median'}, inplace=True)
    nstat.drop(index='count', inplace=True)
    # count outliers and add them in nstat
    iqr = nstat.loc['75%'] - nstat.loc['25%']
    outliers_treshold_q3 = nstat.loc['75%'] + 1.5 * iqr
    outliers_treshold_q1 = nstat.loc['25%'] - 1.5 * iqr
    outliers_count = {}
    for colname in num_columns:
        high = df[colname][df[colname] > outliers_treshold_q3[colname]].size
        low = df[colname][df[colname] < outliers_treshold_q1[colname]].size
        outliers_count[colname] = high + low
    nstat.loc['outliers count'] = outliers_count
    print('Statictics for numeric variables:')
    display(nstat)
    print()

    # missing values
    print(f'Total {df.isna().values.sum()} missing values in '
          f'{df.isna().shape[0]} rows')
    na_colnames = df.isna().sum()[df.isna().sum().values > 0].index
    print('Next columns contain missing values:\n\t', *na_colnames)    
    # duplicated rows
    print(f'Count of duplicated rows: {df.duplicated().sum()}')
    print()
    
    # PLOTS!!!
    # print('\U0001F4AB SOME PLOTS FOR PUPPY FUN!!! \U0001F4AB')
    # missing values
    na_values = df.isna().sum() / len(df)
    na_values = na_values.to_frame(name = 'NA Rate')
    na_values = na_values[na_values['NA Rate'] > 0 ]
    sns.barplot(na_values, x = na_values.index, y='NA Rate', color='pink')
    # plt.xticks(rotation=20)
    plt.xlabel('\nColumns', fontsize=12)
    plt.title('Missing values proportion')
    plt.tight_layout()
    plt.show()
    
    # correlation heatmap
    corr = df[num_columns].corr()
    sns.heatmap(corr, annot=True, annot_kws={"size": 7}, cmap='Pastel1')
    plt.title('Correlation heatmap')
    plt.show()
    
    # histogram with marginal boxplot
    hexadecimal_alphabets = '0123456789ABCDEF'
        # random colors generation
    colors = ["#" + ''.join([random.choice(hexadecimal_alphabets) for j in range(6)]) for i in range(len(num_columns)+1)]
    color_idx = 0
        # plots generation
    for colname in num_columns:
        fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})
        plt.title(f'"{colname}" distribution', style='italic')
        sns.boxplot(df[colname], orient='h', ax=ax_box, color=colors[color_idx])
        sns.histplot(data = df, x = colname, ax = ax_hist, color=colors[color_idx])
        ax_box.set(xlabel='')
        plt.show()
        color_idx += 1