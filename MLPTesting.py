from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score
from numpy import sqrt
from numpy import argmax
from numpy import arange
from sklearn.metrics import roc_curve
from matplotlib import pyplot
import warnings
from sklearn.metrics import precision_recall_curve
import pandas as pd

warnings.filterwarnings('ignore')


def get_dates():
    dates = []
    with open('./data/dates.txt', 'r') as fp:
        for line in fp:
            x = line[:-1]
            dates.append(x)
    return dates


def to_labels(pos_probs, threshold):
    return (pos_probs >= threshold).astype('int')


# calculate the threshold based on the roc curve using precision and recoil to determain what would be the best thresh
def get_roc_pre_rec(model, x_validate, y_validate):
    predictions = model.predict_proba(x_validate)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_validate, predictions)
    fscore = (2 * precision * recall) / (precision + recall)
    ix = argmax(fscore)
    print('Best Threshold=%f, F-Score=%.3f' % (thresholds[ix], fscore[ix]))
    no_skill = len(y_validate[y_validate == 1]) / len(y_validate)
    pyplot.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    pyplot.plot(recall, precision, marker='.', label='Logistic')
    pyplot.scatter(recall[ix], precision[ix], marker='o', color='black', label='Best')
    pyplot.xlabel('Recall')
    pyplot.ylabel('Precision')
    pyplot.legend()
    pyplot.show()
    return thresholds[ix]


# return the top values as True and other values as False
def top_values(numbers, x):
    sorted_numbers = sorted(numbers, reverse=True)
    top_x = sorted_numbers[:x]
    is_top_value = [False] * len(numbers)
    counter = 0
    for i, number in enumerate(numbers):
        if counter >= x:
            break
        if number in top_x and number >= 0.5 and is_top_value[i] == False:
            is_top_value[i] = True
            counter += 1
            top_x.remove(number)
    return is_top_value


def create_sumfile(model, ind, thresh, starting_date, ending_date, save_path):
    dates = get_dates()
    df_model = pd.DataFrame(index=dates[starting_date:ending_date])
    model_name = model.__class__.__name__
    if "Classifier" in model_name:
        model_name = model_name[:-10]
    print(model_name)
    precision, recall, TpCounter, daily_positive, DailyGainer, positionscounter = [], [], [], [], [], []
    target_df = pd.read_csv(f"./data/targets/Targets{ind}.csv")
    param_grid = {
        'hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (100,)],
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'alpha': [0.0001, 0.05],
        'learning_rate': ['constant', 'adaptive'],
    }

    clf = GridSearchCV(MLPClassifier(max_iter=1000), param_grid, cv=3, scoring='precision')
    for i in range(starting_date, ending_date):
        TPcount, TotalGain, positions_opend = 0, 0, 0
        date_df = pd.read_csv(f"./data/dates{ind}/{dates[i]}.csv")
        next_date_df = pd.read_csv(f"./data/dates{ind}/{dates[i + 1]}.csv")
        results_df = pd.read_csv(f"./data/dates{ind}/{dates[i + 2]}.csv")
        x_train = date_df.drop(['ticker'], axis=1)  # the current day data
        x_validate = next_date_df.drop(['ticker'], axis=1)  # next day data
        y_train = target_df[dates[i + 1]]  # takes the next day targets
        y_validate = target_df[dates[i + 2]]  # day after the next target
        scaler = preprocessing.StandardScaler()  # define scaler
        x_train = scaler.fit_transform(x_train)
        x_validate = scaler.transform(x_validate)
        if y_train.sum() == 0 or y_train.sum() == 1:
            precision.append(0)
            recall.append(0)
            daily_positive.append(0)
            DailyGainer.append(0)
            TpCounter.append(0)
            positionscounter.append(0)
        else:
            clf.fit(x_train, y_train)
            predictions = top_values(model.predict_proba(x_validate)[:, 1], thresh)
            precision.append(precision_score(y_validate, predictions))
            recall.append(recall_score(y_validate, predictions))
            daily_positive.append(target_df[dates[i + 2]].sum())
            for j in range(len(predictions)):
                if predictions[j] == True:
                    positions_opend += 1
                    TotalGain = TotalGain + round(results_df.at[j, 'Close Change'] * 100, 2)
                    if y_validate[j] == 1:
                        TPcount += 1
            TpCounter.append(TPcount)
            positionscounter.append(positions_opend)
            if (positions_opend == 0):
                DailyGainer.append(0)

            else:
                DailyGainer.append(TotalGain / positions_opend)

    df_model[f'P-{model_name}'] = precision
    df_model[f'R-{model_name}'] = recall
    df_model[f'TP-{model_name}'] = TpCounter
    df_model[f'DG-{model_name}'] = DailyGainer
    df_model[f'DPO-{model_name}'] = positionscounter
    df_model['DP'] = daily_positive
    df_model.to_csv(f'{save_path}/{ind}/{model_name}-{ind}-{thresh}.csv', index=False)

    return df_model  # return df_model for appending it to all_results

if __name__ == '__main__':
    model = MLPClassifier(alpha=1, max_iter=1000)
    indices = [2.5, 2, 1.5]
    threshes = [5, 20, 50, 100]
    all_results = pd.DataFrame()  # a new DataFrame to store all results
    for ind in indices:
        for thresh in threshes:
            result = create_sumfile(model, ind, thresh, 209, 225, "./data/PrecisionTesting/02.03.2020-23.03.2020")
            all_results = pd.concat([all_results, result], ignore_index=True)

    # save all results in a single CSV file
    all_results.to_csv(f'data/all_results.csv', index=False)