import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import learning_curve
import validation_curve
def main():
    df = pd.read_csv(open('training.csv'), header=0)
    class_mapping = {label: idx for idx, label in enumerate(np.unique(df['class']))}
    df['class'] = df['class'].map(class_mapping)
    inv_class_mapping = {v: k for k, v in class_mapping.items()}
    X_train, y_train = df.iloc[:, 1:].values, df.iloc[:, 0].values

    tf = pd.read_csv(open('testing.csv'), header=0)
    tf['class'] = tf['class'].map(class_mapping)
    X_test, y_test = tf.iloc[:, 1:].values, tf.iloc[:, 0].values

    param_range = list(range(1, 41))
    param_grid = [{'n_estimators': param_range, 'max_depth': param_range, 'min_samples_split': list(range(2, 41)),
                   'min_samples_leaf': param_range}]
    rs = RandomizedSearchCV(estimator=RandomForestClassifier(criterion='entropy', random_state=1), param_distributions=param_grid, scoring='accuracy',
                            cv=10, n_jobs=-1, n_iter=100)
    rs = rs.fit(X_train, y_train)
    print('Najlepsza dokladnosc znaleziona przeszukiwaniem losowym: %.4f' % rs.best_score_)
    print('Parametry dla najlepszego wyniku: ')
    print(rs.best_params_)
    forest = rs.best_estimator_
    learning_curve.draw_learing_curve(forest, X_train, y_train)
    forest.fit(X_train, y_train)
    print("Dokladnosc na zbiorze testowym: %.4f" % forest.score(X_test, y_test))

    #wywolanie funkcji do rysowania krzywej walidacji
    #validation_curve_testing('n_estimators', X_test ,y_test, list(range(1,50)))

#para przyjmuje wartosci: 'n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf'
#param_range zakres parametru np: list(range(1,10))
def validation_curve_testing(para, X_train, y_train, param_range):
    forest = RandomForestClassifier(criterion='entropy', n_estimators=22, random_state=1, n_jobs=2)
    if(para == 'n_estimators' or para == 'max_depth' or para == 'min_samples_split' or para == 'min_samples_leaf'):
        validation_curve.draw_validation_curve(forest, X_train,y_train, para, param_range)
    else:
        print("Podano zły parametr")

if __name__ == '__main__':
    main()