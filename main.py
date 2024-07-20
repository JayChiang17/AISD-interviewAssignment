import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, roc_auc_score, precision_recall_curve
from xgboost import XGBClassifier
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.metrics import roc_curve, auc
warnings.filterwarnings('ignore')

# 載入數據
def load_data():
    train_data = pd.read_csv('train.csv')
    test_data = pd.read_csv('test.csv')
    return train_data, test_data

# 資料預處理
def preprocess_data(data):
    # 刪除不需要的列
    columns_to_drop = ['id', 'CustomerId', 'Surname']
    data = data.drop(columns_to_drop, axis=1, errors='ignore')
    # 編碼處理
    label_encoder = LabelEncoder()
    data['Gender'] = label_encoder.fit_transform(data['Gender'])
    data['Geography'] = label_encoder.fit_transform(data['Geography'])
    # 資料型別轉換
    data['HasCrCard'] = data['HasCrCard'].astype(int)
    data['IsActiveMember'] = data['IsActiveMember'].astype(int)
    # 數值特徵縮放
    scaler = MinMaxScaler()
    data['EstimatedSalary'] = scaler.fit_transform(data[['EstimatedSalary']])
    data['Balance'] = (data['Balance'] == 0).astype(int)
    return data

# 處理類別不平衡
def balance_data(data):
    majority = data[data['Exited'] == 0]
    minority = data[data['Exited'] == 1]
    majority_downsampled = resample(majority, replace=False, n_samples=12000, random_state=42)
    balanced_data = pd.concat([majority_downsampled, minority])
    return balanced_data

# 訓練模型
def train_model(X_train, y_train, X_test, y_test):
    model = XGBClassifier(n_estimators=1000, max_depth=4, learning_rate=0.01, subsample=0.8, colsample_bytree=0.8, use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_train, y_train, 
              eval_set=[(X_train, y_train), (X_test, y_test)],  # 提供訓練集和測試集作為評估集
              early_stopping_rounds=50,
              verbose=100)  # 每100次迭代後列印評估指標
    return model


# 模型評估及視覺化
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_scores = model.predict_proba(X_test)[:, 1]  # 預測機率，用於ROC和Precision-Recall
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    # 計算精確度
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    precision = precision_score(y_test, y_pred)
    print("Precision:", precision)

    # 計算召回率
    recall = recall_score(y_test, y_pred)
    print("Recall:", recall)

    # 計算F1分數
    f1 = f1_score(y_test, y_pred)
    print("F1 Score:", f1)
    plot_roc_curve(y_test, y_scores)
    plot_precision_recall_curve(y_test, y_scores)

# 繪製ROC曲線
def plot_roc_curve(y_test, y_scores):
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

# 繪製精確率-召回率曲線
def plot_precision_recall_curve(y_test, y_scores):
    precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
    plt.figure()
    plt.plot(recall, precision, color='darkorange', lw=2, label='Precision-Recall curve')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='orange')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall Curve')
    plt.legend(loc="upper right")
    plt.show()


def main():
    train_data, test_data = load_data()

    original_test_data = test_data.copy()

    train_data = preprocess_data(train_data)
    train_data = balance_data(train_data)
    test_data = preprocess_data(test_data)

    X = train_data.drop('Exited', axis=1)
    y = train_data['Exited']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 將X_test和y_test也傳遞給train_model函數
    model = train_model(X_train, y_train, X_test, y_test)
    evaluate_model(model, X_test, y_test)

    # 對測試資料進行預測並保存
    test_predictions = model.predict_proba(test_data)[:, 1]
    test_predictions = ( test_predictions >= 0.5).astype(int)  # 使用0.5作為閾值轉換機率為標籤
    original_test_data['Predicted_Exited'] = test_predictions

    # 保存包含預測的測試數據
    original_test_data.to_csv('predictions.csv', index=False)

if __name__ == "__main__":
    main()
