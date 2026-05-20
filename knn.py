import pickle
import numpy as np
import os

def load_cifar10_batch(file):
    """ 한 개의 배치를 불러오는 함수 """
    with open(file, 'rb') as fo:
        # 파이썬 3 환경에서는 encoding='latin1' 설정이 필요할 수 있습니다.
        dict = pickle.load(fo, encoding='bytes')
    
    # 데이터와 라벨 추출
    X = dict[b'data']
    y = dict[b'labels']
    
    # (N, 3072) 형태로 유지 (k-NN용)
    # 픽셀 값을 0~1 사이로 정규화
    X = X.astype("float") / 255.0
    y = np.array(y)
    
    return X, y

def load_cifar10_all(root_path):
    """ 5개의 훈련 배치와 1개의 테스트 배치를 모두 합치는 함수 """
    xs = []
    ys = []
    
    # 1번부터 5번 배치까지 합치기
    for b in range(1, 6):
        f = os.path.join(root_path, f'data_batch_{b}')
        X, y = load_cifar10_batch(f)
        xs.append(X)
        ys.append(y)
     
    X_train = np.concatenate(xs)
    y_train = np.concatenate(ys)
    
    # 테스트 배치 불러오기
    X_test, y_test = load_cifar10_batch(os.path.join(root_path, 'test_batch'))
    
    return X_train, y_train, X_test, y_test

# 사용 예시 
root_path = './cifar-10-batches-py'
X_train, y_train, X_test, y_test = load_cifar10_all(root_path)

# 2. 데이터 샘플링 
num_training = 5000
num_test = 500

X_train = X_train[:num_training]
y_train = y_train[:num_training].flatten()
X_test = X_test[:num_test]
y_test = y_test[:num_test].flatten()

# 3. Flatten: (N, 32, 32, 3) -> (N, 3072)
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))

print(f"학습 데이터 모양: {X_train.shape}") # (5000, 3072)

def compute_distances(X_test, X_train, metric='l2'):
    num_test = X_test.shape[0]
    num_train = X_train.shape[0]
    # 결과가 담길 (500, 5000) 크기의 행렬
    distances = np.zeros((num_test, num_train))

    for i in range(num_test):
        if metric == 'l1':
            # 한 번에 테스트 이미지 1개와 모든 훈련 데이터 사이의 거리 계산
            # (1, 3072) 와 (5000, 3072)의 연산이므로 메모리 부담이 적음
            distances[i] = np.sum(np.abs(X_train - X_test[i]), axis=1)
        elif metric == 'l2':
            # L2 거리: sqrt(sum((x1 - x2)^2))
            distances[i] = np.sqrt(np.sum(np.square(X_train - X_test[i]), axis=1))
            
    return distances

class KNN:
    def __init__(self):
        pass 

    def train(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1, metric='l2'):
        distances = compute_distances(X, self.X_train, metric)
        num_test = X.shape[0]
        y_pred = np.zeros(num_test, dtype=self.y_train.dtype)

        for i in range(num_test):
            # 가장 가까운 k개의 이웃의 인덱스 찾기
            closest_y_indices = np.argsort(distances[i])[:k]
            closest_y = self.y_train[closest_y_indices]
            # 가장 빈번한 라벨을 예측값으로 선택
            y_pred[i] = np.bincount(closest_y).argmax()

        return y_pred
    
def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)

def confusion_matrix(y_true, y_pred, num_classes=10):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm
    
import matplotlib.pyplot as plt
import seaborn as sns

def nKNN():

    knn = KNN()
    knn.train(X_train, y_train)

    k_choices = [1, 3, 5, 7, 9]
    metrics = ['l1', 'l2']

    for m in metrics:
        for k in k_choices:
            y_test_pred = knn.predict(X_test, k=k, metric=m)
            acc = accuracy_score(y_test, y_test_pred)
            print(f"Metric: {m}, k: {k} => Accuracy: {acc:.4f}")
            
            # 마지막 k값에서 Confusion Matrix 하나 출력해보기
            if k == 9:
                cm = confusion_matrix(y_test, y_test_pred)
                plt.figure(figsize=(10,8))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.title(f"Confusion Matrix (k={k}, metric={m})")
                plt.xlabel("Predicted")
                plt.ylabel("True")
                plt.show()

    num_training = 1000
    num_test = 100

def KNNwithFold():
    knn = KNN()
    num_folds = 5
    
    # 1. 데이터 분할
    X_train_folds = np.array_split(X_train, num_folds)
    y_train_folds = np.array_split(y_train, num_folds)
    
    k_choices = [1, 3, 5, 7, 9]
    # 결과를 저장할 리스트 초기화
    k_to_accuracies = {k: [] for k in k_choices}


    for i in range(num_folds):
        print(f"Processing Fold {i+1}/{num_folds}...")
        
        # Validation/Train set 구성
        X_val = X_train_folds[i]
        y_val = y_train_folds[i]
        X_train_current = np.concatenate([X_train_folds[j] for j in range(num_folds) if j != i])
        y_train_current = np.concatenate([y_train_folds[j] for j in range(num_folds) if j != i])
        
        knn.train(X_train_current, y_train_current)
        

        dists = compute_distances(X_val, X_train_current, metric='l2')
        
        for k in k_choices:
            # 미리 계산된 dists를 사용하여 k값만 바꿔가며 예측
            num_test_val = X_val.shape[0]
            y_val_pred = np.zeros(num_test_val, dtype=y_train.dtype)
            
            for idx in range(num_test_val):
                closest_y_indices = np.argsort(dists[idx])[:k]
                closest_y = y_train_current[closest_y_indices]
                y_val_pred[idx] = np.bincount(closest_y).argmax()
            
            # 정확도 계산 및 저장
            acc = accuracy_score(y_val, y_val_pred)
            k_to_accuracies[k].append(acc)

    # 2. 결과 출력 및 최적의 k 확인
    print("\n=== Cross Validation Results ===")
    best_k = k_choices[0]
    max_avg_acc = 0

    for k in sorted(k_to_accuracies):
        mean_acc = np.mean(k_to_accuracies[k])
        std_acc = np.std(k_to_accuracies[k])
        print(f'k = {k:2d}, Mean Accuracy = {mean_acc:.4f} (std: {std_acc:.4f})')
        
        if mean_acc > max_avg_acc:
            max_avg_acc = mean_acc
            best_k = k
            
    print(f"\n최적의 하이퍼파라미터: k = {best_k} (정확도: {max_avg_acc:.4f})")
    
    plot_cross_validation(k_to_accuracies)

import matplotlib.pyplot as plt

def plot_cross_validation(k_to_accuracies):
    # 그래프 데이터 준비
    k_choices = sorted(k_to_accuracies.keys())
    
    plt.figure(figsize=(10, 6))

    # 각 k값마다 5개의 fold 결과를 산점도로 표시
    for k in k_choices:
        accuracies = k_to_accuracies[k]
        # x축을 k값들로 채움 (예: [k, k, k, k, k])
        plt.scatter([k] * len(accuracies), accuracies, color='blue', alpha=0.5)

    # k값별 평균 및 표준편차 계산하여 에러바 그래프 그리기
    accuracies_mean = np.array([np.mean(v) for k, v in sorted(k_to_accuracies.items())])
    accuracies_std = np.array([np.std(v) for k, v in sorted(k_to_accuracies.items())])

    plt.errorbar(k_choices, accuracies_mean, yerr=accuracies_std, 
                 fmt='-o', color='red', ecolor='gray', capsize=5, label='Mean Accuracy')

    # 그래프 스타일링
    plt.title('Cross-Validation on $k$')
    plt.xlabel('$k$ (Number of Neighbors)')
    plt.ylabel('Cross-Validation Accuracy')
    plt.xticks(k_choices) # x축에 k값들만 표시
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    plt.show()

# KNNwithFold 함수 마지막에 아래 한 줄을 추가하세요:
# plot_cross_validation(k_to_accuracies)

if __name__ == "__main__":
    #nKNN()
    KNNwithFold()

