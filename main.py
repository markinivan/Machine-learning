import pandas as pd
import numpy as np
from scipy.io import arff
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("\n=== 1. ЧТЕНИЕ ДАННЫХ ===")
def load_arff_scipy(file_path):
    try:
        data, meta = arff.loadarff(file_path)
        df = pd.DataFrame(data)
        for col in df.columns:
            if df[col].dtype == object and isinstance(df[col].iloc[0], bytes):
                df[col] = df[col].str.decode('utf-8')

        print("Файл успешно загружен с помощью scipy!")
        return df

    except Exception as e:
        print(f"Ошибка при загрузке: {e}")
        return None

file_path = '1year.arff'
df = load_arff_scipy(file_path)

if df is not None:
    print(f"Размер данных: {df.shape}")
    print("Первые 5 строк:")
    print(df.head())
    print("\nИнформация о данных:")
    print(df.info())

print("\n=== 2. РАЗБИЕНИЕ НА ВЫБОРКИ ===")

X = df.drop('class', axis=1)
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

print(f"Обучающая выборка: {X_train.shape[0]} samples")
print(f"Тестовая выборка: {X_test.shape[0]} samples")
print(f"Соотношение классов в обучающей выборке:")
print(y_train.value_counts(normalize=True))
print(f"Соотношение классов в тестовой выборке:")
print(y_test.value_counts(normalize=True))

print("\n=== 3. ВИЗУАЛИЗАЦИЯ И АНАЛИЗ ===")

print("Базовые статистики числовых признаков:")
print(X_train.describe())

plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
y_train.value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
plt.title('Распределение классов (Обучающая выборка)')
plt.xlabel('Класс (0 - не банкрот, 1 - банкрот)')
plt.ylabel('Количество')
for i, v in enumerate(y_train.value_counts()):
    plt.text(i, v, str(v), ha='center', va='bottom')

plt.subplot(2, 3, 2)
X_train['Attr1'].hist(bins=50, alpha=0.7, label='Attr1')
X_train['Attr2'].hist(bins=50, alpha=0.7, label='Attr2')
plt.title('Распределение признаков Attr1 и Attr2')
plt.legend()

plt.subplot(2, 3, 3)
X_train[['Attr1', 'Attr2', 'Attr3']].boxplot()
plt.title('Boxplot признаков (выбросы)')
plt.xticks(rotation=45)

plt.subplot(2, 3, 4)
corr_matrix = X_train.iloc[:, :10].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
            square=True, fmt='.2f')
plt.title('Корреляционная матрица (первые 10 признаков)')

plt.subplot(2, 3, 5)
class_means = pd.DataFrame({
    'Не банкроты': X_train[y_train == 0].mean()[:5],
    'Банкроты': X_train[y_train == 1].mean()[:5]
})
class_means.plot(kind='bar', ax=plt.gca())
plt.title('Средние значения признаков по классам')
plt.xticks(rotation=45)

plt.subplot(2, 3, 6)
feature_variance = X_train.var().sort_values(ascending=False)[:10]
feature_variance.plot(kind='bar')
plt.title('Топ-10 признаков по разбросу')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

print("\nИНТЕРПРЕТАЦИЯ КОРРЕЛЯЦИЙ:")
high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i + 1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > 0.7:
            high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j],
                                    corr_matrix.iloc[i, j]))

if high_corr_pairs:
    print("Обнаружены высокие корреляции (>0.7):")
    for pair in high_corr_pairs[:5]:  # Показываем первые 5
        print(f"  {pair[0]} - {pair[1]}: {pair[2]:.3f}")
else:
    print("Высоких корреляций (>0.7) не обнаружено")

print("\n=== 4. ОБРАБОТКА ПРОПУЩЕННЫХ ЗНАЧЕНИЙ ===")

missing_before = X_train.isnull().sum().sum()
print(f"Пропущенных значений в обучающей выборке: {missing_before}")

if missing_before > 0:
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    X_train = pd.DataFrame(X_train_imputed, columns=X_train.columns, index=X_train.index)
    X_test = pd.DataFrame(X_test_imputed, columns=X_test.columns, index=X_test.index)

    print("Пропущенные значения заполнены медианой")
else:
    print("Пропущенных значений не обнаружено")

print(f"Пропущенных значений после обработки: {X_train.isnull().sum().sum()}")

print("\n=== 5. ОБРАБОТКА КАТЕГОРИАЛЬНЫХ ПРИЗНАКОВ ===")

categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns
if len(categorical_cols) > 0:
    print(f"Обнаружены категориальные признаки: {list(categorical_cols)}")
    X_train = pd.get_dummies(X_train, columns=categorical_cols, drop_first=True)
    X_test = pd.get_dummies(X_test, columns=categorical_cols, drop_first=True)
    print("Категориальные признаки преобразованы с помощью One-Hot Encoding")
else:
    print("Категориальных признаков не обнаружено - все признаки числовые")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

print("Данные нормализованы с помощью StandardScaler")
print("Проверка средних и стандартных отклонений после нормализации:")
print(f"Средние: {X_train.mean().mean():.2f} (должно быть ~0)")
print(f"Стандартные отклонения: {X_train.std().mean():.2f} (должно быть ~1)")

print("\n=== 7. ЗАПУСК КЛАССИФИКАТОРА KNN ===")

knn_basic = KNeighborsClassifier(n_neighbors=5)
knn_basic.fit(X_train, y_train)

y_train_pred = knn_basic.predict(X_train)
y_test_pred = knn_basic.predict(X_test)

print("Базовый KNN (k=5) обучен")

print("\n=== 8. ОЦЕНКА МОДЕЛИ И ПОДБОР ГИПЕРПАРАМЕТРОВ ===")


def evaluate_model(model, X_train, X_test, y_train, y_test, model_name=""):
    """Оценка качества модели"""
    # Предсказания
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Метрики
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    print(f"\n{model_name}")
    print(f"Точность на обучающей выборке: {train_accuracy:.4f}")
    print(f"Точность на тестовой выборке: {test_accuracy:.4f}")

    cm = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Не банкрот', 'Банкрот'],
                yticklabels=['Не банкрот', 'Банкрот'])
    plt.title(f'Матрица рассогласования\n{model_name}')
    plt.ylabel('Истинный класс')
    plt.xlabel('Предсказанный класс')
    plt.show()

    print("\nДетальный отчет классификации:")
    print(classification_report(y_test, y_test_pred,
                                target_names=['Не банкрот', 'Банкрот']))

    return test_accuracy

evaluate_model(knn_basic, X_train, X_test, y_train, y_test, "KNN (k=5)")

print("\n--- ПОДБОР ОПТИМАЛЬНОГО КОЛИЧЕСТВА СОСЕДЕЙ ---")

k_values = range(1, 21, 2)
train_scores = []
test_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    train_score = knn.score(X_train, y_train)
    test_score = knn.score(X_test, y_test)

    train_scores.append(train_score)
    test_scores.append(test_score)

    print(f"k={k:2d}: Train accuracy = {train_score:.4f}, Test accuracy = {test_score:.4f}")

plt.figure(figsize=(12, 6))
plt.plot(k_values, train_scores, 'o-', label='Обучающая выборка', linewidth=2)
plt.plot(k_values, test_scores, 'o-', label='Тестовая выборка', linewidth=2)
plt.xlabel('Количество соседей (k)')
plt.ylabel('Точность')
plt.title('Зависимость точности от количества соседей')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

optimal_k = k_values[np.argmax(test_scores)]
optimal_score = max(test_scores)

print(f"\nОПТИМАЛЬНЫЙ РЕЗУЛЬТАТ:")
print(f"Лучшее k: {optimal_k}")
print(f"Лучшая точность на тесте: {optimal_score:.4f}")

print("\n--- ФИНАЛЬНАЯ МОДЕЛЬ С ОПТИМАЛЬНЫМ K ---")
final_knn = KNeighborsClassifier(n_neighbors=optimal_k)
final_knn.fit(X_train, y_train)

final_accuracy = evaluate_model(final_knn, X_train, X_test, y_train, y_test,
                                f"Финальный KNN (k={optimal_k})")

from sklearn.inspection import permutation_importance

print("\n--- АНАЛИЗ ВАЖНОСТИ ПРИЗНАКОВ ---")
perm_importance = permutation_importance(final_knn, X_test, y_test,
                                         n_repeats=10, random_state=42)

feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': perm_importance.importances_mean
}).sort_values('importance', ascending=False)

print("Топ-10 самых важных признаков:")
print(feature_importance.head(10))

plt.figure(figsize=(10, 8))
top_features = feature_importance.head(15)
plt.barh(top_features['feature'], top_features['importance'])
plt.xlabel('Важность признака')
plt.title('Топ-15 самых важных признаков для прогнозирования банкротства')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()