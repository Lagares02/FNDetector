import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Leer dataset desde un archivo CSV
df = pd.read_csv('get_news/Dataset/dataset.csv', header=None, names=['texto', 'clase'])

# Preprocesamiento de datos
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['texto'].values.astype('U'))
y = df['clase']

# Dividir conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Entrenar modelo SVM
clf = SVC(kernel='linear', C=1, probability=True, random_state=42)
clf.fit(X_train, y_train)

# Evaluar modelo en conjunto de prueba
y_pred = clf.predict(X_test)
#print("Accuracy:", accuracy_score(y_test, y_pred))
#print("Classification Report:")
#print(classification_report(y_test, y_pred))

# Utilizar modelo para clasificar nuevos textos
def predict(texto):
    texto_vec = vectorizer.transform([texto])
    proba = clf.predict_proba(texto_vec)[0][1]
    if proba > 0.5:
        resultado = "Es un titular"
        porcentaje = f"{proba*100:.2f}"
    else:
        resultado = "No es un titular"
        porcentaje = f"{100-proba*100:.2f}"
    return ({'resultado': resultado, 'porcentaje': porcentaje})
