import numpy as np
import matplotlib.pyplot as plt

plt.style.use("ggplot")

from helpers import generate1, generate2, generate3

X, y = generate1()

from  scipy.stats import norm


class GaussianNaiveBayes():
    def __init__(self):
        self.means: np.ndarray = {} 
        # Słownik, który docelowo powinien zawierać tablicę/wektor warunkowych średnich dla każdego atrybutu 
        # Każda tablica/wektor powinna być typu np.array
        # np. 1) means[1] powinno zawierać wektor średnich wartości atrybutów  dla klasy o indeksie 1
        #     2) means[0][1] powinno zawierać średnią 1 atrybutu dla klasy o indeksie 0
        # (Możesz spróbować zaimplementować efektywniejszą implementację używając macierzy)
        self.stds: np.ndarray = {} 
        # Analogiczna struktura dla odchyleń standardowych
        self.class_log_prob: np.ndarray = None 
        # Wektor zawierający logarytmy prawdopodobieństwa dla każdej z klas 
        # np. class_log_prob[1] zawiera logarytm prawdopodobieństwa, że klasa jest równa 1 P(C=1)
        
    def fit(self, X: np.ndarray[np.floating], y: np.ndarray[np.integer]):
        # TWÓJ KOD TUTAJ - proces uczenia czyli uzupełniania struktur zainicjaliowanych w init()
        #                  odpowiednimi wartościami
        # X jest macierzą gdzie każdy wiersz zawiera kolejną obserwację (typ np.array) 
        # y jest wektorem wartości indeksu klasy (0 lub 1). Jego wartości odpowiadają kolejnym wierszom X

        # Obliczenie prawdopodobieństwa wystąpienia każdej klasy
        self.class_log_prob = np.log(np.bincount(y) / len(y))

        # Zainicjalizowanie tablic na średnie i odchylenia standardowe
        self.means = np.zeros((X.shape[1], len(self.class_log_prob)))
        self.stds = np.zeros((X.shape[1], len(self.class_log_prob)))

        # Obliczenie średnich i odchyleń standardowych dla każdej klasy
        for c_i, c in enumerate(self.class_log_prob):
            for ch_i in range(X.shape[1]):
                self.means[ch_i][c_i] = np.mean(X[y == c_i, ch_i])
                self.stds[ch_i][c_i] = np.std(X[y == c_i, ch_i])
    
    def probability_for_variant(self, x: np.integer, ch: np.integer, c: np.integer) -> np.floating:
        return norm.pdf(x, self.means[ch][c], self.stds[ch][c])

    def probability_for_class(self, X: np.ndarray[np.floating], c: np.integer) -> np.ndarray[np.floating]:
        variant_probabilities = [self.probability_for_variant(x, ch_i, c) for ch_i, x in enumerate(X)]
        numerator = np.prod(variant_probabilities) * self.class_log_prob[c]
        denominator = 0.0
        for c_other_i, c_other in enumerate(self.class_log_prob):
            variant_probabilities = [self.probability_for_variant(x, ch_i, c_other_i) for ch_i, x in enumerate(X)]
            denominator += np.prod(variant_probabilities) * self.class_log_prob[c_other_i]
        return numerator / denominator

    def predict_proba(self, X: np.ndarray[np.floating]) -> np.ndarray[np.floating]:
        # TWÓJ KOD TUTAJ - predykcja - zwrócenie prawdopodobieństwa dla każdej klasy i każdej obserwacji
        # Funkcja powinna zwrócić macierz o dwóch kolumnach (dwie klasy) w której kolejne wiersze 
        # zawierają prawdopodobieństwa P(c|x) przynależności dla klas dla kolejnych obserwacji w macierzy X

        # Inicjalizacja macierzy prawdopodobieństw
        prob = np.zeros((X.shape[0], len(self.class_log_prob)))

        # Obliczenie prawdopodobieństw dla każdej klasy
        for v_i, v in enumerate(X):
            for c_i, c in enumerate(self.class_log_prob):
                prob[v_i][c_i] = self.probability_for_class(v, c_i)

        return prob
    
    def predict(self, X: np.ndarray[np.floating]):
        # Gotowa funkcja wybierająca klasę z największym prawdopodobieństwem
        prob = self.predict_proba(X)
        return np.argmax(prob, axis=1)

gnb = GaussianNaiveBayes()
gnb.fit(X,y)

#Trafność na zbiorze uczącym
prediction = gnb.predict(X)
print(prediction)