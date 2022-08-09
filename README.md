#Uczenie maszynowe - Automatyczna klasyfikacja tekstów. Autor: Krzysztof Danielak

1. Uruchamianie programu.

Program należy uruchomić komendą:

python __main__.py trainfile [--test=testfile] [--c=classifier] 

gdzie:
trainfile - ścieżka pliku ze zbiorem uczącym
testfile - ścieżka pliku ze zbiorem testowym
classifier - typ klasyfikatora. Opcja przyjmuje wartości "naive" dla naiwnego klasyfikatora bayesowskiego, "knn" dla K najbliższych sąsiadów, "svm" dla algorytmu SVM oraz "ensemble" dla klasyfikatora hybrydowego, który jest domyślny.

2. Wymagane pakiety oprogramowania i ich instalacja.

Wymagany jest Python w wersji 3.6 oraz biblioteki (wersje podane w nawiasach) nltk (3.5), sklearn(0.22.2.post1), numpy (1.18.4)

Biblioteki należy zainstalować komendami: 
pip3 install nltk
pip3 install sklearn
pip3 install pandas
pip3 install numpy

