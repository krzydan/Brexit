# Machine learning - automatic text classification. Author: Krzysztof Danielak

Polish version of readme BELOW

1. Running the program.

To run the program you need to use the command:

python \_\_main\_\_.py trainfile [--test=testfile] [--c=classifier] 

wher:
- trainfile is the path to the train set
- testfile is the path to the test set
- classifier is the classifier type. The option uses "naive" for Naive Bayes classifier, "knn" for K-nearest Neighbours, "svm" for support-vector machine algorithm and ensemble for the mixed classifire which is the default value for this option.

2. Requirements and how to install it.

You need Python 3.6 and the libraries(required versions are in the brackets): NTLK (3.5), scikit-learn (0.22.2.post1) and numpy (1.18.4). You can use newer versions however it is not guaranteed that the code will work properly.

The commands to install the libriaries:
- pip3 install nltk
- pip3 install sklearn
- pip3 install pandas
- pip3 install numpy

# Uczenie maszynowe - Automatyczna klasyfikacja tekstów. Autor: Krzysztof Danielak

1. Uruchamianie programu.

Program należy uruchomić komendą:

python \_\_main\_\_.py trainfile [--test=testfile] [--c=classifier] 

gdzie:
- trainfile - ścieżka pliku ze zbiorem uczącym
- testfile - ścieżka pliku ze zbiorem testowym
- classifier - typ klasyfikatora. Opcja przyjmuje wartości "naive" dla naiwnego klasyfikatora bayesowskiego, "knn" dla K najbliższych sąsiadów, "svm" dla algorytmu SVM oraz "ensemble" dla klasyfikatora hybrydowego, który jest domyślny.

2. Wymagane pakiety oprogramowania i ich instalacja.

Wymagany jest Python w wersji 3.6 oraz biblioteki (wersje podane w nawiasach) nltk (3.5), scikit-learn (0.22.2.post1) oraz numpy (1.18.4). Można użyć nowszych wersji, nie ma jednak gwarancji prawidłowego działania.

Biblioteki należy zainstalować komendami: 
- pip3 install nltk
- pip3 install sklearn
- pip3 install pandas
- pip3 install numpy
