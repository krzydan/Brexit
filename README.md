# Machine learning - automatic text classification. <br> Author: Krzysztof Danielak<br />

Polish version of readme BELOW

This is the program that classifies texts (twitts) about Brexit and assings them to one of the three classes - "neutral", "against Brexit" and "Brexit approval".

1. Running the program.

To run the program you need to use the command:

python \_\_main\_\_.py trainfile [--test=testfile] [--c=classifier] [-b] [-n=ngram]

where:
- trainfile is the path to the train set
- testfile is the path to the test set
- classifier is the classifier type. The option uses "naive" for Naive Bayes classifier, "knn" for K-nearest Neighbours, "svm" for support-vector machine algorithm and ensemble for the mixed classifire which is the default value for this option.
- b - it's an optional flag used to choose representation of text used by classifier. If it's used, then the classifier uses bag-of-words model. In the other case, then the classifier uses TF-IDF (Term Frequency-Inverse Document Frequency). 
- ngram is used when the user wants to use the N-gram model. N is a positive integer that is passed to program as ngram. 

2. Requirements and how to install it.

You need Python 3.6 and the libraries(required versions are in the brackets): NTLK (3.5), scikit-learn (0.22.2.post1) and numpy (1.18.4). You can use newer versions however it is not guaranteed that the code will work properly.

The commands to install the libriaries:
- pip3 install nltk
- pip3 install sklearn
- pip3 install pandas
- pip3 install numpy

3. Dataset

The data have to be in the text file. Every text (twitt) has to be in separate row. Here is the pattern: class (tabulation) text.
Each class have to be represented as diferent integer. You can find the example data in ML4_text.txt file.

# Uczenie maszynowe - Automatyczna klasyfikacja tekstów. <br> Autor: Krzysztof Danielak </br>

Niniejszy program klasyfikuje teksty (twitty) do jednej z trzech klas: "neutralna", "przeciw Brexitowi", "za Brexitem".

1. Uruchamianie programu.

Program należy uruchomić komendą:

python \_\_main\_\_.py trainfile [--test=testfile] [--c=classifier] [-b] [-n=ngram]

gdzie:
- trainfile - ścieżka pliku ze zbiorem uczącym
- testfile - ścieżka pliku ze zbiorem testowym
- classifier - typ klasyfikatora. Opcja przyjmuje wartości "naive" dla naiwnego klasyfikatora bayesowskiego, "knn" dla K najbliższych sąsiadów, "svm" dla algorytmu SVM oraz "ensemble" dla klasyfikatora hybrydowego, który jest domyślny.
- b - jest to opcja do wyboru reprezentacji tekstu używanej przez klasyfikator. Jeśli jest użyta, to klasyfikator używa modelu bag-of-words. W przeciwnym wypadku wykorzystuje TF-IDF (Term Frequency-Inverse Document Frequency).
- ngram jest to opcja do wykorzystania modelu N-gram, gdzie N jest liczbą dodatnią podaną w opcji ngram.

2. Wymagane pakiety oprogramowania i ich instalacja.

Wymagany jest Python w wersji 3.6 oraz biblioteki (wersje podane w nawiasach) nltk (3.5), scikit-learn (0.22.2.post1) oraz numpy (1.18.4). Można użyć nowszych wersji, nie ma jednak gwarancji prawidłowego działania.

Biblioteki należy zainstalować komendami: 
- pip3 install nltk
- pip3 install sklearn
- pip3 install pandas
- pip3 install numpy

3. Zbiór danych

Dane muszą znajdować się w pliku tekstowym. Każdy tekst (twitt) musi zostać podany w osobnym wierszu wg następującego wzoru: klasa (tabulacja) tekst.
Każda klasa musi zostać oznaczone różnymi liczbami całkowitymi. Przykładowe dane zamieszczono w pliku ML4_text.
