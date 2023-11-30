
### Wstęp

Celem jest zaproponowanie metody klasyfikacji, która pozwoli zbudować model o jak największej mocy predykcyjnej. Dysponujemy sztucznie wygenerowanym zbiorem danych _artificial_, w którym zostały ukryte istotne zmienne. Należy dokonać klasyfikacji do dwóch klas. Dokładność modelu będzie mierzona za pomocą miary zrównoważonej dokładności _balanced accuracy_.

Model należy przygotować w dwóch wariantach:

a) ręcznie, czyli wybrać rodzaj modelu, hiperparametry,

b) wykorzystując poznane frameworki AutoMLowe.

### Zbiór danych

Dane do projektu to sztucznie wygenerowany zbiór, który zawiera 500 zmiennych objaśniających (część z tych kolumn może być zbędna). Zbiór treningowy zawiera 2000 obserwacji, natomiast zbiór testowy 600.

Dostępne są następujące pliki:

-   zbiór treningowy: `artifical_train.data`
-   etykiety zbioru treningowego: `artifical_train.labels`
-   zbiór testowy: `artifical_test.data`

### Oczekiwany wynik

Na przygotowanie rozwiązania projektu będą składały się następujące elementy:

-   jakość predykcji modelu zbudowanego ręcznie na zbiorze testowym mierzona przez `balanced accuracy`
-   jakość predykcji modelu przy użyciu frameworka AutoML na zbiorze testowym mierzona przez `balanced accuracy`
-   raport opisujący wykorzystane metody i wyniki eksperymentów dla obu modeli (maksymalnie 4 strony A4),
-   prezentacja podsumowująca rozwiązanie[*]

[*] - w przypadku gdy została ona zgłoszona na początku semestru

`balanced accuracy` = $\frac{1}{2}\Big(\frac{TP}{P} + \frac{TN}{N}\Big)$

### Szczegóły rozwiązania

Zbiór treningowy oraz etykiety do zbioru treningowego należy wykorzystać do przygotowania modelu. Oczekiwany wynik to wektor prawdopodobieństw przynależności do klasy 1 dla obserwacji ze zbioru testowego.

Rozwiązanie powinno zawierać pliki:

-   `NUMERINDEKSU_artifical_model_prediction.tx` - prawdopodobieństwo przynależności do klasy 1 dla danych testowych z modelu tworzonego ręcznie, gdzie `NUMERINDEKSU` zastępujemy swoim numerem indeksu (przykładowy plik `example_artifical_prediction.txt`). W przypadku pracy zespołowej plik należy nazwać `NUMERINDEKSU1_NUMERINDEKSU2_artifical_model_prediction.txt`.,
-   `NUMERINDEKSU_artifical_automl_prediction.txt` - prawdopodobieństwo przynależności do klasy 1 dla danych testowych z modelu z frameworka AutoML, gdzie `NUMERINDEKSU` zastępujemy swoim numerem indeksu (przykładowy plik `example_artifical_prediction.txt`). W przypadku pracy zespołowej plik należy nazwać `NUMERINDEKSU1_NUMERINDEKSU2_artifical_model_prediction.txt`.,
-   folder `Kody` zawierający wszystkie potrzebne kody do przygotowania rozwiązania pracy domowej,
-   plik `NUMERINDEKSU_raport.pdf` opisujący wykorzystane metody i wyniki eksperymentów (maksymalnie 4 strony). W przypadku pracy zespołowej plik należy nazwać `NUMERINDEKSU1_NUMERINDEKSU2_raport.pdf`.,
-   plik `NUMERINDEKSU_prezentacja.pdf` prezentacja podsumowująca rozwiązanie. `NUMERINDEKSU1_NUMERINDEKSU2_prezentacja.pd` [*].

[*] - w przypadku gdy została ona zgłoszona na początku semestru

### Ocena

Łączna liczba punktów do zdobycia jest równa 40, w tym:

-   jakość kodu (porządek, czytelność, obszerność eksperymentów) - 10 punktów,
-   jakość predykcji rozwiązania modelu tworzonego ręcznie[**] - 10 punktów,
-   jakość predykcji rozwiązania modelu tworzonego z frameworka AutoML[**] - 10 punktów,
-   raport - 10 punktów.

[**] - jakoś predykcji jest oceniana miarą `balanced accuracy` na zbiorze testowym. Wyniki zostaną ustawione w ranking (od najlepszego do najgorszego). Osoba z najlepszym wynikiem (najbliższym wartości 1) zyskuje 10 punktów. Osoba z najgorszym wynikiem (najbliższym wartości 0) zyskuje 0 punktów. Pozostałe wyniki zostaną przeskalowane i zaokrąglone do wartości 0.5.

### Oddanie projektu

Wszystkie punkty z sekcji _Szczegóły rozwiązania_ należy umieścić w katalogu o nazwie `NUMERINDEKSU1`. W przypadku pracy zespołowej katalog należy nazwać `NUMERINDEKSU1_NUMERINDEKSU2`. Tak przygotowany katalog należy umieścić na repozytorium przedmiotu w folderze `homeworks/homework2`

### Terminy

Termin oddania pracy domowej to 16.01.2023 EOD. Prezentacje będą się odbywały zgodnie z listą zamieszczoną na MS Teams.
