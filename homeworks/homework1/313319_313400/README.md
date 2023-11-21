# Analiza tunowalności hiperparametrów

Wszelkie informację o procesie przeprowadzenia eksperymentu znajdują się w raporcie. Poniżej zawarto informacje dotyczące uruchomienia kodu oraz omówiono szczegóły implementacyjne. Analiza wyników ze względu na swoją obszerność została szczegółowo opisana w pliku Analisis.ipynb, w raporcie znajduje się skrót analizy, pozbawiony
szczegółowej argumentacji

## Autorzy

- Lubaszka Damian
- Mikołaj Nowak

## Opis Folderów

### Kody

Folder zawiera następujące pliki:

- RandomForest.ipynb - zawiera kod pozwalający na stworzenie historii tuningu hiperparametrów dla lasu losowego,
- GradientBoosting.ipynb - zawiera kod pozwalający na stworzenie historii tuningu hiperparametrów dla wzmocnienia gradientowego,
- SVC.ipynb - zawiera kod pozwalający na stworzenie historii tuningu hiperparametrów dla maszyny wektorów nośnych,
- Analisis.ipynb - zawiera kod pozwalący na analizę wszystkich historii przeprowadzonych tutningów hiperparametrów.

Podfolder Utils zawiera pliki zawierające funkcje pomocnicze, których nazwy w dostatecznym stopniu opisuję ich funkcjonalności.

### Wyniki

Folder zawiera historię tunowalności poszczególnych algorytmów uczenia maszynowego w zależności od wykorzystanej metody samplingu. Historia została użyta do analizy wyników i wyciągnięcia wniosków w pliku Analisis.ipynb.

Ponadto w folderze zapisane są pliki .jpg, które są wykresami wygenerowanymi poprzez plik Analisis.ipynb.

### Dokumentacja

- Plik 313319_313400_raport.pdf opisuje eksperyment i wyniki eksperymentów,
- Plik 313319_313400_prezentacja.pptx to prezentacja podsumowująca rozwiązanie.

## Uwagi od autorów

- Kod pomiędzy notatnikami powtarza się, ale celowo nie został wyabstrahowany aby zwiększyć czytelność i ułatwić zrozumienie kodu bez konieczności przenoszenia się pomiędzy wieloma plikami.

- Outputy dla wszystkich plików .ipynb zostały usunięte.

- Nie stworzono specjalnej komórki w plikach z folderu Kody, która umożliwiała by pobranie wszystkich niezbędnych bibliotek. Jednak liczymy, że osoba uruchamiająca program będzie miała wszystkie biblioiteki zanstalowane. Wszystkie biblioteki zostały zainstalowane w defaultowych wersjach. Wersja używanego Pythona to 3.10.

- Pliki w folderze wyniki umożliwiają oddzielenie logiki generowania danych od ich analizowania, co umożliwia osobie uruchomienie pliku Analisis.ipynb, bez wcześniejszego uruchomienia plików do szukania hiperparametrów, dzięki dostarczonej historii tuningu.

- Raport zawiera więcej niż 4 strony A4. Zgodnie z umową z labów (mniej wiecej 4 strony tesktu, reszta to strony z wykresami i strony formatujące).
