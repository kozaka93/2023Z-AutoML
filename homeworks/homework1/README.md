## Wstęp

Celem jest przeanalizowanie tunowalności hiperparametrów 3 wybranych algorytmów uczenia maszynowego na co najmniej 4 zbiorach danych. Do tunowania modeli należy wykorzystać min. 2 różne techniki losowania punktów.

### Metody samplingu

1.  Co najmniej jedna metoda powinna się opierać na wyborze punktów z rozkładu jednostajnego. Przykładami mogą być:

-   Uniform grid search
-   Random Search

2.  Co najmniej jedna technika powinna opierać się na technice bayesowskiej

-   Bayes Optimization
    
    _warto wykorzystać pakiet SMAC3 do dostosowania metody_
    

Wyniki z poszczególnych metod tunowania (historia tuningu) powinny być wykorzystywane do wyznaczenia tunowalności algorytmów.

Tunowalność algorytmów i hiperparametrów powinna być określona zgodnie z definicjami w [Tunability: Importance of Hyperparameters of Machine Learning Algorithms](https://jmlr.org/papers/volume20/18-444/18-444.pdf).

### Punkty, które należy rozważyć:

1.  ile iteracji każdej metody potrzebujemy żeby uzyskać stabilne wyniki optymalizacji
    
2.  określenie zakresów hiperparametrów dla poszczególnych modeli - motywacja wynikająca z literatury
    
3.  tunowalność poszczególnych algorytmów
    
    
    
4.  tunowalność poszczególnych hiperparametrów
    

    
5.  czy technika losownia punktów wpływa na różnice we wnioskach w punktach 3. i 4. dotyczących tunowalności algorytmów i hiperparametrów - Odpowiedź na pytanie czy występuje bias sampling.
    

### Potencjalne punkty rozszerzające PD

-   Zastosowanie testów statystycznych _do porównania różnic wyników pomiędzy technikami losowania hiperparametrów_
-   Zastosowanie **[Critical Difference Diagrams](https://github.com/hfawaz/cd-diagram#critical-difference-diagrams) -** w przypadku zastosowania większej liczby technik losowania punktów
-   Zaproponowanie wizualizacji i analiz wyników innych niż użyte w cytowanym artykule

### Oczekiwany wynik

Na przygotowanie rozwiązania projektu będą składały się następujące elementy:

-   raport opisujący wykorzystane metody i wyniki eksperymentów dla obu technik (maksymalnie 4 strony A4),
-   prezentacja podsumowująca rozwiązanie [*]

[*] - w przypadku gdy została ona zgłoszona na początku semestru

### Szczegóły rozwiązania

Rozwiązanie powinno zawierać pliki:

-   folder `Kody` zawierający wszystkie potrzebne kody do przygotowania rozwiązania pracy domowej,
-   plik `NUMERINDEKSU_raport.pdf` opisujący wykorzystane metody i wyniki eksperymentów (maksymalnie 4 strony). W przypadku pracy zespołowej plik należy nazwać `NUMERINDEKSU1_NUMERINDEKSU2_raport.pdf`.,
-   plik `NUMERINDEKSU_prezentacja.pdf` prezentacja podsumowująca rozwiązanie. `NUMERINDEKSU1_NUMERINDEKSU2_prezentacja.pd` [*].

[*] - w przypadku gdy została ona zgłoszona na początku semestru

### Ocena

Łączna liczba punktów do zdobycia jest równa 40, w tym:

-   jakość kodu (porządek, czytelność) - 10 punktów,
    
-   raport - 30 punktów, w tym:
    
    -   jakość eksperymentu - 10 pkt     
    -   analiza tunowalności algorytmów - 7pkt
    -   analiza tunowalności hiperparametrów - 7 pkt
    -   jakość opisu i wizualizacji - 6 pkt

### Oddanie projektu

Wszystkie punkty z sekcji _Szczegóły rozwiązania_ należy umieścić w katalogu ZIP o nazwie `NUMERINDEKSU1`. W przypadku pracy zespołowej katalog należy nazwać `NUMERINDEKSU1_NUMERINDEKSU2`.
