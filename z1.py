import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats
from scipy.interpolate import griddata


#1. Załadowanie danych
penguins = sns.load_dataset('penguins')
penguins_clean = penguins.dropna() #CZYSCI ELEMENTY Z BAZY DANYCH KTORE MAJA WARTOSC NAN

#2. Tworzenie wykresu 1D
def wykres1d():
    mean_body_mass_g = penguins.groupby('species')['body_mass_g'].mean()
    mean_body_mass_g.plot( marker='o')
    plt.title('Srednia masa ciała dla kazdego gatunku')
    plt.xlabel('Gatunek')
    plt.ylabel('Srednia dlugosc kielicha')
    plt.show()

#3. Tworzenie wykresu 2D
def wykres2d():
    plt.figure(figsize=(10, 6)) # Tworzy okno w rozmiarach 10 na 6 cala gdzie beda wszystkie wyswietlane wyniki
    sns.scatterplot(data=penguins,x='bill_length_mm', y='bill_depth_mm', hue='species') #tworzy wykres rozrzutu
    plt.title('Rozrzut miedzy dlugoscia dzioba a szerokoscia dzioba')
    plt.xlabel('Sredia dlugosc dzioba')
    plt.ylabel('Srednia szerokosc dzioba')
    plt.show()

#4. Tworzenie wykresu 3D
def wykres3d():
    x = penguins_clean['bill_length_mm'].values #DO X DAJE WARTOSCI Z BILL_LENGTH_MM
    y = penguins_clean['bill_depth_mm'].values
    z = penguins_clean['body_mass_g'].values

    x_grid = np.linspace(x.min(),x.max(), 100) # TWORZY SIATKE Z X_MIN, X_MAX - 100 WARTOSCI
    y_grid = np.linspace(y.min(),y.max(), 100)
    x_grid, y_grid = np.meshgrid(x_grid,y_grid) # TWORZY SIATKE 2D
    z_grid = griddata((x, y), z, (x_grid, y_grid)) # INTERPOLACJA WARTOSCI NA SIATCE (X_GRID, Y_GRID) OPIERAJAC SIE NA WARTOSCIACH X Y Z

    fig =  plt.figure()
    ax = fig.add_subplot(111, projection = '3d') #Dodaje trójwymiarowy subplot do figury
    surface = ax.plot_surface(x_grid, y_grid, z_grid, cmap='viridis') #Tworzy trójwymiarową powierzchnię na podstawie danych x_grid, y_grid, z_grid z kolorem viridis.
    fig.colorbar(surface, ax=ax, shrink=0.1, aspect=5) #Dodaje pasek kolorów do wykresu surface z pomniejszoną szerokością i zmodyfikowanym aspektem.

    ax.set_title('Wykres 3D dlugosc dzioba, szerokosc dzioba i mase ciala!')
    ax.set_xlabel('bill_length_mm')
    ax.set_ylabel('bill_depth_mm')
    ax.set_zlabel('body_mass_g')
    plt.show()

#5.Tworzenie histogramu z dopasowaniem FGP
def histogramFGP():
    data = penguins_clean['flipper_length_mm']

    hist, bins = np.histogram(data, bins = 100, density= True) # Tworzy histogram z danych data z 100 przedziałami o znormalizowanej wartości.
    mu, std = stats.norm.fit(data) #Dopasowanie rozkladu normalnego do danych, a nestepnie wygernerowani wykresu gestosci rozkladu
    x1 = np.linspace(min(data),max(data), 1000)# sitaka wartosci z min, maksymalnymi 1000 wartosci
    fgp = stats.norm.pdf(x1, mu, std) #Oblicza gestosc prawdopodobienstwa rozkladu normalnego w punktach x1, i parametrow paowyzej
    fig, ax = plt.subplots()

       # Rysowanie parametrycznej FGP
    ax.plot(x1, fgp, color='b', label='Funkcja Gęstości Prawdopodobieństwa(FGP)')
    ax.hist(data, bins=100, density=True, alpha=0.6, color='g', label='Histogram') #– Rysuje histogram danych data w kolorze zielonym z przezroczystością 0.6.
        # Dodanie tytułu i etykiet
    ax.set_title('Histogram i Funkcja Gęstości Prawdopodobieństwa (FGP)')
    ax.set_xlabel('Wartości')
    ax.set_ylabel('Gęstość')
        # Dodanie legendy
    ax.legend()
        # Wyświetlenie wykresu
    plt.show()

#6. WYRKES ROZRZUTU
def wykres_rozrzutu():
    a = penguins_clean['body_mass_g']
    b = penguins_clean['bill_length_mm']

    plt.figure(figsize=(10, 6)) # Tworzy okno w rozmiarach 10 na 6 cala gdzie beda wszystkie wyswietlane wyniki
    sns.scatterplot(data=penguins_clean,x=a, y=b,hue='species') #tworzy wykres rozrzutu
    plt.title('Rozrzut miedzy masa ciala a dlugoscia dzioba')
    plt.xlabel('Sredia masa ciala w gramach')
    plt.ylabel('Srednia dlugosc dzioba w mm')
    plt.show()

#7. MAPA CIEPLNA
def mapa_cieplna():
    correlation_matrix = penguins.select_dtypes(include='number').corr()
    #correlation_matrix1 = penguins_clean.select_dtypes(include='number').corr()
        # Mapa cieplna
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm') #– Tworzy mapę cieplną correlation_matrix z adnotacjami, wykorzystując kolory od niebieskiego do czerwonego.
    #sns.heatmap(correlation_matrix1, annot=True, cmap='coolwarm')
    plt.title('Mapa cieplna korelacji')
    plt.show()

wykres1d()
wykres2d()
wykres3d()
histogramFGP()
wykres_rozrzutu()
mapa_cieplna()

