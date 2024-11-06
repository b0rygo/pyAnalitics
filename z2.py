import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats
from scipy.interpolate import griddata

#1. ZALADOWOWANIE DANYCH
diamonds = sns.load_dataset('diamonds')
diamonds_clear = sns.load_dataset('diamonds').dropna()

def wykres1D():
    mean_price = diamonds_clear.groupby('color')['price'].mean().sort_index()
    mean_price.plot(kind='line', marker='o')
    plt.title('Srednia cena dla kazdego koloru')
    plt.xlabel('Kolor')
    plt.ylabel('Srednia cena')
    plt.show()

def wykres2D():
    plt.figure(figsize=(12,8))
    sns.scatterplot(diamonds, x='carat', y='price', hue='clarity')
    plt.title('Wykres przedstawiajacy zaleznosc miedzy karatami a cena')
    plt.xlabel('Karaty')
    plt.ylabel('Cena')
    plt.show()

def wykres3D():
    x = diamonds_clear['carat'].values
    y = diamonds_clear['depth'].values
    z = diamonds_clear['price'].values

    x_grid = np.linspace(x.min(),x.max(), 100)  #– Tworzy równomiernie rozłożoną siatkę wartości x od minimum do maksimum x z 100 punktami.
    y_grid = np.linspace(y.min(),y.max(), 100)
    x_grid, y_grid = np.meshgrid(x_grid, y_grid)
    z_grid = griddata((x, y), z, (x_grid, y_grid), method='linear') #– Interpoluje wartości z w siatce (x_grid, y_grid) przy użyciu istniejących danych (x, y, z) metodą liniową.

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d') #111 rozmiar 1x1 z 1 wykresem
    surface = ax.plot_surface(x_grid, y_grid, z_grid, cmap='viridis')
    fig.colorbar(surface, ax=ax, shrink=0.5, aspect=5) #Dodaje pasek kolorów do wykresu surface z pomniejszoną szerokością i zmodyfikowanym aspektem.

    ax.set_title('Wykres przedstawiajacy karaty,glebokosc oraz cene diamentów')
    ax.set_xlabel('carat')
    ax.set_ylabel('depth')
    ax.set_zlabel('price')
    plt.show()

def histogram():
    data = diamonds_clear['depth']

    hist, bins = np.histogram(data, bins=100, density=True)

    # Dopasowanie rozkładu Weibulla
    shape, loc, scale = stats.weibull_min.fit(data, floc=0) ####

    x1 = np.linspace(min(data), max(data), 1000)
    fgp = stats.weibull_min.pdf(x1, shape, loc, scale) #### parametryczna funkcja gestosci prawdopodobienstwa

    fig, ax = plt.subplots()

    # Rysowanie parametrycznej FGP
    ax.plot(x1, fgp, color='b', label='Funkcja Gęstości Prawdopodobieństwa(Weibull)')
    ax.hist(data, bins=100, density=True, alpha=0.6, color='g', label='Histogram')
    # Dodanie tytułu i etykiet
    ax.set_title('Histogram i Funkcja Gęstości Prawdopodobieństwa (Weibull)')
    ax.set_xlabel('Głebokosc')
    ax.set_ylabel('Gęstość')
    # Dodanie legendy
    ax.legend()
    # Wyświetlenie wykresu
    plt.show()

def wykres_rozrzutu():
    a = diamonds_clear['price']
    b = diamonds_clear['x']

    plt.figure(figsize=(10, 6))  # Tworzy okno w rozmiarach 10 na 6 cala gdzie beda wszystkie wyswietlane wyniki
    sns.scatterplot(data=diamonds_clear, x=a, y=b)  # tworzy wykres rozrzutu
    plt.title('Rozrzut miedzy cena a dlugoscia')
    plt.xlabel('Srednia cena')
    plt.ylabel('Srednia dlugosc')
    plt.show()

def mapa_cieplna():
    correlation_matrix = diamonds.select_dtypes(include='number').corr()
    #correlation_matrix1 = diamonds_clear.select_dtypes(include='number').corr()
        # Mapa cieplna
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    #sns.heatmap(correlation_matrix1, annot=True, cmap='coolwarm')
    plt.title('Mapa cieplna korelacji')
    plt.show()

wykres1D()
wykres2D()
wykres3D()
histogram()
wykres_rozrzutu()
mapa_cieplna()