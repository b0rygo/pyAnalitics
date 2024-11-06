import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.interpolate import griddata
from matplotlib.ticker import LinearLocator, FormatStrFormatter

titanic = sns.load_dataset('titanic')
titanic_clear = sns.load_dataset('titanic').dropna()
titanic_encoded = pd.get_dummies(titanic_clear, columns=['survived'])

#print(titanic_encoded)

def wykres3D():
    x = titanic_encoded['age'].values
    y = titanic_encoded['fare'].values
    z = titanic_encoded['sibsp'].values


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d') # Tworzy nową oś 3D w figurowym obiekcie w rozmiarze 1x1 z 1 wykresem
    ax_scatter = ax.scatter(x,y,z,cmap='viridis')

    ax.set_title('Wykres przedstawiający wiek, opłatę oraz liczbę rodzeństwa/małżonków')
    ax.set_xlabel('age')
    ax.set_ylabel('fare')
    ax.set_zlabel('sibsp')
    plt.show()

def wykres_rozrzutu():
    a = titanic_encoded['age']
    b = titanic_encoded['parch']

    plt.figure(figsize=(10, 6))  # Tworzy okno w rozmiarach 10 na 6 cala gdzie beda wszystkie wyswietlane wyniki
    sns.scatterplot(data=titanic_encoded, x=a, y=b)  # tworzy wykres rozrzutu
    plt.title('Rozrzut miedzy wiekiem a liczba rodzicow/dzieci')
    plt.xlabel('Wiek')
    plt.ylabel('Liczba rodzicow/dzieci')
    plt.show()

def mapa_cieplna():
    correlation_matrix = titanic_encoded.select_dtypes(include='number').corr()
        # Mapa cieplna
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Mapa cieplna korelacji')
    plt.show()

def wykres_pudelkowy():
    # Wykres pudełkowy
    sns.boxplot(x='age', y='sex', data=titanic_encoded, hue='class')
    plt.title('Rozklad wieku dla kazdej plci')
    plt.xlabel('Wiek')
    plt.ylabel('Plec')
    plt.show()

def wykres_slupkowy():
    passenger_count = titanic_encoded.groupby(['class', 'embarked']).size().reset_index(name='count')
    sns.barplot(x='class', y='count', hue='embarked', data=passenger_count)
    plt.title('Wykres slupkowy przedstawiajacy liczbe pasazerow dla kazdej klasy i portu zaokretowania')
    plt.xlabel('Liczba')
    plt.ylabel('Klasa')
    plt.legend(title='Port zaokrętowania')
    plt.show()

def wykres_powierzchniowy():
    # Make data.
    X = titanic_clear['age'].values
    Y = titanic_clear['fare'].values
    Z = titanic_clear['sibsp'].values

    x_grid = np.linspace(X.min(), X.max(), 100) # TWORZY SIATKE Z X_MIN, X_MAX - 100 WARTOSCI
    y_grid = np.linspace(Y.min(), Y.max(), 100)
    x_grid, y_grid = np.meshgrid(x_grid, y_grid) # TWORZY SIATKE 2D
    z_grid = griddata((X, Y), Z, (x_grid, y_grid), method='linear') # INTERPOLACJA WARTOSCI NA SIATCE (X_GRID, Y_GRID) OPIERAJAC SIE NA WARTOSCIACH X Y Z

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(x_grid, y_grid, z_grid, cmap='coolwarm', linewidth=0, antialiased=False)

    ax.zaxis.set_major_locator(LinearLocator(10)) # Ustala liczbę głównych znaczników na osi Z.
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f')) # Ustala formatowanie wartości na osi Z na dwie liczby po przecinku.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_title('Wykres powierzchniowy: Wiek, Opłata i Liczba Rodzeństwa/Małżonków')
    ax.set_xlabel('Wiek (age)')
    ax.set_ylabel('Opłata (fare)')
    ax.set_zlabel('Liczba Rodzeństwa/Małżonków (sibsp)')
    plt.show()

print(titanic_clear)
wykres3D()
wykres_rozrzutu()
mapa_cieplna()
wykres_pudelkowy()
wykres_slupkowy()
wykres_powierzchniowy()