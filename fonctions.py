import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import statsmodels.api as sm
from urllib.parse import urlencode

import geopandas as gpd
from shapely.geometry import Point
from geopy.geocoders import Nominatim

from sklearn.linear_model import LinearRegression
import sklearn.metrics 
import seaborn as sns
import random

def creer_url_api(colonnes, limit=-1):
    """
    Crée une URL de requête pour sélectionner les colonnes demandées.
    """
    if not colonnes:
        raise ValueError("La liste des colonnes ne peut pas être vide.")

    parametres = {
        "select": ",".join(colonnes),
        "limit": limit
    }

    base_url = (
        "https://equipements.sports.gouv.fr/"
        "api/explore/v2.1/catalog/datasets/data-es/records"
    )

    return f"{base_url}?{urlencode(parametres)}"

def call_csv(url):
    csv_url = url.replace("records", "exports/csv")

    response = requests.get(csv_url)

    if response.status_code != 200:
        raise RuntimeError(
            f"Erreur lors de la récupération du CSV "
            f"(HTTP {response.status_code}) :\n{response.text}"
        )

    with open("equipements_ile_de_france.csv", "wb") as f:
        f.write(response.content)

    return pd.read_csv(
        "equipements_ile_de_france.csv",
        delimiter=";",
    dtype={
        "dep_code_filled": str,
        "new_code": str
    }
    )

def afficher_pourcentage(n):
    compteur = {"i": 0}

    def format_pct(pct):
        i = compteur["i"]
        compteur["i"] += 1

        if i >= n or pct < 2:
            return ""
        return f"{pct:.1f}%"

    return format_pct

def annee_aleatoire(periode):
    if pd.isna(periode):
        return None

    periode = str(periode).strip()

    if periode.startswith("A partir de"):
        debut = int(periode.replace("A partir de", "").strip())
        return random.randint(debut, 2026)

    if periode.startswith("Avant"):
        fin = int(periode.replace("Avant", "").strip())
        return random.randint(1900, fin)

    debut, fin = map(int, periode.split("-"))
    return random.randint(debut, fin)

def obtenir_decennie(annee):
    if pd.isna(annee):
        return None

    annee = int(annee)

    if annee >= 2005:
        return "A partir de 2005"

    if annee < 1945:
        return "Avant 1945"

    if 1945 <= annee < 1965:
        return "1945-1964"

    debut = ((annee - 5) // 10) * 10 + 5
    fin = debut + 9

    return f"{debut}-{fin}"

def regression(x, y):
    x_reg = sm.add_constant(x)
    y_reg = y.values.ravel()
    return(sm.OLS(y_reg, x_reg).fit(cov_type='HC0'))

def prediction(x, y, afficher = True):
    #séparation des données en 2 échantillon 
    X_train, X_test, y_train, y_test = x,x,y,y

    #régression et prédiction
    ols = LinearRegression().fit(X_train, y_train)
    y_pred = ols.predict(X_test)
    
    #calcul des métriques
    rmse = sklearn.metrics.mean_squared_error(y_test, y_pred)
    rsq = sklearn.metrics.r2_score(y_test, y_pred) 
    
    #nuage de points des valeurs observées
    tempdf = pd.DataFrame({"prediction": y_pred, "observed": y_test,
                           "error": y_test - y_pred})
    
    if afficher == True : 
        print("intercept : ",ols.intercept_)
        print("coeffs : ", ols.coef_)
        print("rsq : ", rsq)
        print("rmse : ", rmse)

        fig = plt.figure()
        g = sns.scatterplot(data = tempdf, x = "observed", y = "error")
        g.axhline(0, color = "red")

    return tempdf

def gini(df, X_col, Y_col):

    populations = df[X_col].values
    installations = df[Y_col].values 
    
    # Trier les départements par leur ratio nb_install / X (contrôle)
    sorted_indices = np.argsort(installations / populations)
    sorted_populations = populations[sorted_indices]
    sorted_installations = installations[sorted_indices]
    
    # Compute cumulative sums
    cumulative_populations = np.cumsum(sorted_populations)
    cumulative_installations = np.cumsum(sorted_installations)
    total_population = np.sum(populations)
    total_installations = np.sum(installations)
    
    normalized_populations = cumulative_populations / total_population
    normalized_installations = cumulative_installations / total_installations
    
    # Calcul du coefficient par méthode graphique
    lorenz_x = np.insert(normalized_populations, 0, 0)
    lorenz_y = np.insert(normalized_installations, 0, 0)
    area_under_curve = np.trapezoid(lorenz_y, lorenz_x)
    gini = 1 - 2 * area_under_curve
    
    return gini

def recuperer_licences(federation, limit=100):
    url = "https://data.sports.gouv.fr/api/explore/v2.1/catalog/datasets/lics_dep/records"

    params = {
        "where": f"code_fede_ref = '{federation}'",
        "select": "dep, dep_name, lics, pop, annee",
        "order_by": "dep",
        "limit": limit,
        "offset": 0,
    }

    toutes_les_lignes = []

    while True:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()

        lignes = response.json()["results"]

        if not lignes:
            break

        toutes_les_lignes.extend(lignes)
        params["offset"] += limit

    return pd.DataFrame(toutes_les_lignes)

# Toutes les fonctions relatives à la distance à l'infrastructure la plus proche

def closest_infra(x,y, gdf) :
    point = Point(x,y) # Coord. en format WGS84
    gdf_proj = gdf.to_crs(epsg=3857) 
    point_proj = gpd.GeoSeries([point], crs="EPSG:4326").to_crs(epsg=3857)[0] # Reprojeter les coordonnées (format WGS84) en système métrique pour augmenter la précision
    gdf_proj['distance'] = gdf_proj.geometry.distance(point_proj) # Calcul des distances entre notre point et toutes les infrastructures, grâce à GeoPandas
    closest_id = gdf_proj['distance'].idxmin() # Retourner l'indice du minimul avec idxmin
    dist = gdf_proj.loc[closest_id, 'distance']
    return gdf.loc[closest_id], dist

def closest_infra_address(address, gdf):
    geolocator = Nominatim(user_agent="projet_python") #Géocodage de l'adresse
    location = geolocator.geocode(address)
    #
    if location is None:
        return("Adresse introuvable")
    #
    input_point = Point(location.longitude, location.latitude) #Récupération des coordonnées géographiques
    input_gdf = gpd.GeoDataFrame(geometry=[input_point], crs="EPSG:4326")
    #
    gdf_proj = gdf.to_crs("EPSG:2154") # Projection de gdf en système métrique, base temporaire (pour obtenir la distance en mètres)
    input_gdf_proj = input_gdf.to_crs("EPSG:2154") # Projection des coordonnées géographiques en système métrique
    input_point_proj = input_gdf_proj.geometry.iloc[0] # Extraction du point reprojeté en système métrique
    #
    gdf_proj['distance'] = gdf_proj.geometry.distance(input_point_proj) # Calcul des distances en système métrique
    closest_id = gdf_proj['distance'].idxmin() # Retourner l'indice du minimul avec idxmin
    closest = gdf.loc[closest_id] # Obtenir l'infrastructure la plus proche dans la base principale
    dist = gdf_proj.loc[closest_id, 'distance'] # Obtenir la distance
    return closest, dist, "Distance en m : {}".format(round(dist))

def closest_infra_address_equiptype(address, equip_type, gdf):
    gdf_filter = gdf[gdf['equip_type_name'] == equip_type]
    gdf_filter = gdf_filter.copy()
    return closest_infra_address(address, gdf_filter)

def generate_random_coord() :
    lat_min, lat_max = 43.0, 50.0 # Limites approximatives de latitude de la France métropolitaine
    lon_min, lon_max = -1.0, 7.0 # Limites approximatives de longitude de la France métropolitaine
    random_lat = random.uniform(lat_min, lat_max)
    random_lon = random.uniform(lon_min, lon_max) # Génération de coordonnées aléatoires 
    return random_lon, random_lat

def mean_dist(n, gdf, coord_generator=generate_random_coord) :
    somme = 0
    for _ in range(n) :
        x,y = coord_generator()
        if isinstance(closest_infra(x,y, gdf)[1], float) : # Permet de s'assurer que la distance est bien un float
            somme+=closest_infra(x,y, gdf)[1]
    return somme/n

def mean_dist_by_department(n, geodf, departments, seed=None):
    """Calcule mean_dist pour chaque département en joignant par nom."""
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n doit être un entier strictement positif.")

    departments = departments[["nom", "geometry"]].copy()
    departments["nom_departement"] = departments["nom"].astype(str).str.strip()
    if geodf.empty or departments.empty:
        raise ValueError("Les données ne doivent pas être vides.")

    rng = random.Random(seed)
    resultats = []
    for _, departement in departments.iterrows():
        nom = departement["nom_departement"]

        polygon = gpd.GeoSeries([departement["geometry"]], crs=departments.crs).to_crs(4326).iloc[0]
        min_x, min_y, max_x, max_y = polygon.bounds

        def generate_coord_in_department():
            while True:
                point = Point(rng.uniform(min_x, max_x), rng.uniform(min_y, max_y))
                if polygon.contains(point):
                    return point.x, point.y

        resultats.append({
            "LIBELLE_DEPARTEMENT": nom,
            "mean_dist": mean_dist(
                n,
                geodf,
                coord_generator=generate_coord_in_department,
            ),
            "n": n,
        })

    return pd.DataFrame(resultats)