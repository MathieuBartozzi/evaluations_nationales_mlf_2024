
import streamlit as st
import pandas as pd
import plotly.express as px
import statsmodels.api as sm
import numpy as np

st.set_page_config(layout="wide")
st.logo('logo_osui.png',size='large')


# Fonction pour charger un onglet spécifique depuis Google Sheets
@st.cache_data
def load_sheet(file_id, gid):
    url = f"https://docs.google.com/spreadsheets/d/{file_id}/export?format=csv&gid={gid}"
    df = pd.read_csv(url)
    return df

# Charger chaque onglet dans un DataFrame
file_id = st.secrets["google_sheets"]["file_id"]
# file_id = "1DoYkiK9hmuoXnw2J4Hu0DC9K-WV53DH3Mus8oAkMW88"

# Dictionnaire des onglets et leurs identifiants GID
sheets = {
    "cp": "598781322",
    "ce1": "780189338",
    "ce2": "780104876",
    "cm1": "1004968514",
    "cm2": "537652203",
    "6e": "515412955",
    "4e": "1016063815",
    "2nde": "989209309",
    'geo':'191664424'
}

# Colonnes à conserver en string
STRING_COLUMNS = ["Nom d'établissement", "Pays", "Ville", "Statut MLF"]

@st.cache_data
def load_sheet(file_id, gid):
    """Charge un onglet spécifique depuis Google Sheets et détecte les lignes supprimées."""
    url = f"https://docs.google.com/spreadsheets/d/{file_id}/export?format=csv&gid={gid}"

    # Lire les données brutes sans suppression
    raw_df = pd.read_csv(url, encoding="utf-8", dtype=str)

    # Charger les données avec on_bad_lines="skip"
    df = pd.read_csv(url, encoding="utf-8", on_bad_lines="skip", dtype=str)

    # Vérifier le nombre de lignes supprimées
    lines_skipped = len(raw_df) - len(df)

    print(f"⚠️ {lines_skipped} lignes ont été ignorées lors du chargement.")

    return df

def find_conversion_errors(df):
    """Identifie les valeurs non convertibles dans les colonnes numériques."""
    errors = {}
    for col in df.columns:
        if col not in STRING_COLUMNS:
            invalid_values = df[df[col].str.contains(r"[^0-9,.\-]", regex=True, na=False)][col].tolist()
            if invalid_values:
                errors[col] = invalid_values
    return errors

def clean_numeric_columns(df):
    """Remplace les virgules par des points et convertit les valeurs en float."""
    for col in df.columns:
        if col not in STRING_COLUMNS:
            df[col] = df[col].str.replace(',', '.', regex=True)  # Remplacement des virgules
            df[col] = pd.to_numeric(df[col], errors='coerce')  # Convertir en float
    return df

def convert_column_types(df):
    """Convertit les colonnes texte en str et les autres en float après nettoyage."""
    for col in df.columns:
        if col in STRING_COLUMNS:
            df[col] = df[col].astype(str)  # Assurer que les colonnes restent du texte

    # Identifier les erreurs de conversion AVANT nettoyage
    errors = find_conversion_errors(df)
    if errors:
        print("🚨 Erreurs de conversion détectées :")
        for col, values in errors.items():
            print(f"   - {col}: {values[:5]} ...")  # Afficher seulement quelques erreurs
    else:
        print("✅ Aucune erreur de conversion détectée avant nettoyage.")

    df = clean_numeric_columns(df)  # Nettoyer et convertir les nombres
    return df

def process_all_sheets(file_id, sheets):
    """Charge, convertit et vérifie tous les onglets du Google Sheets."""
    dataframes = {}
    for name, gid in sheets.items():
        print(f"\n📥 Chargement du fichier : {name}")
        df = load_sheet(file_id, gid)

        # Vérifier si le fichier a des colonnes Unnamed (cas de res_4)
        if any("Unnamed" in col for col in df.columns):
            print("⚠️ Ce fichier contient des colonnes Unnamed, elles seront supprimées.")
            df = df.loc[:, ~df.columns.str.contains("Unnamed")]

        print(f"🔍 Vérification des types AVANT conversion :")
        print(df.dtypes)

        df = convert_column_types(df)

        print(f"✅ Vérification des types APRÈS conversion :")
        print(df.dtypes)

        dataframes[name] = df
    return dataframes

# Exécuter le processus sur tous les onglets
dataframes = process_all_sheets(file_id, sheets)


# Dictionnaire unique associant chaque compétence à une matière
competences_matiere = {
    # 📘 Compétences en Maths
    "Lire des nombres": "Maths",
    "Résoudre des problèmes": "Maths",
    "Quantifier et dénombrer": "Maths",
    "Comparer des nombres": "Maths",
    "Placer un nombre sur une ligne numérique": "Maths",
    "Reconnaitre des nombres": "Maths",
    "Calculer en ligne": "Maths",
    "Calculer mentalement": "Maths",
    "Nommer, lire, écrire, représenter des nombres": "Maths",
    "Calculer": "Maths",
    "Ordonner des nombres": "Maths",
    "Calculer avec des nombres entiers": "Maths",
    "Résolution de problème : résoudre des problèmes en utilisant des nombres, des données et des grandeurs": "Maths",
    "Espaces et géométrie": "Maths",
    "Grandeurs et mesures": "Maths",
    "Nombres et calcul": "Maths",
    "Calcul littéral : Utiliser des expressions littérales pour traduire ou résoudre des problèmes": "Maths",
    "Calcul littéral : Connaître et utiliser des données et la notion de fonction": "Maths",

    # 📕 Compétences en Français
    "Comprendre un texte lu par l’enseignant(e)": "Français",
    "Comprendre des mots lu par l’enseignant(e)": "Français",
    "Comprendre des phrases lues par l’enseignant(e)": "Français",
    "Discriminer des sons": "Français",
    "Reconnaitre des lettres": "Français",
    "Comprendre un texte lu seul(e)": "Français",
    "Comprendre des phrases lues seul(e)": "Français",
    "Comprendre des mots et des phrases lus par l’enseignant(e)": "Français",
    "Écrire des syllabes": "Français",
    "Lire": "Français",
    "Écrire des mots dictés": "Français",
    "Orthographe de base": "Français",
    "Se repérer dans une phrase": "Français",
    "Maîtriser l’orthographe grammaticale de base": "Français",
    "Construire le lexique": "Français",
    "Lire et comprendre un texte": "Français",
    "Comprendre le fonctionnement de la langue : Comprendre et mobiliser le lexique": "Français",
    "Comprendre et s'exprimer à l'oral : comprendre un message oral": "Français",
    "Comprendre le fonctionnement de la langue : Se repérer dans une phrase et identifier sa composition": "Français",
    "Comprendre le fonctionnement de la langue : maîtriser l'orthographe": "Français"
}

# Définition des niveaux de primaire et secondaire
niveaux_primaire = ["cp", "ce1", "ce2", "mc1", "mc2"]
niveaux_secondaire = ["6e", "4e", "2nde"]

# Fonction pour calculer la moyenne d'une matière en fonction des compétences associées
def calculer_moyenne_par_matiere(dataframes, competences_matiere, matiere):
    scores = []
    for niveau, df in dataframes.items():
        for comp in df.columns:
            if comp in competences_matiere and competences_matiere[comp] == matiere:
                scores.extend(df[comp].dropna().tolist())
    return sum(scores) / len(scores) if scores else 0

# Calcul des moyennes
moyenne_maths = calculer_moyenne_par_matiere(dataframes, competences_matiere, "Maths")
moyenne_francais = calculer_moyenne_par_matiere(dataframes, competences_matiere, "Français")


# 📌 Création d'un DataFrame unique pour les moyennes par établissement
moyennes_etablissements = {"Primaire": {}, "Secondaire": {}, "Générale": {}}
etablissements=pd.DataFrame()



# 📌 Calcul des moyennes pour chaque niveau
for matiere in ["Maths", "Français"]:
    moyennes_primaire, moyennes_secondaire, moyennes_generale = [], [], []

    for niveau, df in dataframes.items():
        if not df.empty:
            colonnes_matiere = [col for col in df.columns if competences_matiere.get(col) == matiere]
            if colonnes_matiere:
                df[colonnes_matiere] = df[colonnes_matiere].apply(pd.to_numeric, errors='coerce')
                scores_moyens = df.groupby("Nom d'établissement")[colonnes_matiere].mean().mean(axis=1)

                if niveau in niveaux_primaire:
                    moyennes_primaire.append(scores_moyens)
                if niveau in niveaux_secondaire:
                    moyennes_secondaire.append(scores_moyens)
                moyennes_generale.append(scores_moyens)

    # 📌 Fusion des moyennes pour chaque catégorie
    moyennes_etablissements["Primaire"][matiere] = pd.concat(moyennes_primaire).groupby(level=0).mean()
    moyennes_etablissements["Secondaire"][matiere] = pd.concat(moyennes_secondaire).groupby(level=0).mean()
    moyennes_etablissements["Générale"][matiere] = pd.concat(moyennes_generale).groupby(level=0).mean()

# 📌 Transformer les séries en DataFrames et renommer les colonnes
for categorie in ["Primaire", "Secondaire", "Générale"]:
    moyennes_etablissements[categorie]["Maths"] = moyennes_etablissements[categorie]["Maths"].rename(f"Moyenne Maths {categorie}").to_frame()
    moyennes_etablissements[categorie]["Français"] = moyennes_etablissements[categorie]["Français"].rename(f"Moyenne Français {categorie}").to_frame()

# 📌 Fusion des moyennes avec le DataFrame des établissements
etablissements = pd.DataFrame(dataframes["geo"][["Nom d'établissement", "Ville", "Pays"]].drop_duplicates())

for categorie in ["Primaire", "Secondaire", "Générale"]:
    etablissements = etablissements.merge(moyennes_etablissements[categorie]["Maths"], on="Nom d'établissement", how="left")
    etablissements = etablissements.merge(moyennes_etablissements[categorie]["Français"], on="Nom d'établissement", how="left")

# 📌 Ajout des coordonnées GPS depuis dataframes['geo']
etablissements = etablissements.merge(dataframes['geo'][["Nom d'établissement", "Latitude", "Longitude"]], on="Nom d'établissement", how="left")

# Suppression des établissements sans moyenne et sans coordonnées GPS
etablissements = etablissements.dropna(subset=["Moyenne Maths Générale", "Moyenne Français Générale", "Latitude", "Longitude"])

# 📌 Calcul des moyennes globales
etablissements["Moyenne Maths/Français Générale"] = (etablissements["Moyenne Maths Générale"] + etablissements["Moyenne Français Générale"]) / 2
etablissements["Moyenne Maths/Français Primaire"] = (etablissements["Moyenne Maths Primaire"] + etablissements["Moyenne Français Primaire"]) / 2
etablissements["Moyenne Maths/Français Secondaire"] = (etablissements["Moyenne Maths Secondaire"] + etablissements["Moyenne Français Secondaire"]) / 2

# def carte_etablissements(etablissements, niveau,titre):
#     """
#     Génère une carte interactive des établissements scolaires en fonction d'une moyenne choisie (Primaire, Secondaire, ou Générale),
#     en filtrant ceux qui n'ont pas de données pour le niveau sélectionné.

#     :param etablissements: DataFrame contenant les établissements et leurs coordonnées.
#     :param niveau: "Générale", "Primaire" ou "Secondaire" pour choisir la moyenne affichée.
#     :param titre: Titre de la carte.
#     :return: Figure Plotly.
#     """

#     # Sélection de la colonne correspondante et filtrage des établissements
#     if niveau == "Générale":
#         colonne_moyenne = "Moyenne Maths/Français Générale"
#         df_filtre = etablissements  # Conserver tous les établissements
#     elif niveau == "Primaire":
#         colonne_moyenne = "Moyenne Maths/Français Primaire"
#         df_filtre = etablissements.dropna(subset=[colonne_moyenne])  # Supprime les établissements sans résultats en primaire
#     elif niveau == "Secondaire":
#         colonne_moyenne = "Moyenne Maths/Français Secondaire"
#         df_filtre = etablissements.dropna(subset=[colonne_moyenne])  # Supprime les établissements sans résultats en secondaire
#     else:
#         raise ValueError("Le niveau doit être 'Générale', 'Primaire' ou 'Secondaire'.")

#     # Création de la carte avec les établissements filtrés
#     fig = px.scatter_map(
#         df_filtre,
#         lat="Latitude",
#         lon="Longitude",
#         hover_name="Nom d'établissement",
#         hover_data={
#             "Ville": True,
#             colonne_moyenne: True,
#             "Latitude": False,
#             "Longitude": False
#         },
#         color=colonne_moyenne,  # Dégradé de couleur basé sur la moyenne sélectionnée
#         zoom=0.5,  # Zoom initial
#         height=700,
#         color_continuous_scale="RdYlGn",  # Dégradé de rouge (faible) à vert (fort)
#     )

#     # Fixer la taille des points et l'opacité
#     fig.update_traces(marker=dict(size=20, opacity=0.7))

#     # Mise en page et affichage
#     fig.update_layout(
#         map_style="open-street-map",
#         margin={"r": 0, "t": 0, "l": 0, "b": 0},
#         coloraxis_colorbar=dict(title=None)  # Ajout de la barre de couleur
#     )

#     fig.update_layout(
#         height=350
#         )

#     return fig

def jitter_coordinates(df, lat_col="Latitude", lon_col="Longitude", jitter=0.1):
    """
    Ajoute un léger bruit aléatoire aux coordonnées latitude/longitude
    pour éviter la superposition des points.

    :param df: DataFrame contenant les établissements
    :param lat_col: Nom de la colonne de latitude
    :param lon_col: Nom de la colonne de longitude
    :param jitter: Amplitude du bruit aléatoire ajouté
    :return: DataFrame avec coordonnées ajustées si nécessaire
    """
    coords_count = df.groupby([lat_col, lon_col])[lat_col].transform('count')
    mask = coords_count > 1

    df.loc[mask, lat_col] += np.random.uniform(-jitter, jitter, size=mask.sum())
    df.loc[mask, lon_col] += np.random.uniform(-jitter, jitter, size=mask.sum())

    return df

def carte_etablissements(etablissements, niveau, titre, jitter=0.1):
    """
    Génère une carte interactive des établissements scolaires en fonction d'une moyenne choisie (Primaire, Secondaire, ou Générale),
    en filtrant ceux qui n'ont pas de données pour le niveau sélectionné et en ajustant les coordonnées si nécessaire.

    :param etablissements: DataFrame contenant les établissements et leurs coordonnées.
    :param niveau: "Générale", "Primaire" ou "Secondaire" pour choisir la moyenne affichée.
    :param titre: Titre de la carte.
    :param jitter: Amplitude du bruit aléatoire pour éviter la superposition des points (0 pour désactiver).
    :return: Figure Plotly.
    """
    # Sélection de la colonne correspondante et filtrage des établissements
    if niveau == "Générale":
        colonne_moyenne = "Moyenne Maths/Français Générale"
        df_filtre = etablissements.copy()
    elif niveau == "Primaire":
        colonne_moyenne = "Moyenne Maths/Français Primaire"
        df_filtre = etablissements.dropna(subset=[colonne_moyenne]).copy()

    elif niveau == "Secondaire":
        colonne_moyenne = "Moyenne Maths/Français Secondaire"
        df_filtre = etablissements.dropna(subset=[colonne_moyenne]).copy()
    else:
        raise ValueError("Le niveau doit être 'Générale', 'Primaire' ou 'Secondaire'.")

    # Appliquer le jitter si nécessaire
    if jitter > 0:
        df_filtre = jitter_coordinates(df_filtre, jitter=jitter)

    # Création de la carte avec les établissements filtrés
    fig = px.scatter_map(
        df_filtre,
        lat="Latitude",
        lon="Longitude",
        hover_name="Nom d'établissement",
        hover_data={
            "Ville": True,
            colonne_moyenne: True,
            "Latitude": False,
            "Longitude": False
        },
        color=colonne_moyenne,  # Dégradé de couleur basé sur la moyenne sélectionnée
        zoom=1,  # Zoom initial
        height=350,
        color_continuous_scale="RdYlGn",  # Dégradé de rouge (faible) à vert (fort)
    )

    # Fixer la taille des points et l'opacité
    fig.update_traces(marker=dict(size=15, opacity=0.7))

    # Mise en page et affichage
    fig.update_layout(
        map_style="open-street-map",
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        coloraxis_colorbar=dict(title="Moyenne")  # Ajout de la barre de couleur
    )

    return fig



def creer_scatter_maths_francais(dataframes):
    """
    Crée un graphique de corrélation entre Maths et Français.
    Chaque point représente un établissement, avec une couleur différente selon le pays.
    La taille des points est proportionnelle à l'écart entre les moyennes Maths et Français.
    """

    etablissements = {}

    for niveau, df in dataframes.items():
        for _, row in df.iterrows():
            nom = row["Nom d'établissement"]
            pays = row["Pays"]  # Ajout de la colonne Pays
            maths = row[[col for col in df.columns if competences_matiere.get(col) == "Maths"]].mean()
            francais = row[[col for col in df.columns if competences_matiere.get(col) == "Français"]].mean()

            if pd.notna(maths) and pd.notna(francais):  # Vérifie qu'il y a des valeurs valides
                if nom not in etablissements:
                    etablissements[nom] = {"maths": [], "francais": [], "pays": pays}

                etablissements[nom]["maths"].append(maths)
                etablissements[nom]["francais"].append(francais)

    # Création du DataFrame final
    data = []
    for nom, valeurs in etablissements.items():
        moyenne_maths = sum(valeurs["maths"]) / len(valeurs["maths"])
        moyenne_francais = sum(valeurs["francais"]) / len(valeurs["francais"])
        ecart = abs(moyenne_maths - moyenne_francais)
        data.append([nom, moyenne_maths, moyenne_francais, ecart, valeurs["pays"]])

    df_final = pd.DataFrame(data, columns=["Établissement", "Moyenne Maths", "Moyenne Français", "Taille Point", "Pays"])

    # Création du graphique Scatter avec couleur selon le pays et une taille variable
    fig = px.scatter(
        df_final,
        x="Moyenne Maths",
        y="Moyenne Français",
        hover_name="Établissement",
        size="Taille Point",  # Taille des points proportionnelle à l'écart
        color="Pays"  # Couleur différente par pays
    #     trendline="ols"  # Régression linéaire pour tous
    )

    # Ajout de la régression linéaire pour tous les points
    fig.add_traces(px.scatter(df_final, x="Moyenne Maths", y="Moyenne Français", trendline="ols").data[1])

    # Mise en forme
    fig.update_layout(
        xaxis_title="Moyenne Maths (%)",
        yaxis_title="Moyenne Français (%)",
        template="plotly_white",
        height=400
    )

    return fig

# Fonction pour calculer la moyenne par cycle (primaire ou secondaire)
def calculer_moyenne_cycle(dataframes, competences_matiere, matiere, niveaux):
    scores = []
    for niveau in niveaux:
        if niveau in dataframes:
            df = dataframes[niveau]
            for comp in df.columns:
                if comp in competences_matiere and competences_matiere[comp] == matiere:
                    scores.extend(df[comp].dropna().tolist())
    return sum(scores) / len(scores) if scores else 0

# Calcul des moyennes par cycle
moyenne_maths_primaire = calculer_moyenne_cycle(dataframes, competences_matiere, "Maths", niveaux_primaire)
moyenne_francais_primaire = calculer_moyenne_cycle(dataframes, competences_matiere, "Français", niveaux_primaire)
moyenne_maths_secondaire = calculer_moyenne_cycle(dataframes, competences_matiere, "Maths", niveaux_secondaire)
moyenne_francais_secondaire = calculer_moyenne_cycle(dataframes, competences_matiere, "Français", niveaux_secondaire)

def creer_bar_chart_maths_francais(maths_primaire, francais_primaire, maths_secondaire, francais_secondaire):
    """
    Crée un graphique en barres comparant les moyennes en Maths et Français pour le Primaire et le Secondaire,
    avec suppression de "Matière" dans la légende et du titre de l'axe X.
    """

    # Création du DataFrame
    df = pd.DataFrame({
        "Cycle": ["Primaire", "Primaire", "Secondaire", "Secondaire"],
        "Matière": ["Maths", "Français", "Maths", "Français"],
        "Moyenne": [maths_primaire, francais_primaire, maths_secondaire, francais_secondaire]
    })

    # Création du graphique avec Plotly Express
    fig = px.bar(
        df,
        x="Cycle",
        y="Moyenne",
        color="Matière",
        barmode="group",
        color_discrete_sequence=px.colors.qualitative.G10
    )

    # Suppression du titre de la légende
    fig.update_layout(
        xaxis_title="",  # Supprime "Cycle"
        yaxis_title="Moyenne (%)",
        legend_title_text="",  # Supprime "Matière" dans la légende
        yaxis=dict(range=[0, 100]),  # Échelle de 0 à 100%
        template="plotly_white"
    )

    return fig

def creer_boxplot_combine(dataframes):
    """
    Crée un boxplot combiné avec Maths et Français pour chaque niveau.
    """
    data = []

    for niveau, df in dataframes.items():
        for matiere in ["Maths", "Français"]:
            competences = [col for col in df.columns if competences_matiere.get(col) == matiere]

            for col in competences:
                for score in df[col].dropna():
                    data.append([niveau, score, matiere])

    df_final = pd.DataFrame(data, columns=["Niveau", "Score", "Matière"])

    # Création du boxplot combiné
    fig = px.box(
        df_final,
        x="Niveau",
        y="Score",
        color="Matière",  # Utilisation de la couleur pour différencier Maths/Français
        color_discrete_sequence=px.colors.qualitative.G10
    )

    # Mise en forme
    fig.update_layout(
        xaxis_title="",
        yaxis_title="Résultats (%)",
        #legend_title_text="",
        template="plotly_white",
        height=400,
        legend=dict(
        x=1.05,  # Déplace la légende à droite du graphique
        y=0.5,   # Place la légende à mi-hauteur
        xanchor="left",  # S'assure que la légende est alignée à gauche de x=1.05
        yanchor="middle",  # Centre verticalement la légende
        orientation="v"  # Garde la légende verticale
    ),
        yaxis=dict(range=[0, 130])
    )

    return fig


def evolution_moyenne_globale_par_niveau(dataframes, competences_matiere):
    """
    Crée un graphique en ligne montrant l'évolution des moyennes globales en Français et en Maths
    pour l'ensemble des établissements au cours des différents niveaux (du CP à la 2nde).

    :param dataframes: Dictionnaire contenant les DataFrames des différents niveaux scolaires.
    :param competences_matiere: Dictionnaire associant chaque compétence à une matière.
    :return: Figure Plotly.
    """

    # Liste des niveaux dans l'ordre
    niveaux = ["cp", "ce1", "ce2", "cm1", "cm2", "6e", "4e", "2nde"]
    niveau_labels = {
        "cp": "CP", "ce1": "CE1", "ce2": "CE2", "cm1": "CM1", "cm2": "CM2",
        "6e": "6e", "4e": "4e", "2nde": "2nde"
    }

    # Initialisation d'un dictionnaire pour stocker les moyennes
    moyenne_globale = {"Niveau": [], "Matière": [], "Moyenne": []}

    # Parcours de chaque niveau et calcul des moyennes globales
    for niveau in niveaux:
        if niveau in dataframes:
            df = dataframes[niveau]

            # Calcul de la moyenne en maths et français pour tous les établissements
            maths_moyenne = df[[col for col in df.columns if competences_matiere.get(col) == "Maths"]].mean().mean()
            francais_moyenne = df[[col for col in df.columns if competences_matiere.get(col) == "Français"]].mean().mean()

            if not np.isnan(maths_moyenne):
                moyenne_globale["Niveau"].append(niveau_labels[niveau])
                moyenne_globale["Matière"].append("Maths")
                moyenne_globale["Moyenne"].append(maths_moyenne)

            if not np.isnan(francais_moyenne):
                moyenne_globale["Niveau"].append(niveau_labels[niveau])
                moyenne_globale["Matière"].append("Français")
                moyenne_globale["Moyenne"].append(francais_moyenne)

    # Création du DataFrame final
    df_moyenne_globale = pd.DataFrame(moyenne_globale)

    # Création du graphique en ligne
    fig = px.line(
        df_moyenne_globale,
        x="Niveau",
        y="Moyenne",
        markers=True,
        color="Matière",
        color_discrete_sequence=px.colors.qualitative.G10
    )
    fig.update_layout(
        title='Evolution globale',
        height=400,
        legend_title_text="",
        legend=dict(
            orientation="h",  # Affichage horizontal
            yanchor="top",
            y=-0.3,  # Position sous le graphique
            xanchor="center",
            x=0.5  # Centre la légende horizontalement
        ),
        xaxis_title=None  # Supprime complètement l'axe X
    )

    return fig

competences_fr_primaire = {
    "Comprendre un texte": {
        "Comprendre un texte lu par l’enseignant(e)": {"cp": True, "ce1": False, "ce2": True, "cm1": True, "cm2": True},
        "Comprendre des mots lu par l’enseignant(e)": {"cp": True, "ce1": False, "ce2": False, "cm1": False, "cm2": False},
        "Comprendre des phrases lues par l’enseignant(e)": {"cp": True, "ce1": False, "ce2": False, "cm1": False, "cm2": False},
        "Comprendre des mots et des phrases lus par l’enseignant(e)": {"cp": False, "ce1": True, "ce2": False, "cm1": False, "cm2": False},
        "Comprendre des phrases lues seul(e)": {"cp": False, "ce1": True, "ce2": False, "cm1": False, "cm2": False},
        "Comprendre un texte lu seul(e)": {"cp": False, "ce1": True, "ce2": True, "cm1": True, "cm2": True}
    },
    "Lire et reconnaître les éléments du langage": {
        "Discriminer des sons": {"cp": True, "ce1": False, "ce2": False, "cm1": False, "cm2": False},
        "Lire": {"cp": False, "ce1": True, "ce2": True, "cm1": True, "cm2": True},
        "Se repérer dans une phrase": {"cp": False, "ce1": False, "ce2": True, "cm1": True, "cm2": True},
        "Construire son lexique": {"cp": False, "ce1": False, "ce2": False, "cm1": True, "cm2": True}
    },
    "Écrire et orthographier": {
        "Reconnaitre des lettres": {"cp": True, "ce1": False, "ce2": False, "cm1": False, "cm2": False},
        "Écrire des syllabes": {"cp": False, "ce1": True, "ce2": False, "cm1": False, "cm2": False},
        "Écrire des mots dictés": {"cp": False, "ce1": False, "ce2": True, "cm1": True, "cm2": True},
        "Maîtriser l’orthographe grammaticale de base": {"cp": False, "ce1": False, "ce2": True, "cm1": True, "cm2": True}
    }
}

competences_maths_primaire = {
    "Résolution de problèmes": {
        "Résoudre des problèmes": {"cp": True, "ce1": True, "ce2": True, "cm1": True, "cm2": True}
    },
    "Compréhension et représentation des nombres": {
        "Lire des nombres": {"cp": True, "ce1": False, "ce2": False, "cm1": False, "cm2": False},
        "Ecrire des nombres": {"cp": True, "ce1": True, "ce2": False, "cm1": False, "cm2": False},
        "Comparer des nombres": {"cp": True, "ce1": False, "ce2": False, "cm1": False, "cm2": False},
        "Placer un nombre sur une ligne numérique": {"cp": True, "ce1": True, "ce2": False, "cm1": False, "cm2": False},
        "Reconnaitre des nombres": {"cp": False, "ce1": True, "ce2": False, "cm1": False, "cm2": False},
        "Ordonner des nombres": {"cp": False, "ce1": False, "ce2": True, "cm1": False, "cm2": False},
        "Nommer, lire, écrire, représenter des nombres": {"cp": False, "ce1": False, "ce2": True, "cm1": True, "cm2": True}
    },
    "Calcul et opérations": {
        "Calculer en ligne": {"cp": False, "ce1": True, "ce2": False, "cm1": False, "cm2": False},
        "Calculer mentalement": {"cp": False, "ce1": True, "ce2": False, "cm1": False, "cm2": False},
        "Calculer": {"cp": False, "ce1": False, "ce2": True, "cm1": True, "cm2": True},
        "Quantifier et dénombrer": {"cp": True, "ce1": False, "ce2": False, "cm1": False, "cm2": False},
    }
}

competences_fr_secondaire = {
    "Comprendre un texte": {
        "Lire et comprendre un texte":{"6e": True, "4e":True, "2nde": True},
        "Comprendre et s'exprimer à l'oral : comprendre un message oral":{"6e": True, "4e":True, "2nde": True}},
        "Orthographier": {"Comprendre le fonctionnement de la langue : maîtriser l'orthographe":{"6e": True, "4e":True, "2nde": True}},
    "Reconnaître les éléments du langage": {
        "Comprendre le fonctionnement de la langue : Se repérer dans une phrase et identifier sa composition":{"6e": True, "4e":True, "2nde": True},
        "Comprendre le fonctionnement de la langue : Comprendre et mobiliser le lexique": {"6e": True, "4e":True, "2nde": True}},
}

competences_maths_secondaire = {
    "Résolution et modélisation": {
        "Résolution de problème : résoudre des problèmes en utilisant des nombres, des données et des grandeurs": {"6e": True, "4e":True, "2nde": False},
        "Calcul littéral : Utiliser des expressions littérales pour traduire ou résoudre des problèmes": {"6e": False, "4e":False, "2nde": True},
        "Connaître et utiliser des données et la notion de fonction": {"6e": False, "4e":True, "2nde": True}
    },
    "Procédures et calculs": {
        "Automatismes : Mobiliser directement des procédures et des connaissances": {"6e": True, "4e":True, "2nde": True},
        "Nombres et calcul : connaître les nombres et les utiliser dans les calculs": {"6e": True, "4e":True, "2nde": True},
    },
    "Espace et mesures": {
        "Espaces et géométrie : connaître et utiliser des notions de géométrie": {"6e": True, "4e":True, "2nde": True},
        "Grandeurs et mesures : Connaître des grandeurs et utiliser des mesures": {"6e": False, "4e":False, "2nde": True},
    },
}




#### FOCNTIONNE ###
# # Fonction pour calculer la moyenne par compétence principale
# def calculer_moyenne_par_competence_principale(dataframes, competences_par_niveau):

#     niveaux = ["cp", "ce1", "ce2", "cm1", "cm2",'6e','4e','2nde']
#     moyenne_globale = {"Niveau": [], "Compétence": [], "Moyenne": []}

#     for competence_generale, sous_competences in competences_par_niveau.items():
#         for niveau in niveaux:
#             if niveau in dataframes:
#                 df = dataframes[niveau]
#                 # Sélection des colonnes correspondant aux sous-compétences
#                 cols = [col for col in sous_competences if col in df.columns and sous_competences[col][niveau]]
#                 if cols:
#                     moyenne = df[cols].mean().mean()
#                     st.write(f"📌 Moyenne calculée pour {niveau.upper()} - {competence_generale} : {moyenne}")# Moyenne des sous-compétences disponibles
#                     if not np.isnan(moyenne):
#                         moyenne_globale["Niveau"].append(niveau.upper())
#                         moyenne_globale["Compétence"].append(competence_generale)
#                         moyenne_globale["Moyenne"].append(moyenne)
#                     else:
#                         st.write(f"⚠️ La moyenne est NaN pour {niveau.upper()} - {competence_generale}, il n'y a peut-être pas de valeurs numériques.")
#             else:
#                 st.write(f"⚠️ Attention : Données absentes pour le niveau **{niveau.upper()}**")

#     df_result = pd.DataFrame(moyenne_globale)
#     if df_result.empty:
#         st.write("⚠️ **Aucune donnée calculée, vérifiez vos fichiers sources !**")


#     # ✅ Assurer que les niveaux sont bien ordonnés
#     # Appliquer l'ordre des niveaux
#     niveau_ordre = ["CP", "CE1", "CE2", "CM1", "CM2"]
#     df_result["Niveau"] = pd.Categorical(df_result["Niveau"], categories=niveau_ordre, ordered=True)
#     df_result.sort_values("Niveau", inplace=True)

#     return df_result

# Fonction pour calculer la moyenne par compétence principale
def calculer_moyenne_par_competence_principale(dataframes, competences_par_niveau, niveaux):
    """
    Calcule la moyenne pour chaque compétence principale en parcourant les niveaux.
    """
    moyenne_globale = {"Niveau": [], "Compétence": [], "Moyenne": []}

    for competence_generale, sous_competences in competences_par_niveau.items():
        for niveau in niveaux:
            if niveau in dataframes:
                df = dataframes[niveau]

                # Sélection des colonnes correspondant aux sous-compétences
                cols = [col for col in sous_competences if col in df.columns and sous_competences[col].get(niveau, False)]

                if cols:
                    moyenne = df[cols].mean().mean()  # Moyenne de toutes les sous-compétences disponibles

                    if not np.isnan(moyenne):
                        moyenne_globale["Niveau"].append(niveau.upper())  # ✅ Correction de .upper
                        moyenne_globale["Compétence"].append(competence_generale)
                        moyenne_globale["Moyenne"].append(moyenne)



    df_result = pd.DataFrame(moyenne_globale)

    if df_result.empty:
        st.write("⚠️ **Aucune donnée calculée, vérifiez vos fichiers sources !**")
        return df_result

    # ✅ Assurer que les niveaux sont bien ordonnés (primaire et secondaire)
    ordre_niveaux = ["CP", "CE1", "CE2", "CM1", "CM2", "6E", "4E", "2NDE"]
    df_result["Niveau"] = pd.Categorical(df_result["Niveau"], categories=ordre_niveaux, ordered=True)
    df_result.sort_values("Niveau", inplace=True)

    return df_result


# Fonction pour afficher un seul graphique avec toutes les compétences
def creer_graphique_evolution_global(df_moyenne_globale):
    fig = px.line(
        df_moyenne_globale,
        x="Niveau",
        y="Moyenne",
        color="Compétence",
        markers=True,
    )
    fig.update_layout(
        legend_title_text="",
        legend=dict(
            orientation="h",  # Affichage horizontal
            yanchor="top",
            y=-0.3,  # Position sous le graphique
            xanchor="center",
            x=0.5  # Centre la légende horizontalement
        ),
        xaxis_title=None  # Supprime complètement l'axe X
    )

    return fig



# # Exécution des calculs et affichage des graphiques
# df_moyenne_globale_fr_primaire = calculer_moyenne_par_competence_principale(dataframes, competences_fr_primaire)
# df_moyenne_globale_maths_primaire = calculer_moyenne_par_competence_principale(dataframes, competences_maths_primaire)
# df_moyenne_globale_maths_secondaire = calculer_moyenne_par_competence_principale(dataframes, competences_maths_secondaire)
# df_moyenne_globale_fr_secondaire = calculer_moyenne_par_competence_principale(dataframes, competences_fr_secondaire)

niveaux_primaire = ["cp", "ce1", "ce2", "cm1", "cm2"]
niveaux_secondaire = ["6e", "4e", "2nde"]

# Calcul des moyennes pour le PRIMAIRE
df_moyenne_globale_fr_primaire = calculer_moyenne_par_competence_principale(dataframes, competences_fr_primaire, niveaux_primaire)
df_moyenne_globale_maths_primaire = calculer_moyenne_par_competence_principale(dataframes, competences_maths_primaire, niveaux_primaire)

# Calcul des moyennes pour le SECONDAIRE
df_moyenne_globale_fr_secondaire = calculer_moyenne_par_competence_principale(dataframes, competences_fr_secondaire, niveaux_secondaire)
df_moyenne_globale_maths_secondaire = calculer_moyenne_par_competence_principale(dataframes, competences_maths_secondaire, niveaux_secondaire)




# Trier correctement selon l'ordre pédagogique
df_moyenne_globale_maths_primaire = df_moyenne_globale_maths_primaire.sort_values("Niveau")


tab1, tab2= st.tabs(['**RESULTATS RÉSEAU**','RESULTATS PAR ÉTABLISSEMENT'])

with tab1:

    a, b = st.columns(2)
    a.metric(label="Moyenne en mathématiques", value=f"{moyenne_maths:.2f}%",border=True)
    b.metric(label="Moyenne en Français", value=f"{moyenne_francais:.2f}%",border=True)


    st.markdown('**Primaire / Secondaire : moyenne et répartition géographique des résultats**')
    col1, col2=st.columns([1,2])

    with col1:
        # Génération et Affichagedu graphique
        st.plotly_chart(creer_bar_chart_maths_francais(moyenne_maths_primaire,moyenne_francais_primaire,moyenne_maths_secondaire,moyenne_francais_secondaire))

    with col2 :
        tab1, tab2= st.tabs(["Primaire", 'Secondaire'])
        etablissements=jitter_coordinates(etablissements,jitter=0.001)
        tab1.plotly_chart(carte_etablissements(etablissements, 'Primaire', titre='Primaire'))
        tab2.plotly_chart(carte_etablissements(etablissements, 'Secondaire', titre='Secondaire'))


    # st.plotly_chart(creer_scatter_maths_francais(dataframes))
        # _,col1,_=st.columns(3)


        # with col1:
        #     with st.popover('Interpretation'):
        #         st.markdown("""
        #         Ce graphique représente la relation entre la moyenne en mathématiques et la moyenne en français (%) pour tous les établissements du réseau, chaque point correspondant à un établissement.

        #         - La ligne de tendance suggère une corrélation positive entre les performances en mathématiques et en français : les élèves obtenant de bons résultats en maths ont tendance à réussir également en français.
        #         - La taille des bulles indique l'écart entre les deux moyennes : une grande bulle signifie une différence marquée entre les notes en mathématiques et en français, tandis qu'une petite bulle indique un équilibre entre les deux matières.
        #         """)

st.markdown("**Dispersion par niveaux et corrélation maths/français**")
col1,col2=st.columns(2)

with col1:
    st.plotly_chart(creer_boxplot_combine(dataframes))

    # st.plotly_chart(evolution_moyenne_globale_par_niveau(dataframes, competences_matiere))

with col2 :

    st.plotly_chart(creer_scatter_maths_francais(dataframes))



# st.markdown("**Evolution global, par matiere et degré**")

col1, col2, col3=st.columns(3)

with col1 :
    st.plotly_chart(evolution_moyenne_globale_par_niveau(dataframes, competences_matiere))

with col2:
    tab1,tab2=st.tabs(['Primaire','Secondaire'])
    tab1.plotly_chart(creer_graphique_evolution_global(df_moyenne_globale_fr_primaire))
    tab2.plotly_chart(creer_graphique_evolution_global(df_moyenne_globale_fr_secondaire))

with col3:
    tab1,tab2=st.tabs(['Primaire','Secondaire'])
    tab1.plotly_chart(creer_graphique_evolution_global(df_moyenne_globale_maths_primaire))
    tab2.plotly_chart(creer_graphique_evolution_global(df_moyenne_globale_maths_secondaire))


with tab2:
    print ('hello')
