# LIBRERÍAS #

import streamlit as st
import pandas as pd
import json
import re
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from scipy.spatial.distance import cdist
from rapidfuzz import process
import base64
from pathlib import Path


# BASES DE DATOS #

# Rutas de los archivos locales
MATRIX_FILE_PATH = r"Ingredients Matrix.xlsx"
STANDARDIZATION_FILE_PATH = r"Diccionario de Reemplazo.json"

# Cargar la matriz de ingredientes y el diccionario de reemplazo
try:
    ingredient_matrix = pd.read_excel(MATRIX_FILE_PATH)
    # st.success("Matriz de ingredientes cargada correctamente desde el archivo local!")
except Exception as e:
    st.error(f"Error al cargar la matriz de ingredientes: {e}")

try:
    with open(STANDARDIZATION_FILE_PATH, "r", encoding="utf-8") as f:
        ingredient_standardization = json.load(f)
    # st.success("Diccionario de estandarización cargado correctamente desde el archivo local!")
except Exception as e:
    st.error(f"Error al cargar el diccionario de estandarización: {e}")


# FUNCIONES #

# Función para procesar ingredientes
def process_ingredients(input_ingredients, replace_dict, data):

    # 1º Función: estandarización y limpieza de la lista de entrada
    def standardize_ingredients(ingredients):
        if isinstance(ingredients, str):
            standardized = []
            temp_ingredient = []
            parenthesis_count = 0
            i = 0

            while i < len(ingredients):
                char = ingredients[i]

                if char == '(':
                    parenthesis_count += 1
                elif char == ')':
                    parenthesis_count -= 1

                if char == ',' and parenthesis_count == 0:
                    if not (i > 0 and i < len(ingredients) - 2 and
                            ingredients[i - 1].isdigit() and 
                            ingredients[i + 1] == ' ' and 
                            ingredients[i + 2].isdigit() and 
                            (ingredients[i - 1], ingredients[i + 2]) == ('1', '2')):
                        standardized.append(''.join(temp_ingredient).strip())
                        temp_ingredient = []
                        i += 1
                        continue

                temp_ingredient.append(char)
                i += 1

            if temp_ingredient:
                standardized.append(''.join(temp_ingredient).strip())

            for i in range(len(standardized)):
                ingredient = standardized[i].lower().replace('*', '').replace('.', '')
                ingredient = re.sub(r'\s{2,}', ' ', ingredient)
                ingredient = re.sub(r'(?<=\S)\(', ' (', ingredient)
                ingredient = re.sub(r'\)(?=\S)', ') ', ingredient)
                standardized[i] = ingredient

            return standardized
        else:
            return "Error: Por favor, introduce una lista válida de ingredientes"

    # 2º función: Reemplazar ingredientes según diccionario de reemplazo
    def replace_ingredients(ingredients_list):
        return [replace_dict.get(ingredient, ingredient) for ingredient in ingredients_list]
        
    # Ejecución de funciones anidadas
    standardized_ingredients = standardize_ingredients(input_ingredients)
    if isinstance(standardized_ingredients, str):
        return standardized_ingredients
    
    replaced_ingredients = replace_ingredients(standardized_ingredients)
    
    return replaced_ingredients


# Función de análisis de la lista de ingredientes dada: ingredientes naturales, no-naturales y propiedades presentes

def list_analisis(ingredients_list, data):
    # Se guardan todos los ingredientes naturales y artificiales en diccionarios
    natural_data = data[data['Natural'] == 1].set_index('Ingredients')['Natural'].to_dict()
    non_natural_data = data[data['Natural'] == 0].set_index('Ingredients')['Natural'].to_dict()
    ingredient_data = {**natural_data, **non_natural_data}  # Merge a los diccionarios

    # Separamos los ingredientes de la lista dada entre naturales y artificiales
    natural_ingredients = []
    artificial_ingredients = []
    for ingredient in ingredients_list:
        if ingredient in ingredient_data:
            if ingredient_data[ingredient] == 1:
                natural_ingredients.append(ingredient)
            else:
                artificial_ingredients.append(ingredient)

    # Recopilamos las propiedades sin repetir de los ingredientes de la lista dada
    ingredients_list_data = data[data['Ingredients'].isin(ingredients_list)]
    if not ingredients_list_data.empty:  # Check if DataFrame is empty
        mask = (ingredients_list_data.iloc[:, 3:-3] >= 1).any(axis=0)
        true_properties = ingredients_list_data.columns[3:-3][mask].tolist()
    
    return natural_ingredients, artificial_ingredients, true_properties



# Funciones para el sistema de recomendación de ingredientes

# 1º Función: Obtener los vectores de las propiedades de cada uno de los ingredientes artificiales dados
def get_property_vectors(artificial_ingredients, matrix):
    property_vectors = {}
    for ingredient in artificial_ingredients:
        properties = matrix.loc[matrix['Ingredients'] == ingredient, matrix.columns[2:-3]].values[0]
        roles = matrix.loc[matrix['Ingredients'] == ingredient, matrix.columns[-3:]].values[0]
        property_vector = list(properties) + list(roles)
        property_vectors[ingredient] = property_vector
    return property_vectors


# 2º Función: mediante una métrica de distancia (ej: coseno), obtener los valores de similitud entre los artificiales y los naturales de la base
## Se queda con los top 10 en puntaje de similitud y crea un diccionario con estos reemplazos para cada ingrediente artificial
def get_replacements(property_vectors, matrix):
    replacements = {}
    for ingredient, vector in property_vectors.items():
        natural_rows = matrix[matrix['Natural'] == 1]
        natural_vectors = natural_rows[matrix.columns[2:]].values
        distances = cdist([vector], natural_vectors, metric='cosine')[0]
        top_indices = distances.argsort()[:10]
        replacements[ingredient] = list(natural_rows.iloc[top_indices]['Ingredients'])
    return replacements

# 3º Función: Elabora las recomendaciones de ingredientes a partir de los resultados de la 2º Función. 
## Ingresa en el diccionario elaborado en la 2º Función, copia la fórmula original, y para cada caso del 1 al 10, inserta el reemplazo.
def create_recommendations(ingredients, replacements):
    recommendations = []
    # Deberíamos generar 10 recomendaciones (para cada uno de los posibles reemplazos)
    for idx in range(10):
        recommendation = ingredients.copy()
        for i, ingredient in enumerate(recommendation):
            if ingredient in replacements:
                # Reemplazamos con el idx-ésimo reemplazo para cada ingrediente artificial
                recommendation[i] = replacements[ingredient][idx]
        recommendations.append(recommendation)
    return recommendations



# INTERFAZ DE USUARIO #

# Initialize all session state variables
if "final_ingredients" not in st.session_state:
    st.session_state.final_ingredients = []
if "processing_complete" not in st.session_state:
    st.session_state.processing_complete = False
if "selections_made" not in st.session_state:
    st.session_state.selections_made = []

# Función para encontrar ingredientes similares
def find_similar_ingredients(ingredient, ingredient_list, threshold=80):
    matches = process.extract(ingredient, ingredient_list, limit=3)
    return [match for match, score, _ in matches if score >= threshold]

# Función para seleccionar ingredientes sin reiniciar toda la app usando la fragmentación
# Crear un fragmento para cada selectbox
@st.fragment
def ingredient_selector(ingredient, i, suggestions):
    selection_key = f"selection_{i}_{ingredient}"
    
    # Inicializar la clave en session_state si no existe
    if selection_key not in st.session_state:
        st.session_state[selection_key] = None
    
    # Definir callback para manejar la selección
    def update_selection():
        selected = st.session_state[selection_key]
        if selected is not None and selected != "Ninguna de las anteriores":
            if selected not in st.session_state.final_ingredients:
                st.session_state.final_ingredients.append(selected)
    
    # Selección del usuario con callback
    st.selectbox(
        f"Selecciona una alternativa para '{ingredient}'",
        suggestions + ["Ninguna de las anteriores"],
        index=None,
        placeholder="Selecciona una opción...",
        key=selection_key,
        on_change=update_selection
    )

# LOGO Y TÍTULOS DE LA APP

# Set wide layout
st.set_page_config(layout="wide")

# Function to load and encode the image
def get_base64_encoded_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Create two columns for logo and title
logo_col, title_col = st.columns([1, 4])  # Adjust ratio as needed

# Add logo to the first column
with logo_col:
    try:
        # Option 1: Using direct image display
        st.image("Logo.jpg", width=150)  # Adjust width as needed
        
        # Option 2: If you prefer the base64 approach
        # img_base64 = get_base64_encoded_image("Logo.jpg")
        # st.markdown(f'<img src="data:image/jpeg;base64,{img_base64}" width="150">', unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error loading logo: {e}")

# Add title to the second column
with title_col:
    st.title("FrancesLab")
    st.write("¡El hogar de nuestro algoritmo FRANCES!")

st.header("Sistema de Recomendación de Ingredientes Cosméticos")


# Botón para generar recomendaciones, inicio del proceso
if st.session_state.processing_complete is False:
    
    # Input de la lista de ingredientes
    st.write("Ingresa una lista de ingredientes para obtener recomendaciones.")
    ingredients_list = st.text_area("Lista de Ingredientes", placeholder="Ejemplo: water, sodium hydroxide, fragrance")
    
    if st.button("Generar Recomendaciones"):
        # Preprocesamiento de los ingredientes
        clean_ingredients = process_ingredients(ingredients_list, ingredient_standardization, ingredient_matrix)
        if not clean_ingredients:
            st.stop()
    
        ## Revisamos si todos los ingredientes se encuentran en la base de datos
        # Lista de los ingredientes de la Ingredients Matrix
        matrix_ingredients = ingredient_matrix['Ingredients'].str.lower().tolist()
        # Lista de los ingredientes definitivos para el análisis
        unidentified = []
        # Agregamos a los ingredientes finales solo los que sí se encuentran dentro de la base
        for ingredient in clean_ingredients:
            if ingredient in matrix_ingredients:
                st.session_state.final_ingredients.append(ingredient)
            else:
                unidentified.append(ingredient)
    
        
        # Selección de ingredientes sugeridos para unidentified
        if unidentified:
            # Inicializar una variable para rastrear si todos los ingredientes tienen selecciones
            if "selections_made" not in st.session_state:
                st.session_state.selections_made = [False] * len(unidentified)
            
            # Mostrar todos los selectboxes
            for i, ingredient in enumerate(unidentified):
                suggestions = find_similar_ingredients(ingredient, matrix_ingredients)
                if suggestions:
                    ingredient_selector(ingredient, i, suggestions)
                    # Verificar si se ha hecho una selección para este ingrediente
                    selection_key = f"selection_{i}_{ingredient}"
                    if selection_key in st.session_state and st.session_state[selection_key] is not None:
                        st.session_state.selections_made[i] = True
                else:
                    st.write(f"No se encontró una coincidencia para el ingrediente '{ingredient}', por favor revisa su nombre o elimínalo de la lista ingresada de ingredientes y reinténtalo")
                    st.stop()
            
            # Botón de confirmación que solo procede si todas las selecciones están hechas
            def confirm_selections():
                if all(st.session_state.selections_made):
                    st.session_state.processing_complete = True
            
            # Luego en tu código:
            if st.button("Confirmar Selecciones", on_click=confirm_selections):
                if not all(st.session_state.selections_made):
                    st.warning("Por favor, selecciona una opción para cada ingrediente no identificado.")
            
        else:
            # Mark processing as complete to avoid rerunning this section
            st.session_state.processing_complete = True
            st.rerun()  # Force a clean rerun with the new state
           
# Display results after processing is complete
if st.session_state.processing_complete is True:
    final_ingredients = st.session_state.final_ingredients
    if not final_ingredients:
        st.error("No se han procesado ingredientes válidos.")
        st.stop()
    st.success(f"Ingredientes Procesados: {final_ingredients}")   
    
    # Análisis de los ingredientes    
    natural_ingredients, artificial_ingredients, true_properties = list_analisis(final_ingredients, ingredient_matrix)
    st.write("### Análisis de los ingredientes de la lista:\n")
    st.write("##### Naturales:")
    st.write(f"{', '.join(natural_ingredients)}\n")
    st.write("##### Artificiales:")
    st.write(f"{', '.join(artificial_ingredients)}")
    st.write(f"##### Propiedades únicas de los ingredientes:")
    st.write(f"{true_properties}")

    # Recomendaciones de ingredientes
    # Ejecutar el programa
    property_vectors = get_property_vectors(artificial_ingredients, ingredient_matrix)
    replacements = get_replacements(property_vectors, ingredient_matrix)
    recommendations = create_recommendations(final_ingredients, replacements)

    # Imprimir los resultados
    st.write("### Sistema de Recomendación de Ingredientes\n")
    st.write("#### Reemplazos naturales:")
    for ingredient, replacements in replacements.items():
        st.write(f"\n*'{ingredient}'* -> {', '.join(replacements)}")
    st.write("#### Top 10 recomendaciones:")
    for i, recommendation in enumerate(recommendations, 1):
        st.write(f"{i}. {recommendation}\n")

    def reset_session():
        st.session_state.final_ingredients = []
        st.session_state.processing_complete = False
        st.session_state.selections = []
    
    if st.button("Nueva búsqueda"):
        reset_session()
        st.rerun()
