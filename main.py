# codigo de examen para subir
# Este sistema funciona correctamente
# Dependencias que se usan
import streamlit as st  # aplicaciones web interactivas
from transformers import pipeline  # pipeline crear un objeto que permite clasificar

st.title("Sistema de Clasificación de Oraciones")

# Carga el modelo preentrenado de Zero-shot
# el modelo realiza tareas de clasificación en varios idiomas.
model_name = "facebook/bart-large-mnli"

# llamamos al pipeline y lo guardamos en classifier llamando a nuestro modelo.
classifier = pipeline("zero-shot-classification", model=model_name)

# Entrada de texto por teclado
# Dos campos de entrada
input_text = st.text_input("Ingresa una oración:")
input_text2 = st.text_input("Ingresa otra oración:")

# Le damos la condicion para cada input con las etiquetas que necesitemos para la clasificación
if input_text:
    # Clasificar la oración ingresada
    labels = ["Deporte", "Cultura", "Política", "Religión", "Videojuegos"]
    result = classifier(input_text, labels)

    # Titulo para nuestra salida
    st.subheader("Clasificación de la Oración:")
    # Impresion de las etiquetas validas
    st.write(f"Etiquetas: {labels}")
    # Imprimimos la salida del input con nuestra oracion
    st.write(f"Oración ingresada: {input_text}")
    # Imprimimos la etiqueta que define el modelo para nuestra oración
    st.write(f"Pertenece a: {result['labels'][0]}")

if input_text2:
    # Clasificar la segunda oración ingresada
    result2 = classifier(input_text2, labels)

    # Imprimimos la salida del input con nuestra oracion
    st.write(f"Oración ingresada: {input_text2}")
    # Imprimimos la etiqueta que define el modelo para nuestra oracion
    st.write(f"Pertenece a: {result2['labels'][0]}")

