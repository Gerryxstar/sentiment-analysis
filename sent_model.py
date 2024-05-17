from flask import Flask, request, jsonify
import joblib
from nltk.tokenize import word_tokenize
import nltk

# Descargar el tokenizador Punkt si aún no está instalado
nltk.download('punkt')

app = Flask(__name__)

# Cargar el modelo persistente
modelo_sentimientos = joblib.load('C:\\Users\\cpfso\\OneDrive\\Documents\\Gerardo Solares\\Data Science\\U-Camp\\Modulo 7 - Tecnicas avanzadas y empleabilidad\\Proyecto modulo 7\\Play store\\modelo_sentimientos.pkl')
vectorizador_sentimientos = joblib.load('C:\\Users\\cpfso\\OneDrive\\Documents\\Gerardo Solares\\Data Science\\U-Camp\\Modulo 7 - Tecnicas avanzadas y empleabilidad\\Proyecto modulo 7\\Play store\\vectorizador_sentimientos.pkl')

@app.route('/analizar_sentimiento', methods=['POST']) # metodo post, se usa porque es el metodo mas seguro que otros, ejemplo: GET.
def analizar_sentimiento():
    datos = request.json['Review'] # en esta parte viene implicacitamente que se solicita una cadena de texto para poder dar respuesta a la solicitud.
    resultados = []
    
    for dato in datos:
        if isinstance(dato, list):  # Verificar si los datos son una lista de cadenas
            # Unir todas las listas de tokens si es una lista de cadenas
            tokens = ' '.join(datos)
        else:
        # Si los datos no son una lista, asumimos que es una cadena y tokenizamos directamente
            tokens = dato

        # Tokenizar el texto
        tokens_tokenizados = word_tokenize(tokens)

        # Transformar los datos utilizando el vectorizador
        datos_transformados = vectorizador_sentimientos.transform([' '.join(tokens_tokenizados)])

        # Realizar la predicción utilizando el modelo de sentimientos
        prediccion_sentimientos = modelo_sentimientos.predict(datos_transformados)

        # Mapear la predicción numérica a etiquetas de texto (para que arroje una etiqueta de texto)
        label_mapping = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
        resultado = label_mapping[prediccion_sentimientos[0]]

        resultados.append(resultado)

    return jsonify({'sentimiento': resultados})

if __name__ == '__main__':
    app.run(debug=True)

    #  la API puede teóricamente manejar cualquier cantidad de comentarios, pero es importante considerar el impacto en el rendimiento y establecer límites adecuados según las necesidades y capacidades de tu aplicación y servidor.