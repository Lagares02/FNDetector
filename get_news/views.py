from django.shortcuts import render, HttpResponse
from django.db import IntegrityError
from django.db.models import Q


import feedparser
from datetime import datetime

from .models import News

import spacy
import json
import nltk
#nltk.download('wordnet')
#nltk.download('omw-1.4')
#nltk.download('punkt')
from nltk.corpus import wordnet as wn
from nltk.metrics import jaccard_distance
from nltk.tokenize import word_tokenize
from collections import defaultdict
from .identifyModel.SVM import predict
from django.views.decorators.csrf import csrf_exempt


def index(request): 
    return render(request, 'index.html')

def news(request):
    # URLs de los feeds RSS
    rss_urls = [
        
        # feeds nacionales 
        # el tiempo 
            # opinion
        'https://www.eltiempo.com/rss/opinion.xml',
        'https://www.eltiempo.com/rss/opinion_editorial.xml',
        'https://www.eltiempo.com/rss/opinion_mas-opinion.xml',
        'https://www.eltiempo.com/contenido/feed',

            # regiones
        'https://www.eltiempo.com/rss/colombia.xml',
        'https://www.eltiempo.com/rss/colombia_barranquilla.xml',
        'https://www.eltiempo.com/rss/colombia_medellin.xml',
        'https://www.eltiempo.com/rss/colombia_cali.xml',
        'https://www.eltiempo.com/rss/colombia_otras-ciudades.xml',

            # mundo
        'https://www.eltiempo.com/rss/mundo.xml',
        'https://www.eltiempo.com/rss/mundo_latinoamerica.xml',
        'https://www.eltiempo.com/rss/mundo_eeuu-y-canada.xml',
        'https://www.eltiempo.com/rss/mundo_europa.xml',
        'https://www.eltiempo.com/rss/mundo_medio-oriente.xml',
        'https://www.eltiempo.com/rss/mundo_asia.xml',
        'https://www.eltiempo.com/rss/mundo_africa.xml',
        'https://www.eltiempo.com/rss/mundo_mas-regiones.xml',
          
            # economia 
        'https://www.eltiempo.com/rss/economia.xml',
        'https://www.eltiempo.com/rss/economia_finanzas-personales.xml ',
        'https://www.eltiempo.com/rss/economia_empresas.xml ',
        'https://www.eltiempo.com/rss/economia_sectores.xml',
        'https://www.eltiempo.com/rss/economia_sector-financiero.xml',

        # BBC
            # Últimas Noticias: 
        'http://www.bbc.co.uk/mundo/ultimas_noticias/index.xml',

            # Internacional: 
        'http://www.bbc.co.uk/mundo/temas/internacional/index.xml',

            # América Latina: 
        'http://www.bbc.co.uk/mundo/temas/america_latina/index.xml',

            # Ciencia: 
        'http://www.bbc.co.uk/mundo/temas/ciencia/index.xml',

            # Salud: 
        'http://www.bbc.co.uk/mundo/temas/salud/index.xml',

            # Tecnología: 
        'http://www.bbc.co.uk/mundo/temas/tecnologia/index.xml',

            # Economía: 
        'http://www.bbc.co.uk/mundo/temas/economia/index.xml',

            # Cultura: 
        'http://www.bbc.co.uk/mundo/temas/cultura/index.xml',
        
        # portafolio.co
            # Economia
        'http://portafolio.co/rss/economia',

            # Finanzas
        'http://portafolio.co/rss/economia/finanzas',
        
            # Gobierno
        'http://portafolio.co/rss/economia/gobierno',
        
            # Infraestructura
        'http://www.portafolio.co/rss/economia/infraestructura',
        
            # Empleo
        'http://portafolio.co/rss/economia/empleo',
        
            # Impuestos
        'http://portafolio.co/rss/economia/impuestos',
        
            # Negocios
        'http://portafolio.co/rss/negocios',
        
            # Empresas
        'http://portafolio.co/rss/negocios/empresas',
        
            # Emprendimiento
        'http://portafolio.co/rss/negocios/emprendimiento',
        
            # Inversión
        'http://portafolio.co/rss/negocios/inversion',
        
            # Internacional
        'http://www.portafolio.co/rss/internacional',
        
            # Innovación
        'http://portafolio.co/rss/innovacion',
        
            # Ahorro
        'http://portafolio.co/rss/mis-finanzas/ahorro',
        
            # Vivienda
        'http://portafolio.co/rss/mis-finanzas/vivienda',
        
            # Tendencias
        'http://portafolio.co/rss/tendencias',
        
    ]

    noticias = []
    i = 0

    for url in rss_urls:
        feed = feedparser.parse(url)
        for entry in feed.entries:
            title = entry.get('title', '')
            description = entry.get('description', '')
            link = entry.get('link', '')
            date_published = entry.get('published', '')
            image = entry.get('media_content', '') or entry.get('enclosure', '')
            if image:
                image = image[0].get('url', '')

            try:
                if not (title and description and link and date_published):
                    raise ValueError('Noticia incompleta')

                if '-' in date_published:
                    date_format = '%Y-%m-%dT%H:%M:%S%z'
                else:
                    date_format = '%a, %d %b %Y %H:%M:%S %z'
                public_date = datetime.strptime(date_published, date_format)

                noticia = {
                    "title": title,
                    "description": description,
                    "url": link,
                    "public_date": public_date.isoformat(),
                    "image": image,
                    "status": False,
                    "verificado": False
                }

                noticias.append(noticia)
                i += 1

            except (ValueError, TypeError):
                pass

    with open('db.json', 'w', encoding='utf-8') as json_file:
        json.dump(noticias, json_file, ensure_ascii=False)

    return HttpResponse(f"Recibidos {i} artículos.")


def valid_new(request):
     if request.method == "POST":
        # Carga del modelo en español
        nlp = spacy.load("es_core_news_sm")

        texto = request.POST["texto"]
        # Procesamiento del texto
        doc = nlp(texto)

        # Tokenización
        print("Tokenización:")
        for token in doc:
            print(token.text)

        # Análisis de entidades
        print("\nEntidades:")
        for entidad in doc.ents:
            print(entidad.text, "-", entidad.label_)
            
        conectores_ignorar = ["el", "la", "los", "las", "de", "del", "al", "a", "ante", "bajo", "cabe", "con", "contra", "desde", "en", "entre", "hacia", "hasta", "para", "por", "según", "sin", "sobre", "tras", "durante", "mediante", "versus", "vía", "excepto", "incluso", "más", "menos", "salvo", "donde", "cuando", "si", "pero", "aunque", "como", "porque", "pues", "ya", "o", "u", "siempre", "jamás", "nunca", "también", "además", "asimismo", "sin embargo", "no obstante", "por lo tanto", "entonces", "por ende", "por consiguiente", "así que", "así pues", "por eso", "por esta razón", "por esa razón", "por este motivo", "por ese motivo", "en conclusión"]

        # Extracción de palabras clave (sustantivos y adjetivos) y eliminación de conectores
        palabras_clave = [token for token in doc if token.pos_ in ['NOUN', 'ADJ'] and token.text.lower() not in conectores_ignorar]

        # Búsqueda de sinónimos
        sinonimos = defaultdict(list)
        for palabra in palabras_clave:
            synsets = wn.synsets(palabra.text, lang='spa')
            for synset in synsets:
                for lemma in synset.lemmas(lang='spa'):
                    sinonimo = lemma.name()
                    if sinonimo != palabra.text and sinonimo not in sinonimos[palabra.text]:
                        sinonimos[palabra.text].append(sinonimo)

        # Resultados
        print("Palabras clave y sinónimos:")
        for palabra, sinonimos_palabra in sinonimos.items():
            print(f"{palabra}: {', '.join(sinonimos_palabra)}")
  
        # Resultado de clasificacion
        result = predict(texto)
        print(result)
        
        # Agregar el titular a la sesión del usuario
        if 'titulares' not in request.session:
            request.session['titulares'] = []
        request.session['titulares'].append(texto)

        # Buscar noticias similares
        similares = {}  # Usar un diccionario en lugar de una lista o un conjunto

        for palabra, sinonimos_palabra in sinonimos.items():
            for sinonimo in sinonimos_palabra:
                with open('db.json', 'r', encoding='utf-8') as json_file:
                    noticias = json.load(json_file)
                    
                for noticia in noticias:
                    noticia_tokens = set(word_tokenize(noticia['title'].lower()))  # Tokenización del título de la noticia en minúsculas
                    
                    num_palabras_similares = sum(1 for token in doc if token.text.lower() in noticia_tokens)
                    similitud = (num_palabras_similares / len(noticia_tokens)) * 100  # Calcular similitud en función del número de palabras similares
                    
                    # Agregar la noticia al diccionario si no existe o actualizar la similitud si ya existe
                    if noticia['url'] not in similares:
                        similares[noticia['url']] = {'title': noticia['title'], 'url': noticia['url'], 'similarity': similitud}
                    else:
                        similares[noticia['url']]['similarity'] = max(similitud, similares[noticia['url']]['similarity'])

        # Búsqueda de noticias similares para las entidades reconocidas
        for entidad in doc.ents:
            entidad_tokens = set(word_tokenize(entidad.text.lower()))  # Tokenización de la entidad en minúsculas
            
            with open('db.json', 'r', encoding='utf-8') as json_file:
                noticias = json.load(json_file)
            
            for noticia in noticias:
                noticia_tokens = set(word_tokenize(noticia['title'].lower()))  # Tokenización del título de la noticia en minúsculas
                
                # Verificar si la entidad está presente en el título de la noticia
                if entidad.text.lower() in noticia['title'].lower():
                    num_palabras_similares = sum(1 for token in entidad_tokens if token in noticia_tokens)
                    similitud = (num_palabras_similares / len(noticia_tokens)) * 100  # Calcular similitud en función del número de palabras similares
                else:
                    similitud = 0  # Establecer similitud como cero si la entidad no está en el título
                
                # Agregar la noticia al diccionario si no existe o actualizar la similitud si ya existe
                if noticia['url'] not in similares:
                    similares[noticia['url']] = {'title': noticia['title'], 'url': noticia['url'], 'similarity': similitud}
                else:
                    similares[noticia['url']]['similarity'] = max(similitud, similares[noticia['url']]['similarity'])

        similares = sorted(similares.values(), key=lambda k: k['similarity'], reverse=True)[:10]

        if 'similares' not in request.session:
            request.session['similares'] = []
        request.session['similares'] = similares

        print(similares)

        return render(request, 'verificar.html', {'result': result, 'similares': similares})