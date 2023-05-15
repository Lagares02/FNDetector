from django.shortcuts import render, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.db import IntegrityError
from django.db.models import Q


import feedparser
from datetime import datetime

from .models import News

import spacy
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

            # deportes 
        'https://www.eltiempo.com/rss/deportes.xml ',
        'https://www.eltiempo.com/rss/deportes_futbol-internacional.xml ',
        'https://www.eltiempo.com/rss/deportes_futbol-colombiano.xml ',
        'https://www.eltiempo.com/rss/deportes_tenis.xml',
        'https://www.eltiempo.com/rss/deportes_ciclismo.xml',
        'https://www.eltiempo.com/rss/deportes_automovilismo.xml',
        'https://www.eltiempo.com/rss/deportes_otros-deportes.xml',

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

            # Video: 
        'http://www.bbc.co.uk/mundo/temas/video/index.xml',

            # Fotos: 
        'http://www.bbc.co.uk/mundo/temas/fotos/index.xml',

            # Aprenda Inglés: 
        'http://www.bbc.co.uk/mundo/temas/aprenda_ingles/index.xml',
        
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
        
            # Opinión
        'http://portafolio.co/rss/opinion',
        
            # Editorial
        'http://portafolio.co/rss/opinion/editorial',
        
            # Tendencias
        'http://portafolio.co/rss/tendencias',
        
            # Entretenimiento
        'http://portafolio.co/rss/tendencias/entretenimiento',
        
            # Sociales
        'http://portafolio.co/rss/tendencias/sociales',
        
            # Lujo
        'http://portafolio.co/rss/tendencias/lujo'
    ]

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
            except (ValueError, TypeError):
                pass
            else:
                try:
                    new = News.objects.create(
                        title=title,
                        description=description,
                        url=link,
                        public_date=public_date,
                        image=image
                    )
                    i += 1
                except IntegrityError:
                    pass

    return HttpResponse(f"Recibidos {i} artículos.")

@csrf_exempt
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

        # Extracción de palabras clave (sustantivos y adjetivos)
        palabras_clave = [token for token in doc if token.pos_ in ['NOUN', 'ADJ']]

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
                noticias_similares = News.objects.filter(Q(description__icontains=palabra) | Q(description__icontains=sinonimo), status=False, verificado=False)
                for noticia in noticias_similares:
                    # Calcular la similitud de Jaccard
                    noticia_tokens = set(word_tokenize(noticia.description))
                    similitud = 100 * (1 - jaccard_distance(set(palabras_clave), noticia_tokens))
                    
                    # Agregar la noticia al diccionario si no existe o actualizar la similitud si ya existe
                    if noticia.url not in similares:
                        similares[noticia.url] = {'title': noticia.title, 'url': noticia.url, 'similarity': similitud}
                    else:
                        similares[noticia.url]['similarity'] = max(similitud, similares[noticia.url]['similarity'])
                    
                    # Marcar la noticia como verificada
                    noticia.status = True
                    noticia.verificado = True 
                    noticia.save()

        for entidad in doc.ents:
            noticias_similares = News.objects.filter(title__icontains=entidad.text, status=False, verificado=False)
            for noticia in noticias_similares:
                # Calcular la similitud de Jaccard
                noticia_tokens = set(word_tokenize(noticia.description))
                similitud = 100 * (1 - jaccard_distance(set([entidad.text]), noticia_tokens))
                    
                # Agregar la noticia al diccionario si no existe o actualizar la similitud si ya existe
                if noticia.url not in similares:
                    similares[noticia.url] = {'title': noticia.title, 'url': noticia.url, 'similarity': similitud}
                else:
                    similares[noticia.url]['similarity'] = max(similitud, similares[noticia.url]['similarity'])
                    
                # Marcar la noticia como verificada
                noticia.status = True
                noticia.verificado = True 
                noticia.save()
                
        similares = sorted(similares.values(), key=lambda k: k['similarity'], reverse=True)[:10]
        
        if 'similares' not in request.session:
            request.session['similares'] = []
        request.session['similares'] = similares

        print(similares)

        return render(request, 'verificar.html', {'result': result, 'similares': similares})

