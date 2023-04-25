from django.shortcuts import render, HttpResponse, redirect
from django.db import IntegrityError


import feedparser
from datetime import datetime

from .models import News
import spacy
import nltk
from nltk.corpus import wordnet as wn
from collections import defaultdict

# modelo de procesamiento de texto 
nlp = spacy.load("es_core_news_md")

def index(request):
    
    return render(request, 'index.html', {
        "articles": News.objects.all()
    })
    

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
        'http://www.bbc.co.uk/mundo/temas/aprenda_ingles/index.xml'
    ]

    i=1
    for url in rss_urls:
        feed = feedparser.parse(url)
        for entry in feed.entries:
            try:
                new = News.objects.create(
                    title = entry.title,
                    description = entry.summary,
                    url =entry.links[0].href,
                    public_date = datetime.strptime(entry.published, '%a, %d %b %Y %H:%M:%S %z'),
                    image = entry.links[1].href if len(entry.links)>1 else ""
                )
                new.save()
            except IntegrityError:
                    pass
           
            i+=1
        
    
    return HttpResponse(f"recibidos {i} articulos ")

def valid_new(request):
     if request.method =="POST":
        
        texto = request.POST["text"]
        print(texto)
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

        
        
        return HttpResponse("validar noticias")



    

