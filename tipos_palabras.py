import spacy
from collections import defaultdict


nlp = spacy.load("es_core_news_md")

# Texto de ejemplo muy amplio (puedes ampliarlo o cargar libros, artículos, etc.)
# Aquí usamos varios tipos de palabras para forzar variedad gramatical
textos = [
    "El perro corre rápidamente hacia el parque.",
    "¿Cómo estás? ¡Bienvenidos a la fiesta!",
    "Los números primos son infinitos.",
    "Juan y María están comiendo en el restaurante.",
    "¡Qué sorpresa tan grande!",
    "Este es un ejemplo de oraciones con verbos, sustantivos y adjetivos.",
    "100 personas participaron en la maratón.",
    "Ella ha estado leyendo libros interesantes.",
    "Score y news son palabras extranjeras.",
    "No me lo digas, por favor.",
    "¿Quién fue el primero en llegar?",
    "Madrid está en España.",
]

# Set para almacenar etiquetas únicas
pos_tags = set()
tag_tags = set()
tag_examples = defaultdict(set)

# Procesar todos los textos
for texto in textos:
    doc = nlp(texto)
    for token in doc:
        pos_tags.add(token.pos_)
        tag_tags.add(token.tag_)
        tag_examples[token.tag_].add((token.text, token.pos_))

# Mostrar resultados
print("POS tags universales (token.pos_):")
for pos in sorted(pos_tags):
    print(f"  {pos}")

print("\nEtiquetas gramaticales específicas (token.tag_):")
for tag in sorted(tag_tags):
    ejemplos = ", ".join([t[0] for t in list(tag_examples[tag])[:3]])  # primeros 3 ejemplos
    print(f"  {tag}  → Ejemplos: {ejemplos}")
