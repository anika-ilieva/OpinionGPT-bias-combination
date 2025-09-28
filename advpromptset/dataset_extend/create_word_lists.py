"""""
This script enables the creation of expanded word lists related to
political ideology terms using semantic similarity.
It uses Sentence-BERT for embedding and cosine similarity for finding
semantically close terms.
"""""

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import wordnet as wn
import numpy as np
import nltk

nltk.download('wordnet')
nltk.download('omw-1.4')

# Round 1 of word list expansion
# Define original political terms
political_terms = [
    "liberal", "socialist", "communist", "fascist",
    "conservative", "political ideology"
]

# Round 2 of word list expansion
# Define manually filtered synonyms produced from Round 1
all_ideology_terms = [
    # Liberal
    "liberal", "liberalist", "liberalism", "liberalness", "liberality",
    "liberalise", "liberalistic", "liberalize", "liberalization",
    "liberally", "liberalisation", "neoliberal",

    # Socialist
    "socialist", "socialism", "socialistic",

    # Communist
    "communist", "communism", "communistic", "commie",

    # Fascist
    "fascist", "fascism", "fascistic", "fascista", "nazi",
    "nazism", "naziism", "mussolini", "gestapo", "fabianism",
    "nazify", "hitlerian",

    # Conservative
    "conservative", "conservatively", "conservativism", "conservatism",
    "conservativist", "republican", "neoconservative", "neoconservatism",
    "tory", "republicanism",

    # Political ideology (other)
    "authoritarianism", "libertarianism", "establishmentism", "sovietism",
    "maoist", "maoism", "authoritarian", "nationalist", "secularist",
    "marxist", "marxism", "capitalistic", "capitalism",
    "capitalist", "anarchistic", "stalinism", "mao", "stalinist",
    "totalitarianism"
]

# Extract unique WordNet lemmas
def get_wordnet_lemmas():
    words = set()
    for synset in wn.all_synsets():
        for lemma in synset.lemmas():
            word = lemma.name().replace("_", " ").lower()
            if word.isalpha() and 3 <= len(word) <= 20:
                words.add(word)
    return sorted(words)

# Load WordNet terms
print("Loading WordNet candidate terms...")
candidate_terms = get_wordnet_lemmas()
print(f"Total candidates: {len(candidate_terms)}")

# Load SBERT model
print("Loading SentenceTransformer...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Encode original and candidate terms
# Change to all_ideology_terms for Round 2
print("Encoding terms...")
term_embeddings = model.encode(political_terms, show_progress_bar=True)
candidate_embeddings = model.encode(candidate_terms, show_progress_bar=True)

# Perform KNN search
# Adjust k as needed
def get_top_k_synonyms(query_embedding, candidate_embeddings, candidate_terms, k=20):
    similarities = cosine_similarity([query_embedding], candidate_embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][:k]
    return [(candidate_terms[i], similarities[i]) for i in top_indices]

# Expand and save
# Change to all_ideology_terms for Round 2
expanded_terms = {}
for i, term in enumerate(political_terms):
    neighbors = get_top_k_synonyms(term_embeddings[i], candidate_embeddings, candidate_terms)
    expanded_terms[term] = neighbors

# Save as text file
output_file = "expanded_political_terms.txt"
with open(output_file, "w") as f:
    for term, neighbors in expanded_terms.items():
        f.write(f"\nTop synonyms for '{term}':\n")
        for synonym, score in neighbors:
            f.write(f"  {synonym} (cosine sim: {score:.4f})\n")
print(f"\n Results saved to: {output_file}")
