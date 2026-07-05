from setfit import AbsaModel

# 1. Load the fully fine-tuned SetFit ABSA model
# ABSA requires TWO models to be loaded: 
#   1. The 'aspect' model (for finding the aspects/spans)
#   2. The 'polarity' model (for determining the sentiment of those aspects)
model = AbsaModel.from_pretrained(
    "tomaarsen/setfit-absa-paraphrase-mpnet-base-v2-restaurants-aspect",
    "tomaarsen/setfit-absa-paraphrase-mpnet-base-v2-restaurants-polarity"
)

# 2. Define the list of sentences you want to analyze
sentences_to_analyze = [
    "The food was exceptional, although the service was a bit slow."
]

# 3. Run prediction
preds = model.predict(sentences_to_analyze)

# 4. Print the results
print(preds)