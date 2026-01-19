import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

resources = [
    'averaged_perceptron_tagger_eng',
    'cmudict',
    'punkt'  # Often needed too
]

print("Downloading NLTK resources...")
for res in resources:
    try:
        nltk.download(res)
        print(f"Downloaded {res}")
    except Exception as e:
        print(f"Failed to download {res}: {e}")
