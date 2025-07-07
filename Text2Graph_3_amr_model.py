import amrlib
import penman

# Make sure the model is already extracted here
custom_model_path = "Models/model_parse_xfm_bart_large"
print("Loading model from:", custom_model_path)

# This uses amrlib's internal loader
stog = amrlib.load_stog_model(custom_model_path)

# Input example
sentence = "The doctor ordered a new PET scan for the patient."
graphs = stog.parse_sentences([sentence])

# Print AMR string
print("\n--- AMR Representation ---")
print(graphs[0])

# Parse and print triples
g = penman.decode(graphs[0])
print("\n--- Triples ---")
for triple in g.triples:
    print(triple)
