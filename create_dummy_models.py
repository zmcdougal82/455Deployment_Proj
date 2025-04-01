import pickle

class DummyModel:
    def __init__(self):
        pass

model = DummyModel()

# Save collaborative filtering model
with open('models/collaborative_model.sav', 'wb') as f:
    pickle.dump(model, f)

# Save content filtering model
with open('models/content_model.sav', 'wb') as f:
    pickle.dump(model, f)

print("Dummy model files created successfully!")
