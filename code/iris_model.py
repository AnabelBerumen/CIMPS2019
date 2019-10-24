# Load json and create model:
from tensorflow.keras.models import model_from_json

def trained_model():
    json_file = open('../notebooks/iris_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    print("[INFO]Model architecture loaded from disk.")
    # Load weights into loaded model:
    loaded_model.load_weights("../notebooks/iris_model.h5")
    print("[INFO]Model weight loaded from disk.")
    
    loaded_model.compile(loss='categorical_crossentropy',
                     optimizer='adam',
                     metrics=['accuracy'])
    
    print('[INFO] Model has been compiled. ')
    
    return loaded_model
