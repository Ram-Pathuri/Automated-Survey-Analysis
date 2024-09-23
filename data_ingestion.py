# shared_data.py
import pickle

def save_data(data, filename='data.pkl'):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

def load_data(filename='data.pkl'):
    try:
        with open(filename, 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        return None
