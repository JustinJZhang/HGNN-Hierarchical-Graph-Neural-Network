import pickle

def save_obj(obj, name):
    if '.pkl' not in name:
        name = name + '.pkl'
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    if '.pkl' not in name:
        name = name + '.pkl'
    with open(name, 'rb') as f:
        return pickle.load(f)