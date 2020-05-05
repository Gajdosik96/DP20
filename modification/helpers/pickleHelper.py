import pickle


class PickleHelper:
    @staticmethod
    def save(path, data):
        with open(path, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            return handle

    @staticmethod
    def load(path):
        with open(path, 'rb' ) as handle:
            f = pickle.load(handle)
            return f
