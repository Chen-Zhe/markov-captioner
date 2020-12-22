import pickle


def pickle_save(obj, file_name):
  with open(f"{file_name}.pk", "wb") as file:
    pickle.dump(obj, file)

def pickle_load(file_name):
  with open(f"{file_name}.pk", "rb") as file:
    return pickle.load(file)


class Constant:
  end_token = "<end>"
  unknown_token = "<unk>"
  start_token = "<start>"