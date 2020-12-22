import json
import Utility
import DataLoader
import sys
from multiprocessing import Pool

def gen_caption(img_obj_tuple):
  img_id, objects = img_obj_tuple
  return img_id, markov_captioner.generate_caption(objects.items(),
                                        sentence_length_limit=16,
                                        beam_width=int(sys.argv[3]),
                                        decay_factor=float(sys.argv[4]))[0]

print(f"testing captions with {sys.argv[1]}-gram {sys.argv[2]}-grid using beam width {sys.argv[3]} and decay factor {sys.argv[4]}")

markov_captioner = Utility.pickle_load(f"location-encoded-markov-captioner-{sys.argv[1]}-gram-{sys.argv[2]}-grid")

if __name__ == '__main__':

  test_captions = list()

  _, objects_dict = DataLoader.load_and_preprocess_detected_objects(grid_size = int(sys.argv[2]))

  image_defs = DataLoader.load_image_definition("data/cocotalk.json")

  test_images = list()

  for img_id, _ in image_defs[sys.argv[5]]:
    if img_id in objects_dict:
      test_images.append((img_id, objects_dict[img_id]))

  with Pool(processes=4) as pool:
    captions = pool.imap(gen_caption, test_images)

    for i in range(len(test_images)):
      img_id, caption = next(captions)
      if img_id is not None:
        test_captions.append({
          "image_id": img_id,
          "caption": caption
        })
        

  with open(f'markov_result_{sys.argv[5]}_{sys.argv[1]}_gram_{sys.argv[2]}_grid_{sys.argv[3]}_beam_{sys.argv[4]}_decay.json', 'w') as outfile:
      json.dump(test_captions, outfile)
