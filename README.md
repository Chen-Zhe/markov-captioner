## Markov Captioner
A Markov Chain that can describe an image based on the extracted features, in this case being the object categories and its location.

### Core Mathematics
Select the next best word (or sequence of words when beam width is greater than 1) by maximizing the conditional probability of the word's appearance given its previous few words
(Markov assumption) and the conditional probability of the word's appearance given all provided features.

<img src="https://render.githubusercontent.com/render/math?math=\mathbb{P}_f (w_i) = \mathbb{P}(w_i \mid w_{i-1}, w_{i-2}, x_1, x_2, \dots, x_n)">
<img src="https://render.githubusercontent.com/render/math?math=\stackrel{\textit{Bayes' Rule}}{=}\frac{\mathbb{P}( w_{i-1}, w_{i-2}, x_1, x_2, \dots, x_n \mid w_i) \mathbb{P}(w_i)} {\mathbb{P}(w_{i-1}, w_{i-2}, x_1, x_2, \dots, x_n)}">
<img src="https://render.githubusercontent.com/render/math?math==\frac{\mathbb{P}( w_{i-1}, w_{i-2}, x_1, x_2, \dots, x_n \mid w_i) \mathbb{P}(w_i)}   {\mathbb{P}( w_{i-2}, x_1, x_2, \dots, x_n \mid w_{i-1})   \mathbb{P}(w_{i-1})}">
<img src="https://render.githubusercontent.com/render/math?math=\stackrel{\textit{Naive Bayes}}{=}    \frac{\mathbb{P}( w_{i-1}, w_{i-2} \mid w_i)    \mathbb{P}(w_i)    \prod_{j=1}^{m}\mathbb{P}(x_j \mid w_i)    }    {\mathbb{P}(w_{i-2} \mid w_{i-1})    \mathbb{P}(w_{i-1})    \prod_{j=1}^{m}\mathbb{P}(x_j \mid w_{i-1})    }">

#### Dependencies
- `numpy`
- [`sparse`](https://github.com/pydata/sparse)

#### Parameters
- training
  - `ngram_n`: n-gram size used to train Markov Chain
  - `grid_size`: object location encoding is based on a nxn grid. This controls n
- sentence generation
  - `sentence_length_limit`: sentence cutoff length
  - `beam_width`: beam width for beam search
  - `decay_factor`: incremental penalty for generating the same word. This helps to reduce rambling of the model.

### Results
`ngram_n` = 4, `grid_size` = 2, `sentence_length_limit` = 16, `beam_width` = 20, `decay_factor` = 1e-2

|COCO ID|Image|Caption|
|---|---|---|
|184613| <img src="http://farm3.staticflickr.com/2169/2118578392_1193aa04a0_z.jpg" width="150"/>  |a group of people are standing in the grass near trees |
|272991| <img src="http://farm4.staticflickr.com/3259/5778841359_f4097a8f91_z.jpg" width="150"/>  | a hot dog with ketchup and mustard on top of it |
|403013| <img src="http://farm8.staticflickr.com/7369/8717355931_ebe09c411b_z.jpg" width="150"/>  | modern kitchen with stainless steel appliances and granite counter tops and stainless steel refrigerator microwave toaster  |
|562150| <img src="http://farm8.staticflickr.com/7002/6836351539_d19296470f_z.jpg" width="150"/>  | a cat laying on top of the steering wheel|

By extracting better features from the images, it might perform better than what I have here
