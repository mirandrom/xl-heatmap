# XL Embedding Similarity Heatmap
A simple plotly dash app for visualizing similarities between 
a large number of embeddings as a dynamically downsampled heatmap with 
search capabilities.

## Overview
### Zoom and Pan
![Zoom and Pan](https://raw.githubusercontent.com/mirandrom/xl-heatmap/main/doc/zoom.gif) 

### Search Embeddings
![Zoom and Pan](https://raw.githubusercontent.com/mirandrom/xl-heatmap/main/doc/search.gif) 

### Change Resolution
![Resolution](https://raw.githubusercontent.com/mirandrom/xl-heatmap/main/doc/resolution.gif) 

### Change Similarities
![Similarites](https://raw.githubusercontent.com/mirandrom/xl-heatmap/main/doc/sims.gif)

## Setup
Clone the repository, then in an environment with Python 3.8+ run
```bash
pip install -r requirements.txt
```

If you have custom embeddings you want to analyze, save their torch tensors 
with shape (vocab_size, embed_dim) as a `.pt` file. If you want custom axis labels based on the 
embedding vocabulary, save the vocabulary as a new-line separated `.txt` file 
with the same name:
```python
import torch
from pathlib import Path

vocab_size = 30_000
embed_dim = 768
embeds = torch.rand(vocab_size, embed_dim)
vocab = [str(i) for i in range(vocab_size)]

EMBED_DIR = Path("xl-heatmap")
torch.save(embeds, EMBED_DIR / "your_embeds.pt")
(EMBED_DIR / "your_embeds.txt").write_text("\n".join(vocab))
```

Now you can run the app, using your custom embedding directory so the embeddings
 and vocabulary are automatically discovered:
```bash
python PATH_TO_XL_HEATMAP/app.py --path EMBED_DIR --debug
```
## Caveats
This app was created for language model embeddings with a vocabulary 
size of ~30000 embeddings, and uses mean-pooling to downsample a pre-computed 
array of the full similarities. If you want to use a larger vocabulary size, or you do not have enough space on 
your machine to store the full pre-computed similarities in memory, you can 
change the behaviour of ``get_cached_sims`` and ``downsample`` to compute 
similarities for a downsampled set of embeddings on the fly. 

