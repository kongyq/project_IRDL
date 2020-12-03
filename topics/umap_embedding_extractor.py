import umap,sys
import logging

from joblib import dump, load

logger = logging.getLogger('gensim')
logger.setLevel(logging.INFO)
sh = logging.StreamHandler(sys.stderr)
sh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(sh)


dims = [2,3,5]
metrics = ["cosine", "euclidean"]

for dim in dims:
    for metric in metrics:
        logger.info('Extract ' + str(dim) + ' dimension embedding of documents with ' + metric + "from umap model")
        umap_docs_file = 'umap_docs_'+str(dim)+'_'+metric+'.umap'
        umap_model = load(umap_docs_file)
        umap_docs_embeddings_file = 'umap_docs_embeddings_'+str(dim)+'_'+metric+'.emb'
        dump(umap_model.embedding_, umap_docs_embeddings_file)

for dim in dims:
    for metric in metrics:
        logger.info('Extract ' + str(dim) + ' dimension embedding of words with ' + metric + "from umap model")
        umap_words_file = 'umap_words_'+str(dim)+'_'+metric+'.umap'
        umap_model = load(umap_words_file)
        umap_words_embeddings_file = 'umap_words_embeddings_'+str(dim)+'_'+metric+'.emb'
        dump(umap_model.embedding_, umap_words_embeddings_file)