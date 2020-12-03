import sys, multiprocessing
from top2vec import Top2Vec
import umap, hdbscan
import logging

from joblib import dump, load

logger = logging.getLogger('gensim')
logger.setLevel(logging.INFO)
sh = logging.StreamHandler(sys.stderr)
sh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(sh)

model_file = 'top2vec.model'
model = Top2Vec.load(model_file)

wvs = model.model.wv.vectors
docvecs = model._get_document_vectors()

dims = [5,3,2]
metrics = ['cosine', 'euclidean']

for dim in dims:
    for metric in metrics:
        logger.info('Creating '+str(dim) + ' dimension embedding of documents with ' + metric)
        umap_model = umap.UMAP(n_neighbors=15,
                               n_components=dim,
                               metric=metric).fit(docvecs)

        #logger.info('Finding dense areas of documents')
        #cluster = hdbscan.HDBSCAN(min_cluster_size=15,
                                  #metric='euclidean',
                                  #cluster_selection_method='eom').fit(umap_model.embedding_)

        dump(umap_model, 'umap_docs_'+str(dim)+'_'+metric+'.umap')
        #dump(cluster, 'hdb_docs_'+str(dim)+'_'+metric+'.hdb')

for dim in dims:
    for metric in metrics:
        logger.info('Creating '+str(dim) + ' dimension embedding of words with ' + metric)
        umap_model = umap.UMAP(n_neighbors=15,
                               n_components=dim,
                               metric=metric).fit(wvs)

        #logger.info('Finding dense areas of words')
        #cluster = hdbscan.HDBSCAN(min_cluster_size=15,
                                  #metric='euclidean',
                                  #cluster_selection_method='eom').fit(umap_model.embedding_)

        dump(umap_model, 'umap_words_'+str(dim)+'_'+metric+'.umap')
        #dump(cluster, 'hdb_words_'+str(dim)+'_'+metric+'.hdb')


