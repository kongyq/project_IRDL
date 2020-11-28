import sys, multiprocessing
from top2vec import Top2Vec
import logging

logger = logging.getLogger('gensim')
logger.setLevel(logging.INFO)
sh = logging.StreamHandler(sys.stderr)
sh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(sh)


def parser(file):
    docs = list()
    doc_ids = list()
    with open(file, 'r') as fp:
        for line in fp.readlines():
            elems = line.split('|')
            docs.append(elems[2])
            doc_ids.append(elems[0])
    return docs, doc_ids


def main():
    if not sys.argv[1:]:
        print("need to give dataset path!")
        exit(1)

    data_file = sys.argv[1:][0]

    print(data_file)
    save_file = "top2vec.model"
    docs, doc_ids = parser(data_file)

    model = Top2Vec(documents=docs, document_ids=doc_ids, min_count=20, keep_documents=False,
                    use_corpus_file=True, workers=multiprocessing.cpu_count(),
                    verbose=True)

    model.save(save_file)


if __name__ == "__main__":
    main()
