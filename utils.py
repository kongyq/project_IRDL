

def parser(file):
    docs = list()
    doc_ids = list()
    with open(file, 'r') as fp:
        for line in fp.readlines():
            elems = line.split('|')
            docs.append(elems[2])
            doc_ids.append(elems[0])
    return doc_ids, docs

