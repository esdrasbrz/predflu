from ontologies.sparql import RDFEndpoint
from threading import Thread
import config as cfg
import os

SYMP_RDF = os.path.join(cfg.DATA_PATH, 'symp.xrdf')
FLU_RDF = os.path.join(cfg.DATA_PATH, 'flu.xrdf')

assert os.path.exists(SYMP_RDF)
assert os.path.exists(FLU_RDF)

symp_endpoint = RDFEndpoint(SYMP_RDF)
flu_endpoint = RDFEndpoint(FLU_RDF)

SPARQL_PREFIXES = """
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX obo: <http://purl.obolibrary.org/obo/>
"""


def _add_prefixes(query):
    return "{}\n{}".format(SPARQL_PREFIXES, query)


def _pool_filter(pool, fn, values, chunksize=16):
    res = [v for v, keep in zip(values, pool.imap(fn, values, chunksize=chunksize)) if keep]
    return res


def is_symptom(s):
    query = """
    SELECT ?symp
    WHERE {
        ?symp rdfs:label ?label
        filter(str(?label)="%s")
    }
    """ % (s)

    query = _add_prefixes(query)
    res = symp_endpoint.execute_query(query)
    return len(res['results']['bindings']) > 0


def all_flu_symptoms():
    query = """
    SELECT ?label
    WHERE {
        ?symp rdfs:subClassOf obo:OGMS_0000020 .
        ?symp rdfs:label ?label
    }
    """

    query = _add_prefixes(query)
    res = flu_endpoint.execute_query(query)
    labels = set([l['label']['value'] for l in res['results']['bindings']])
    return labels


def is_flu_symptom(s):
    query = """
    SELECT ?symp
    WHERE {
        ?symp rdfs:subClassOf obo:OGMS_0000020 .
        ?symp rdfs:label ?label
        filter(str(?label)="%s") .
    }
    """ % (s)

    query = _add_prefixes(query)
    res = flu_endpoint.execute_query(query)
    return len(res['results']['bindings']) > 0


def _chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


class ThreadWithReturnValue(Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)

    def join(self, *args):
        Thread.join(self, *args)
        return self._return


def _thread_filter(fn, tokens, n_threads):
    if n_threads <= 1:
        return set(filter(fn, tokens))

    filter_fn = lambda ch: set(filter(fn, ch))
    chunks = _chunks(tokens, len(tokens) // n_threads + 1)
    threads = []
    for ch in chunks:
        th = ThreadWithReturnValue(target=filter_fn, args=(ch,))
        th.start()
        threads.append(th)

    res = set()
    for th in threads:
        ch = th.join()
        if ch:
            res = res.union(ch)

    return res


def symptoms(tokens, n_threads=cfg.N_THREADS):
    return _thread_filter(is_symptom, tokens, n_threads)


def flu_symptoms(tokens, n_threads=1):
    all_symptoms = all_flu_symptoms()

    return _thread_filter(lambda t: t in all_symptoms, tokens, n_threads)

