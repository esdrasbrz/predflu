from ontologies.sparql import RDFEndpoint
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


def is_symptom_of_flu(s):
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

