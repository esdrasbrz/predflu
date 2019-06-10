from rdflib import Graph
from rdflib.plugins.sparql.results.jsonresults import JSONResultSerializer
import io
import json


class RDFEndpoint:
    def __init__(self, endpoint):
        self.endpoint = endpoint
        self._connect()

    def _connect(self, format="xml"):
        self.graph = Graph()
        self.graph.parse(self.endpoint, format=format)

    def execute_query(self, query):
        res = self.graph.query(query)
        res_stream = io.StringIO()
        JSONResultSerializer(res).serialize(res_stream)

        res = res_stream.getvalue()
        json_res = json.loads(res)

        return json_res

