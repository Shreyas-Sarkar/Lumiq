import unittest
import pandas as pd
from services.mode_classifier import ModeClassifier
from services.query_cache import QueryCache
from services.explanation import ExplanationService
from services.evaluator import Evaluator
from models.schemas import ExecutionResult


class BaselineTests(unittest.IsolatedAsyncioTestCase):
    def test_concept_and_comparison_are_not_shadowed_by_executor(self):
        classifier = ModeClassifier()
        df = pd.DataFrame({'revenue': [10], 'region': ['A']})
        self.assertEqual(classifier.classify('What is correlation in general?', df), 'concept')
        self.assertEqual(classifier.classify('What is the weather today?', df), 'irrelevant')
        self.assertEqual(classifier.classify('Compare revenue between regions', df), 'hybrid')

    def test_cache_context_and_invalidation(self):
        cache = QueryCache()
        cache.set('what about that region?', 'dataset', {'answer': 'A'}, context='region A')
        self.assertIsNone(cache.get('what about that region?', 'dataset', context='region B'))
        cache.invalidate('dataset')
        self.assertIsNone(cache.get('what about that region?', 'dataset', context='region A'))

    async def test_plot_only_is_valid_and_plot_with_table_is_retained(self):
        result = ExecutionResult(success=True, output=None, output_type='plot', execution_time_ms=1, serialized_output='', plot_base64='image')
        self.assertTrue((await Evaluator(None).evaluate('plot revenue', '', result)).is_valid)
        result.output = pd.DataFrame({'value': [10]})
        result.output_type = 'dataframe'
        result.serialized_output = '{"columns":["value"],"data":[[10]]}'
        response = ExplanationService().format_enriched_response('', '', result, {}, [], 0, 'executor')
        self.assertEqual(response['plot_base64'], 'image')
        self.assertEqual(response['result']['data'], [[10]])
