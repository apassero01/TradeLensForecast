import numpy as np
from django.test import TestCase

from sequenceset_manager.models import SequenceSet, Sequence
from training_session.models import ModelSet
from training_session.services.ModelSetService import ModelSetService


class ModelSetServiceTestCase(TestCase):
    def setUp(self):
        sequence_set = SequenceSet()
        sequence_set.dataset_type = 'stock'
        sequence_set.sequence_length = 10
        sequence_set.start_timestamp = '2023-01-01'
        sequence_set.metadata = {'ticker': 'AAPL'}
        sequence_set.sequences = []

        self.sequence_set = sequence_set

        sequence1 = Sequence()
        sequence1.id = 1
        sequence1.start_timestamp = '2023-01-01'
        sequence1.end_timestamp = '2023-01-10'
        sequence1.sliced_data = np.random.rand(10,2,2)

        sequence2 = Sequence()
        sequence2.id = 2
        sequence2.start_timestamp = '2023-01-01'
        sequence2.end_timestamp = '2023-01-10'
        sequence2.sliced_data = np.random.rand(10,2,2)

        sequence_set.sequences.append(sequence1)
        sequence_set.sequences.append(sequence2)

        self.model_set = ModelSet()
        self.model_set.data_set = sequence_set



    def test_get_sequence_set_members_by_id(self):
        ids = [1, 2]
        sequence_set, sequences = ModelSetService.get_sequence_set_members_by_id(self.model_set, ids)

        self.assertEqual(len(sequences), 2)
        seq_1 = sequences[0]
        self.assertEqual(seq_1.id, 1)
        seq_2 = sequences[1]
        self.assertEqual(seq_2.id, 2)

        ids = [1]
        sequence_set, sequences = ModelSetService.get_sequence_set_members_by_id(self.model_set, ids)
        self.assertEqual(len(sequences), 1)
        seq_1 = sequences[0]
        self.assertEqual(seq_1.id, 1)

    def test_get_sequence_set_metadata_by_id(self):
        ids = [1, 2]
        sequence_set, sequence_meta_data = ModelSetService.get_sequence_set_metadata_by_id(self.model_set, ids)

        self.assertEqual(len(sequence_meta_data), 2)
        seq_1 = sequence_meta_data[0]
        self.assertEqual(seq_1['id'], 1)
        self.assertEqual(seq_1['start_timestamp'], '2023-01-01')
        self.assertEqual(seq_1['end_timestamp'], '2023-01-10')
        self.assertEqual(seq_1['metadata'], {'ticker': 'AAPL'})

        seq_2 = sequence_meta_data[1]
        self.assertEqual(seq_2['id'], 2)
        self.assertEqual(seq_2['start_timestamp'], '2023-01-01')
        self.assertEqual(seq_2['end_timestamp'], '2023-01-10')
        self.assertEqual(seq_2['metadata'], {'ticker': 'AAPL'})



