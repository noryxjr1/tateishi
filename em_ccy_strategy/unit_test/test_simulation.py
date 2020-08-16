# -*- coding: utf-8 -*-
import os
from datetime import date
import unittest
import numpy as np
import pandas as pd
import pickle as pkl

from simulation.estimated_fc_sim import EstimatedFCSim

class test_simulation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass


    @classmethod
    def tearDownClass(cls):
        pass


    def setUp(self):
        with open(os.path.join('unit_test', 'ans_df.pkl'), 'rb') as f:
            self._expected_df = pkl.load(f)
    
    def tearDown(self):
        pass


    def test_em_ccy_sim(self):
        fc_label_file = os.path.join('unit_test', 'fc_label.csv')
        estimated_fc_sim = EstimatedFCSim(fc_label=fc_label_file, 
                                          end_date=date(2020, 8, 14))
        estimated_fc_sim.execute()
        actual_df = estimated_fc_sim.return_series

        self.assertAlmostEqual((actual_df - self._expected_df).sum().sum(), 0.0)

if __name__ == '__main__':
    import logging.config
    logging.config.fileConfig('./logger_config.ini')
    unittest.main()
