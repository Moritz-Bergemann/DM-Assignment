Data Mining assignment by Moritz Bergemann (19759948)

To run: 
Run run.py to generate predictions file 'predict.csv'
NOTE: pickle files "model1-comnbayes.pickle" and "model2-svc.pickle", and "data2021.student.csv" are required to make prediction via `run.py`

Other notes:
Majority of model training code contained in trials.py - trials experiments performed on different aspects of model.
Different experiments are run by editing the main() function to call the appropriate experiment() function.
'data_prep.py' contains data preparation code - run by importing and calling get_prepped_dataset()
Graphics generation can be found in generate_graphics.ipynb
More general testing in testing.ipynb