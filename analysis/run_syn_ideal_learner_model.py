#!/usr/bin/env python
# encoding: utf-8
"""
================================================
Synesthetic Ideal Learner Model

Python code O.Colizoli 2024 (olympia.colizoli@donders.ru.nl)
Python 3.9

Notes
-----
================================================
"""

# importing python packages
import os, sys, datetime, time, shutil
import numpy as np
import pandas as pd
import syn_ideal_learner_model as higher
#from IPython import embed as shell # for debugging

# -----------------------
# Levels (toggle True/False)
# ----------------------- 
higher_level    = True   # all subjects' dataframe, pupil and behavior higher level analyses & figures (2AFC decision)
 
# -----------------------
# Paths
# ----------------------- 
# set path to home directory
home_dir        = os.path.dirname(os.getcwd()) # one level up from analysis folder
stimuli_folder  = os.path.join(home_dir, 'stimuli')
deriv_folder    = os.path.join(home_dir, 'derivatives')
experiment_name = 'task-syn_ideal_learner_model' # 2AFC Decision Task

# -----------------------
# Participants
# -----------------------
subjects = ['sub-1']
    
# -----------------------
# 2AFC Decision Task, MEAN responses and group level statistics 
# ----------------------- 
if higher_level:  
    higherLevel = higher.higherLevel(
        subjects                = subjects, 
        experiment_name         = experiment_name,
        project_directory       = deriv_folder, 
        stimuli_directory       = stimuli_folder, 
        )
    
    ''' Ideal learner model
    '''
    higherLevel.plot_color_space()
    higherLevel.plot_prediction_space()
    higherLevel.information_theory_estimates()
    higherLevel.pupil_information_correlation_matrix()
    higherLevel.plot_KL_priors()
    # higherLevel.plot_information_parameters()
    # higherLevel.plot_information_parameters_by_prior_strength()
    # higherLevel.plot_information_parameters_congruency_by_frequency()


