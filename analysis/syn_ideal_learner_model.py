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

import os, sys, datetime
import numpy as np
import scipy.stats as stats
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from copy import deepcopy
#from IPython import embed as shell # for debugging only

pd.set_option('display.float_format', lambda x: '%.8f' % x) # suppress scientific notation in pandas

""" Plotting Format
############################################
# PLOT SIZES: (cols,rows)
# a single plot, 1 row, 1 col (2,2)
# 1 row, 2 cols (2*2,2*1)
# 2 rows, 2 cols (2*2,2*2)
# 2 rows, 3 cols (2*3,2*2)
# 1 row, 4 cols (2*4,2*1)
# Nsubjects rows, 2 cols (2*2,Nsubjects*2)

############################################
# Define parameters
############################################
"""
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
sns.set(style='ticks', font='Arial', font_scale=1, rc={
    'axes.linewidth': 1, 
    'axes.labelsize': 7, 
    'axes.titlesize': 7, 
    'xtick.labelsize': 7, 
    'ytick.labelsize': 7, 
    'legend.fontsize': 7, 
    'xtick.major.width': 1, 
    'ytick.major.width': 1,
    'text.color': 'Black',
    'axes.labelcolor':'Black',
    'xtick.color':'Black',
    'ytick.color':'Black',} )
sns.plotting_context()


class higherLevel(object):
    """Define a class for the higher level analysis.

    Parameters
    ----------
    subjects : list
        List of subject numbers
    experiment_name : string
        Name of the experiment for output files
    project_directory : str
        Path to the derivatives data directory
    stimuli_directory : str
        Path to the stimuli directory for input to model

    Attributes
    ----------
    subjects : list
        List of subject numbers
    exp : string
        Name of the experiment for output files
    project_directory : str
        Path to the derivatives data directory
    figure_folder : str
        Path to the figure directory
    deriv_folder : str
        Path to the derivatives directory
    stimuli_folder : str
        Path to the stimuli directory for input
    """
    
    def __init__(self, subjects, experiment_name, project_directory, stimuli_directory):        
        """Constructor method
        """
        self.subjects           = subjects
        self.exp                = experiment_name
        self.project_directory  = project_directory
        self.figure_folder      = os.path.join(project_directory, 'figures')
        self.deriv_folder       = project_directory
        self.stimuli_folder     = stimuli_directory
        
        if not os.path.isdir(self.figure_folder):
            os.mkdir(self.figure_folder)
            
 
    def plot_color_space(self, ):
        """Plot the prediction 2D space for manipulating trained letters.

        Notes
        -----
        1 figure
        Figure output as PDF in figure folder.
        """
        
        from mpl_toolkits import mplot3d
        
        df = pd.read_csv(os.path.join(self.stimuli_folder, 'rgb_colors.csv'))
        
        unique_colors = df['colorcode']

        # single figure        
        # 2D space frequency x probability
        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot(1,2,1, projection='3d')
        
        x = df['r']
        y = df['g']
        z = df['b']
        
        c = []
        for i,r in enumerate(x):
            c.append([x[i]/255, y[i]/255, z[i]/255])
                    
        ax.scatter(x,y,z, c = c)

        # set figure parameters
        ax.set_xlabel('R')
        ax.set_ylabel('G')
        ax.set_zlabel('B')
        
        ax.set_xlim([0,255])
        ax.set_ylim([0,255])
        ax.set_zlim([0,255])
        
        sns.despine(offset=10, trim=True)
        plt.tight_layout()
        fig.savefig(os.path.join(self.figure_folder, '{}_color_space.pdf'.format(self.exp)))
        print('success: plot_color_space')
        
        
    def plot_prediction_space(self, ):
        """Plot the prediction 2D space for manipulating trained letters.

        Notes
        -----
        1 figure
        Figure output as PDF in figure folder.
        """
        df = pd.read_csv(os.path.join(self.stimuli_folder, 'prediction_space.csv'))
        unique_letters = np.arange(len(df['letter']))
        
        colors = pd.read_csv(os.path.join(self.stimuli_folder, 'rgb_colors.csv'))
        c = []
        for rgb in np.arange(len(unique_letters)):
            c.append([colors['rn'][rgb], colors['gn'][rgb], colors['bn'][rgb]])
        
        # single figure
        fig = plt.figure(figsize=(3,4))
        
        # 2D space frequency x probability
        ax = fig.add_subplot(2,1,1)
        ax.axhline(0, lw=1, alpha=1, color = 'k') # Add horizontal line at t=0
        
        x = df['letter_frequency']
        y = df['probability']
        letters = df['letter']
        
        r, p = stats.pearsonr(x, y)
        # plot
        # hack scatter legend
        c1 = ax.scatter(x[0], y[0], c=c[0])
        c2 = ax.scatter(x[1], y[1], c=c[1])
        c3 = ax.scatter(x[2], y[2], c=c[2])
        c4 = ax.scatter(x[3], y[3], c=c[3])
        c5 = ax.scatter(x[4], y[4], c=c[4])
        c6 = ax.scatter(x[5], y[5], c=c[5])
        c7 = ax.scatter(x[6], y[6], c=c[6])
        c8 = ax.scatter(x[7], y[7], c=c[7])
        c9 = ax.scatter(x[8], y[8], c=c[8])
        c10 = ax.scatter(x[9], y[9], c=c[9])
        
        ax.legend((c1, c2, c3, c4, c5, c6, c7, c8, c9, c10),
           (letters),
           scatterpoints=1,
           loc='best',
           ncol=1,
           fontsize=7)
        
        # set figure parameters
        ax.set_title('r = {}, p = {}'.format(np.round(r,2), np.round(p, 3)))
        ax.set_ylabel('Letter-color probability')
        ax.set_xlabel('Relative letter frequency')
        # ax.set_ylim([0, 100])
        
        # # Combined priors
        # ax = fig.add_subplot(2,1,2)
        # ax.axhline(0, lw=1, alpha=1, color = 'k') # Add horizontal line at t=0
        #
        # y = df['prior']
        # x = df['letter_frequency']
        #
        # # plot bar graph
        # ax.scatter(x, y, c=unique_letters, cmap='viridis')
        #
        # # set figure parameters
        # ax.set_title('Combined prior')
        # ax.set_ylabel('Prior')
        # ax.set_xlabel('Relative letter frequency')

        sns.despine(offset=10, trim=True)
        plt.tight_layout()
        fig.savefig(os.path.join(self.figure_folder, '{}_prediction_space.pdf'.format(self.exp)))
        print('success: plot_prediction_space')
        

    def idt_model(self, df, df_data_column, elements, priors):
        """Process Ideal Learner Model.
        
        Parameters
        ----------
        df : pandas dataframe
            The dataframe to apply the Ideal Learner Model to.
        
        df_data_column : str
            The name of the column that refers to the cue-target pairs for all trials in the experiment.
        
        elements : list
            The list of unique indentifiers for the cue-target pairs.
        
        priors : list
            The list of priors as probabilities.
        
        Returns
        -------
        [model_e, model_P, model_p, model_I, model_i, model_H, model_CH, model_D]: list
            A list containing all model parameters (see notes).
            
        Notes
        -----
        Ideal Learner Model adapted from Poli, Mars, & Hunnius (2020).
        See also: https://github.com/FrancescPoli/eye_processing/blob/master/ITDmodel.m
        
        Priors are generated from the probabilities of the letter-color pair in the odd-ball task.
        
        Model Output Notes:
        model_e = trial sequence
        model_P = probabilities of all elements at each trial
        model_p = probability of current element at current trial
        model_I = surprise of all elements at each trial (i.e., complexity)
        model_i = surprise of current element at current trial
        model_H = negative entropy at current trial
        model_CH = cross-entropy at current trial
        model_D = KL-divergence at current trial
        """
        
        data = np.array(df[df_data_column])
    
        # initialize output variables for current subject
        model_e = [] # trial sequence
        model_P = [] # probabilities of all elements at each trial
        model_p = [] # probability of current element at current trial
        model_I = [] # surprise of all elements at each trial
        model_i = [] # surprise of current element at current trial
        model_H = [] # entropy at current trial
        model_CH = [] # cross-entropy at current trial
        model_D = []  # KL-divergence at current trial
    
        # loop trials
        for t in np.arange(df.shape[0]):
            vector = data[:t+1] #  trial number starts at 0, all the targets that have been seen so far

            # print(vector)
            if t < 1: # if it's the first trial, our expectations are based only on the prior (values)
                alpha1 = priors*len(elements) # np.sum(alpha) == len(elements), priors from ???
                p1 = priors # priors based on odd-ball task, np.sum(priors) should equal 1
                p = p1
                
            # at every trial, we compute surprise.
            # Surprise is defined by the negative log of the probability of the current trial given the previous trials.
            I = -np.log2(p)     # complexity of every event (each cue_target_pair is a potential event)
            i = I[vector[-1]]   # surprise of the current event (last element in vector)
    
            # Updated estimated probabilities
            p = []
            for k in elements:
                # +1 because in the prior there is one element of the same type; +4 because in the prior there are 4 elements
                # The influence of the prior should be sampled by a distribution or
                # set to a certain value based on Kidd et al. (2012, 2014)
                p.append((np.sum(vector == k) + alpha1[k]) / (len(vector) + len(alpha1)))   

            model_e.append(vector[-1])  # element in current trial = last element in the vector
            model_P.append(p)           # probability of all elements in NEXT trial
            model_p.append(p[vector[-1]]) # probability of element in NEXT trial
            model_I.append(I)
            model_i.append(i)
    
            # once we have the updated probabilities, we can compute KL Divergence, Entropy and Cross-Entropy
            prevtrial = t-1
            if prevtrial < 0: # first trial
                D = np.sum(p * (np.log2(p / np.array(p1)))) # KL divergence, after vs. before, same direction as Poli et al. 2020
            else:
                D = np.sum(p * (np.log2(p / np.array(model_P[prevtrial])))) # KL divergence, after vs. before, same direction as Poli et al. 2020
            
            H = -np.sum(p * np.log2(p)) # entropy (note that np.log2(1/p) is equivalent to multiplying the whole sum by -1)
    
            CH = H + D # Cross-entropy
    
            model_H.append(H)   # negative entropy
            model_CH.append(CH) # cross-entropy
            model_D.append(D)   # KL divergence
        
        return [model_e, model_P, model_p, model_I, model_i, model_H, model_CH, model_D]
        
        
    def information_theory_estimates(self, ):
        """Run subject loop on Ideal Learner Model and save model estimates.
        
        Notes
        -----
        Ideal Learner Model adapted from Poli, Mars, & Hunnius (2020).
        See also: https://github.com/FrancescPoli/eye_processing/blob/master/ITDmodel.m
        
        Model estimates that are saved in subject's dataframe:
        model_i = surprise of current element at current trial
        model_H = negative entropy at current trial
        model_D = KL-divergence at current trial
        
        Priors have to be non-zero or get division by zero errors!
        """
        
        fn_in = os.path.join(self.stimuli_folder, 'syn_ideal_learner_model_trials_2D.csv')
                
        df_in = pd.read_csv(fn_in, float_precision='high')
        # drop subject and trial_number columns
        df_in = df_in.loc[:, ~df_in.columns.str.contains('trial_num')] # drop all unnamed columns
        df_in = df_in.loc[:, ~df_in.columns.str.contains('subject')] # drop all unnamed columns
    
        elements = np.unique(df_in['letter_color_pair'])
        
        for dprior in ['2d_prior', 'flat_priors']:
            
            df_out = pd.DataFrame()
            df_prob_out = pd.DataFrame() # last probabilities all elements saved
            
            # loop subjects
            for s,subj in enumerate(np.arange(30)+1):
                
                # replicate then randomize trials
                this_df = pd.DataFrame(np.repeat(df_in.values, 20, axis=0))
                this_df.columns = df_in.columns.values
                this_df = this_df.sample(frac=1).reset_index(drop=True) 
                this_df['trial_num'] = np.arange(this_df.shape[0]) + 1
        
                this_subj = 'sub-{}'.format(subj)

                # get current subjects data only
                this_priors = df_in[dprior] # priors for current subject
                
                # the input to the model is the trial sequence = the order of letter-color pair for each participant
                [model_e, model_P, model_p, model_I, model_i, model_H, model_CH, model_D] = self.idt_model(this_df, 'letter_color_pair', elements, this_priors)
            
                # add to subject dataframe
                this_df['subject'] = np.repeat(this_subj, this_df.shape[0])
                this_df['model_p'] = np.array(model_p)
                this_df['model_i'] = np.array(model_i)
                this_df['model_H'] = np.array(model_H)
                this_df['model_D'] = np.array(model_D)
                df_out = pd.concat([df_out, this_df])    # add current subject df to larger df
            
                df_prob_out['{}'.format(this_subj)] = np.array(model_P[-1])
                print(subj)
        
            # save whole DF
            df_out = df_out.loc[:, ~df_out.columns.str.contains('^Unnamed')] # drop all unnamed columns
            fn_out = os.path.join(self.deriv_folder,'syn_ideal_learner_model_parameters_{}.csv'.format(dprior) )
            df_out.to_csv(fn_out, float_format='%.16f')
        print('success: information_theory_estimates')
        
        
    def pupil_information_correlation_matrix(self,):
        """Correlate information variables to evaluate multicollinearity.
        
        Notes
        -----
        Model estimates that are correlated per subject the tested at group level:
        model_i = surprise of current element at current trial
        model_H = negative entropy at current trial
        model_D = KL-divergence at current trial
        
        See figure folder for plot and output of t-test.
        """
        
        ivs = ['model_i', 'model_H', 'model_D']
        labels = ['i' , 'H', 'KL']
        
        for dprior in ['2d_prior', 'flat_priors']: 
                   
            DF = pd.read_csv(os.path.join(self.deriv_folder,'syn_ideal_learner_model_parameters_{}.csv'.format(dprior)), float_precision='high')
        
            corr_out = []

            # loop subjects
            for s, subj in enumerate(np.unique(DF['subject'])):
            
                # get current subject's data only
                this_df = DF[DF['subject']==subj].copy(deep=False)
                            
                x = this_df[ivs] # select information variable columns
                x_corr = x.corr() # correlation matrix

                corr_out.append(x_corr) # beta KLdivergence (target-prediction)
        
            corr_subjects = np.array(corr_out)
            corr_mean = np.mean(corr_subjects, axis=0)
            corr_std = np.std(corr_subjects, axis=0)
        
            t, pvals = stats.ttest_1samp(corr_subjects, 0, axis=0)
        
            f = open(os.path.join(self.figure_folder, '{}_pupil_information_correlation_matrix.txt'.format(self.exp)), "w")
            f.write('corr_mean')
            f.write('\n')
            f.write('{}'.format(corr_mean))
            f.write('\n')
            f.write('\n')
            f.write('tvals')
            f.write('\n')
            f.write('{}'.format(t))
            f.write('\n')
            f.write('\n')
            f.write('pvals')
            f.write('\n')
            f.write('{}'.format(pvals))
            f.close
        
            ### PLOT ###
            fig = plt.figure(figsize=(4,2))
            ax = fig.add_subplot(121)
            cbar_ax = fig.add_subplot(122)
        
            # mask for significance
            mask_pvals = pvals < 0.05
            mask_pvals = ~mask_pvals # True means mask this cell
        
            # plot only lower triangle
            mask = np.triu(np.ones_like(corr_mean))
            mask = mask + mask_pvals # only show sigificant correlations in heatmap
        
            ax = sns.heatmap(corr_mean, vmin=-1, vmax=1, mask=mask, cmap='bwr', cbar_ax=cbar_ax, xticklabels=labels, yticklabels=labels, square=True, annot=True, ax=ax)
        
            # whole figure format
            sns.despine(offset=10, trim=True)
            plt.tight_layout()
            fig.savefig(os.path.join(self.figure_folder,'{}_pupil_information_correlation_matrix_{}.pdf'.format(self.exp, dprior)))
                        
        print('success: pupil_information_correlation_matrix')


    def plot_KL_priors(self, ):
        """Plot the KL distribution for the three prior distributions.

        Notes
        -----
        1 figure, GROUP LEVEL DATA
        Figure output as PDF in figure folder.
        """
        model_dvs = ['model_i', 'model_H', 'model_D']
        ylabels = ['Surprise', 'Negative entropy', 'KL divergence']
        titles = ['Subjective prior distribution', 'Uniform prior distribution']
        factor = '2d_prior'
        colors = ['teal', 'orange', 'purple']
        bar_width = 0.7
        ylim = [-0.3, 0.3]
        
        priors = pd.read_csv(os.path.join(self.deriv_folder, 'syn_ideal_learner_model_parameters_{}.csv'.format('2d_prior')))
        
        xticklabels = np.arange(len(np.unique(priors[factor])))
        xind = np.arange(len(np.unique(priors[factor])))
        
        priors = priors[['2d_prior' , 'flat_priors']]
        
        
        # single figure
        fig = plt.figure(figsize=(6,4))
        counter = 1
        
        for p,dprior in enumerate(['2d_prior' , 'flat_priors']): 
            
            for m, model_dv in enumerate(model_dvs):
                
                ax = fig.add_subplot(2, 3, counter)
                
                ax.axhline(0, lw=1, alpha=1, color = 'k') # Add horizontal line at t=0
            
                DFIN = pd.read_csv(os.path.join(self.deriv_folder, 'syn_ideal_learner_model_parameters_{}.csv'.format(dprior)))
                DFIN = DFIN.loc[:, ~DFIN.columns.str.contains('^Unnamed')] # drop all unnamed columns
    
                subjects = np.unique(DFIN['subject'])    
            
                DFIN = DFIN.groupby(['subject', factor])[model_dv].mean() # means per subject

                GROUP = DFIN.groupby([factor]).mean()

                # Group average
                SEM = np.true_divide(DFIN.groupby([factor]).mean(), np.sqrt(len(subjects)))
                print(GROUP)
            
                # plot bar graph
                ax.bar(xind, np.array(GROUP), width=bar_width, yerr=np.array(SEM), capsize=1, color=colors[m], edgecolor='black', ecolor='black')

                # set figure parameters
                ax.set_title(titles[p])
                ax.set_ylabel(ylabels[m])
                ax.set_xlabel('Strength of prior expectation')
                # ax.set_ylim(ylim)
                ax.set_xticks([xind[0],xind[-1]])
                ax.set_xticklabels(['Low', 'High'])
                
                counter = counter + 1

            sns.despine(offset=10, trim=True)
            plt.tight_layout()
            fig.savefig(os.path.join(self.figure_folder,'{}_model_parameters.pdf'.format(self.exp)))
        print('success: plot_KL_priors')
        

    def plot_information_parameters(self, ):
        """Plot the group level average model parameters across all trials.

        Notes
        -----
        1 figure, GROUP LEVEL DATA
        Figure output as PDF in figure folder.
        """
        dvs = ['model_i', 'model_H', 'model_D']
        ylabels = ['Surprise', 'Negative entropy', 'KL divergence']
        xlabel = 'Letter-color pair prior'
        xticklabels = ['Incongruent','Congruent']
        colors = ['teal', 'orange', 'purple']
        bar_width = 0.7
        xind = np.arange(len(xticklabels))
        ylim = [-0.3, 0.3]

        DFIN = pd.read_csv(os.path.join(self.deriv_folder, 'syn_ideal_learner_model_parameters_{}.csv'.format('2d_prior')), float_precision='high')

        DFIN = DFIN.loc[:, ~DFIN.columns.str.contains('^Unnamed')] # drop all unnamed columns
        subjects = np.unique(DFIN['subject'])

        DFIN = DFIN.groupby(['subject','congruent'])[dvs].mean() # means per subject

        # single figure
        fig = plt.figure(figsize=(2,6))

        for dvi, model_dv in enumerate(dvs):

            ax = fig.add_subplot(3, 1, dvi+1)

            GROUP = DFIN.groupby(['congruent'])[model_dv].mean()

            # Group average
            SEM = np.true_divide(DFIN.groupby(['congruent'])[model_dv].mean(), np.sqrt(len(subjects)))
            print(GROUP)

            ax.axhline(0, lw=1, alpha=1, color = 'k') # Add horizontal line at t=0

            # plot bar graph
            ax.bar(xind, GROUP, width=bar_width, yerr=SEM, capsize=3, color=colors[dvi], edgecolor='black', ecolor='black')

            # set figure parameters
            ax.set_ylabel(ylabels[dvi])
            ax.set_xlabel(xlabel)
            # ax.set_ylim(ylim)
            ax.set_xticks(xind)
            ax.set_xticklabels(xticklabels)

        sns.despine(offset=10, trim=True)
        plt.tight_layout()
        fig.savefig(os.path.join(self.figure_folder,'{}_congruency.pdf'.format(self.exp)))
        print('success: plot_information_parameters')


    def plot_information_parameters_by_prior_strength(self, ):
        """Plot the group level average model parameters as a function of prior strength.

        Notes
        -----
        1 figure, GROUP LEVEL DATA
        x-axis time window.
        Figure output as PDF in figure folder.
        """
        dvs = ['model_i', 'model_H', 'model_D']
        ylabels = ['Surprise', 'Negative entropy', 'KL divergence']
        xlabel = 'Prior strength'
        colors = ['teal', 'orange', 'purple']
        bar_width = 0.7
        # ylim = [-0.012, 0]

        DFIN = pd.read_csv(os.path.join(self.deriv_folder, 'syn_ideal_learner_model_parameters_2d_prior.csv'), float_precision='high')
        DFIN = DFIN.loc[:, ~DFIN.columns.str.contains('^Unnamed')] # drop all unnamed columns

        xticklabels = np.arange(len(np.unique(DFIN['2d_prior'])))
        xind = np.arange(len(np.unique(DFIN['2d_prior'])))

        subjects = np.unique(DFIN['subject'])

        DFIN = DFIN.groupby(['subject','2d_prior'])[dvs].mean() # means per subject

        # single figure
        fig = plt.figure(figsize=(2,6))

        for dvi, model_dv in enumerate(dvs):
            ax = fig.add_subplot(3, 1, dvi+1)

            GROUP = DFIN.groupby(['2d_prior'])[model_dv].mean()

            # Group average
            SEM = np.true_divide(DFIN.groupby(['2d_prior'])[model_dv].mean(), np.sqrt(len(subjects)))
            print(GROUP)

            ax.axhline(0, lw=1, alpha=1, color = 'k') # Add horizontal line at t=0

            # plot bar graph
            ax.bar(xind, GROUP, width=bar_width, yerr=SEM, capsize=1, color=colors[dvi], edgecolor='black', ecolor='black')

            # set figure parameters
            ax.set_ylabel(ylabels[dvi])
            ax.set_xlabel(xlabel)
            # ax.set_ylim(ylim)
            ax.set_xticks(xind)
            ax.set_xticklabels(xticklabels)

            sns.despine(offset=10, trim=True)
            plt.tight_layout()
            fig.savefig(os.path.join(self.figure_folder,'{}_prior_strength.pdf'.format(self.exp)))
        print('success: plot_information_parameters_by_prior_strength')


    def plot_information_parameters_congruency_by_frequency(self, ):
        """Plot the group level average model parameters as a function of letter-frequency and congruency across all trials.

        Notes
        -----
        1 figure, GROUP LEVEL DATA
        Figure output as PDF in figure folder.
        """
        dvs = ['model_i', 'model_H', 'model_D']
        ylabels = ['Surprise', 'Negative entropy', 'KL divergence']
        xlabel = 'Letter frequency'
        colors = ['teal', 'orange', 'purple']
        bar_width = 0.4
        ylim = [-0.3, 0.3]
        alphas = [0.5, 1]

        DFIN = pd.read_csv(os.path.join(self.deriv_folder, 'syn_ideal_learner_model_parameters_2d_prior.csv'), float_precision='high')

        DFIN = DFIN.loc[:, ~DFIN.columns.str.contains('^Unnamed')] # drop all unnamed columns
        subjects = np.unique(DFIN['subject'])
        
        xticklabels = np.unique(DFIN['letter'])
        xind = np.arange(len(xticklabels))

        DFIN = DFIN.groupby(['subject', 'congruent', 'letter'])[dvs].mean() # means per subject

        # single figure
        fig = plt.figure(figsize=(3,6))

        for dvi, model_dv in enumerate(dvs):
            ax = fig.add_subplot(3, 1, dvi+1)
            ax.axhline(0, lw=1, alpha=1, color = 'k') # Add horizontal line at t=0

            GROUP = DFIN.groupby(['letter', 'congruent'])[model_dv].mean() # Group average
            GROUP = GROUP.unstack()
            STDS = DFIN.groupby(['letter', 'congruent'])[model_dv].std()
            STDS = STDS.unstack()
            
            multiplier = 0 # for x indices

            for congruent in [0, 1]:

                this_cond = np.array(GROUP[congruent])
                this_std = np.array(STDS[congruent])
                SEM = np.true_divide(this_std, np.sqrt(len(subjects)))
                print(this_cond)

                offset = bar_width * multiplier
                # plot bar graph
                ax.bar(xind + offset, this_cond, yerr=SEM, width=bar_width, capsize=1, edgecolor='black', ecolor='black', color=colors[dvi], alpha=alphas[congruent], label=str(congruent))
                multiplier += 1

            # set figure parameters
            ax.set_ylabel(ylabels[dvi])
            ax.set_xlabel(xlabel)
            # ax.set_ylim(ylim)
            ax.set_xticks(xind)
            ax.set_xticklabels(xticklabels)
            ax.legend()

            sns.despine(offset=10, trim=True)
            plt.tight_layout()
            fig.savefig(os.path.join(self.figure_folder,'{}_letter_congruency_by_2d_prior.pdf'.format(self.exp)))
        print('success: plot_information_parameters_congruecy_by_frequency')