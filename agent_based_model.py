# Model design
import agentpy as ap
import joblib
import numpy as np
from frozendict import frozendict
import json
from matplotlib import rc
import os

# Visualization
import seaborn as sns
import pandas as pd

import scipy.stats
import datetime
import matplotlib.pyplot as plt

from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
import networkx as nx

from sklearn.ensemble import IsolationForest
from sklearn import mixture
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn import tree
from scipy.stats import ks_2samp

from sklearn.tree import export_text

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support

DEBUG = False
#from IPython import display 

ORNERY_KEYS = ['mean_num_txns', 
               'mean_txn_amounts',
               'agent_type_pair_probs', 'mean_txn_hrs',
               'mean_txn_amounts', 'num_agents_per_type']

class Utility(object):

    @staticmethod
    def process_data_to_datetime(model):
        all_txns = []
        # TODO: refactor so apply to enitre df at once
        # since only need... to... create the df all_txns
        # we can do that in the model, not as a utility method.
        # change to resample_timedata(df_all_txns)  
        for agent in model.agents:
            for timestamp in agent.send_txn_times:
                all_txns.append(Utility.timestep_to_time(timestamp))
        all_txns = pd.DataFrame( all_txns, columns=['send_txn_times' ])
        all_txns.index = all_txns.send_txn_times
        #resampled = all_txns.send_txn_times.resample('15Min')#.count().plot()
        resampled = all_txns.send_txn_times.resample('1H')
        df = pd.DataFrame(resampled.count())
        df.columns=['num_txns']
        df['labels'] = pd.to_datetime(df.index).strftime('%H:%M')
        return df

    @staticmethod
    # have to do this to go through the Experiment framework, 
    # otherwise it chokes trying to save the paramters.
    # So we flatten diciontaries to json string.
    # Then, within each BankModel init, unflatten it before using as nomral
    def get_formatted_param_for_apExperiment(param_changes=None,
                                             flatten=True): 
        # parma changes should be dict 
        # TODO: add types fo fxn calls 
        # parameters_with_dict_values'
        NUM_AGENTS_PER_TYPE = {
            'normal': 1000,
            # 'suspicious': 10, 
        }
        # these are send, rcv pairs 
        AGENT_TYPE_PAIR_PROBS = {
            'normal': {
                'self': 0.9,
                'suspicious': 0.1 },
            'suspicious': {
                'self': 0.7,
                'normal': 0.3
            } }
        MEAN_TXN_HRS = {'normal': 15,
                        'suspicious': 22}

        MEAN_TXN_AMOUNTS = {'normal': 250,
                            'suspicious': 50}  # this shoudl actually vary...
        MEAN_NUM_TXNS = { 'normal': 4, 
                          'suspicious': 10 }
        MINS_PER_STEP = 15

        parameters_exp = {
            'mean_num_txns': frozendict(MEAN_NUM_TXNS),
            'mean_txn_amounts': frozendict(MEAN_TXN_AMOUNTS),
            'agent_type_pair_probs': frozendict(AGENT_TYPE_PAIR_PROBS),
            'mean_txn_hrs': frozendict(MEAN_TXN_HRS),
            'num_agents_per_type': frozendict(NUM_AGENTS_PER_TYPE),
            'mean_txns': 4,  # avg num txns each agent makes
            'starting_balance': 100,
            'seed': 42,
            'mins_per_step': MINS_PER_STEP,  # 1 hr
            'steps': int(24 * (60/MINS_PER_STEP)),  # 24 hours * steps per hr
# hardcode, since can't give combo of options between the two
            'percent_sus': 1/100,
        }

        if param_changes is not None:
            print('\n\n!-- param changes', param_changes)
            for key, value in param_changes.items():
                print(f'{key}: modifying {parameters_exp[key]}->{value}')
                parameters_exp[key] = value
        if flatten:
            # Flag so you can get non flattened params if desired
            # E.g. for use in figure title texts
            parameters_exp = Utility.flatten_params(parameters_exp)
                # print(parameters_exp[param])
                # parameters_exp[param] = frozendict(parameters_exp[param])
        #-----------------------------------------------------------
        # --- NOTE: Setting experiment here! 
        return parameters_exp


    def network_viz(G, model):
        colors = []
        for i in range(len(G.nodes())):
            acct_type = model.agents[i].type
            G.nodes[i+1]['type'] =  model.agents[i].type
            if acct_type == 'normal' :
                colors.append('white')
            else:
                colors.append('red')
        plt.subplots()
        nx.draw(G, with_labels = True, node_color = colors, pos=nx.shell_layout(G))
        #nx.draw(G.nodes['type'] =  model.agents[i].type # todo: select by type?
        plt.show()


    @staticmethod
    def setup_p_txns(total_steps):
        # Mean at 0
        max_std_devs = 5

        txn_probabilities = np.ones(total_steps) * -1

        std_dev_per_step = 2 * max_std_devs / total_steps

        for step in range(total_steps):
            p_txn = \
                scipy.integrate.quad(
                    scipy.stats.norm.pdf,
                    -max_std_devs + step * std_dev_per_step,
                    -max_std_devs + (step+1) * std_dev_per_step
                )  # returns: (integration) value, error
            txn_probabilities[step] = p_txn[0]
            if DEBUG:
                print(step, p_txn[0])
        # The peak should be at 0, not halfway through)
        zeroing_shift = int(np.round(txn_probabilities.shape[0] / 2))
        shifted = np.concatenate(
            (txn_probabilities[-zeroing_shift:],
             txn_probabilities[:-zeroing_shift]))

        if False:
            # DEBUG: In txn time plot,
            # Should see peaks exactly at the means for each type of agent
            txn_probabilities[0] = 1  # set p_txn to 1 to always transact
            txn_probabilities[1:-1] = 0.01  # or Zero

        return shifted

    @staticmethod
    def timestep_to_time(timestep):
        mins_per_step = Utility.get_default_params('mins_per_step')
        date_and_time = datetime.datetime(2022, 10, 31, 0, 0,) # arbitrary date
        time_elapsed = timestep * (mins_per_step)
        time_change = datetime.timedelta(minutes=time_elapsed)
        new_time = date_and_time + time_change
        return new_time

    # TODO
    # -- other
    # set up agent type txn
    #     'network_randomness': 0.5
    #     'number_of_neighbors': 2,
    '''
    # -- TODO
    AGENT_TYPE_INIT_NUM_FRIENDS = {  # i guess... number of neighbors ... but ...
        'normal': '5',
        'suspicious': '2' }
    AGENT_TYPE_MAKE_NEW_FRIENDS = {
        'normal': '0.5',
        'suspicious': '0.5' }
    # -- end TODO
    '''
    #--- HACK

    #@staticmethod
    #def compare_distros():
        #ks_2samp(sender_info.txn_mean_time, sender_info.txn_mean_time)

    @staticmethod
    def flatten_params(params):#, toflatten_keys=):
        for key in ORNERY_KEYS: 
            params[key] = json.dumps(params[key])
        # cannot use Exp uless we take out dictionaries, 
        # due to bug in how parametes for ea. exp are stored
        # flatten for experiment; unflatten at actual model
        return params 

    @staticmethod
    def unflatten_params(params):
        for key in ORNERY_KEYS:
            #print('unflattening key', key, params[key], 'of type', type(key))
            params[key] = json.loads(params[key])
        return params

    @staticmethod
    def get_default_params(key=None):
        NUM_AGENTS_PER_TYPE = {
            'normal': 1000,
            'suspicious': 10, }
        # these are send, rcv pairs 
        AGENT_TYPE_PAIR_PROBS = {
            'normal': {
                'self': 0.9,
                'suspicious': 0.1 },
            'suspicious': {
                'self': 0.7,
                'normal': 0.3
            } }

        MEAN_TXN_HRS = {'normal': 14,
                        'suspicious': 22}
        MEAN_TXN_AMOUNTS = {'normal': 250,
                            'suspicious': 50}  # this shoudl actually vary...
        MEAN_NUM_TXNS = { 'normal': 4, 
                          'suspicious': 10 }
        MINS_PER_STEP = 15

        parameters = {
            'mean_num_txns': MEAN_NUM_TXNS,
            #'mean_txn_amounts': MEAN_TXN_AMOUNTS,
            'num_agents_per_type': NUM_AGENTS_PER_TYPE,
            'agent_type_pair_probs': AGENT_TYPE_PAIR_PROBS,
            'mean_txn_hrs': MEAN_TXN_HRS,
            'mean_txn_amounts ': MEAN_TXN_AMOUNTS,
            'mean_txns': 4,  # avg num txns each agent makes
            'starting_balance': 100,
            'seed': 42,
            'mins_per_step': MINS_PER_STEP,  # 1 hr
            'steps': int(24 * (60/MINS_PER_STEP)),  # 24 hours * steps per hr
            'percent_sus':0.01
        }
        if key is not None:
            return parameters[key]
        return parameters


class BankAgent(ap.Agent):
    def setup(self):
        self.type = None
        self.pair_amts = None

        # -- param for who to partner
        self.pair_probs = None
        # -- param for whether to txn
        self.mean_num_txns = None
        self.txn_probabilities = None
        self.send_txn_times = []
        # -- store info
        self.txns = None
        self.txns_list = []
        # -- parmeters for amt $$$
        self.txn_amt_rng = None #!
        self.acct_balance = None
        self.txn_amts = None

    def setup_txn_amts(self, mean=0, stddev=1, total_steps=0):
        #print('total steps', total_steps)
        self.txn_amts = self.txn_amt_rng.normal(
            loc=mean, scale=stddev,
            size=self.total_steps)
        #print(self.txn_amts)
        self.txn_amts = np.abs(self.txn_amts)

    # NOTE: Timesteps start at 1 # TODO: fix bug. where first timestep should be at midnight.
    # TODO: fix so stored txn time can be timestep, not timestep-1
    # TOOD: issue currently is graph has both midnight 10/31 and midngith 11/1
    # TODO: due to resample rounding up or down or something idk
    def transact(self, timestep):
        # using as index, subtract (timestep starts at 1)
        p_txn = self.txn_probabilities[timestep-1] * self.mean_num_txns

        if np.random.random() < p_txn and self.acct_balance > 0:
            # using as value, do nothing
            self.send_txn_times.append(timestep)
            # randomly choose partner (w. probability per parameters)
            if np.random.random() < self.pair_probs['self']:
                my_partner = self.model.agents.select(
                    self.model.agents.type == self.type).random()
            else:  # TODO: can only accept two agent types at the moment, using !=
                my_partner = self.model.agents.select(
                    self.model.agents.type != self.type).random()

            my_partner = my_partner.to_list()[0]
            # randomly choose amount
            # right now, only scaled by sender
            # so normal will still send large amount to suspicious
            # TODO: vary by receiver also
            amount = self.txn_amts[timestep-1] * self.pair_amts  # [self.type]

            # calculations
            my_partner.acct_balance += amount
            self.acct_balance -= amount

            if DEBUG:  # DEBUG
                print(
                    f' Transaction @ step: {timestep} with ptxn: {p_txn:.2f}, '
                    f'${amount:.2f} from: {self.id}->{my_partner.id}, '
                    f'({self.type} to {my_partner.type}), new balance: {self.acct_balance:.1f}'
                )

            # Note that txn_type is redundant info (e.g. can be derived from +/- of amount)
            self.txns_list.append([timestep, 'send', self.id, self.type,
                                   my_partner.id, my_partner.type, -amount, self.acct_balance])
            # -- also record in partner's txn table
            my_partner.txns_list.append([
                timestep, 'receive', self.id, self.type,
                my_partner.id, my_partner.type, amount, my_partner.acct_balance])

    def cleanup(self):
        self.txns = pd.DataFrame(self.txns_list, columns=['timestep',
                                                          'txn_type',
                                                          'sender_id',
                                                          'sender_type',
                                                          'receiver_id',
                                                          'receiver_type',
                                                          'amount',
                                                          'acct_balance',])


class BankModel(ap.Model):

    def setup(self):
        self.p = Utility.unflatten_params(self.p)
        self.p_txns = Utility.setup_p_txns(self.p.steps)

        # for experiment, vary percent suspicious
        # NOTE: hackish workaround for now to get % as since var
        self.p.num_agents_per_type['suspicious'] = \
            int(self.p.num_agents_per_type['normal'] * self.p.percent_sus)

        num_agents = sum(self.p.num_agents_per_type.values())

        # Setup up rng to generate seeds for rngs for agents

        rng = np.random.default_rng(self.p.seed)
        agent_rng_seeds = rng.random(num_agents).round(3) * 1000
        agent_rng_seeds = np.array(agent_rng_seeds, dtype=int)

        self.agents = ap.AgentList(self) # empty list 
        print(self.p.num_agents_per_type)
        
        # -- shift the probabilitiy tables
        for type in ['normal', 'suspicious']:
            hrs_mean = self.p.mean_txn_hrs[type]
            # shift in steps
            # originally peak at 0, now peak at (mean time)
            shift_amt = hrs_mean * int(60 / self.p.mins_per_step)
            shifted = np.concatenate(
                (self.p_txns[-shift_amt:], self.p_txns[:-shift_amt]))
            agents = ap.AgentList(
                self, self.p.num_agents_per_type[type], BankAgent)
            pair_probs = self.p.agent_type_pair_probs[type]
            agents.mean_num_txns = self.p.mean_num_txns[type]
            agents.pair_probs = pair_probs
            # todo: not paired atm
            agents.pair_amts = self.p.mean_txn_amounts[type]
            agents.txn_probabilities = shifted
            agents.total_steps = self.p.steps
            agents.acct_balance = 100
            agents.type = type

            self.agents += agents

        #if DEBUG:
            #print('here are all the agetns: ')
            #print(self.agents)
            #print([(agent.id, agent.type) for agent in self.agents])

        # -- calcuations for txn $$$ (each agent gets different seed)
        txn_amt_rngs = ap.AttrIter(
            [np.random.default_rng(seed) for seed in agent_rng_seeds])
        self.agents.txn_amt_rng = txn_amt_rngs

        for agent in self.agents:
            agent.setup_txn_amts(
                mean=20, stddev=5, total_steps=self.p.steps)

    def step(self):
        self.agents.transact(self.t)

    def update(self):
        pass
    '''
    if DEBUG:
            total_txns = 0
            for agent in self.agents:
                num_txns = agent.txns[agent.txns.txn_type == 'send'].shape[0]
                total_txns += num_txns
            print('num txns (across all agents)', total_txns)
    '''

    def end(self):
        # i think this includes the final timestep t = 96 as entire column
        self.agents.record('send_txn_times')

        # export data
        for agent in self.agents:
            agent.cleanup()


'''
def debug_printouts():
    DEBUG = False
    with np.printoptions(precision=3, suppress=True):
        test = Utility.setup_p_txns(96)
        # display.display(test)
        print(test)

    if DEBUG:
        # Note: scipy norm has stddev = 1, mean of 0
        # Below shows why we pick max_std_devs = 5
        p_txn = scipy.integrate.quad(
            scipy.stats.norm.pdf, -5, 5)[0]  # quad() Returns: value, error
        print('In total we will capture this amount of the gaussian: ', p_txn)

    # if DEBUG:
        # TODO: define model for thi code to work (pass from bankmodel)
        # for agent in model.agents:
        # print(agent.txns)
    # display(model.agents[0].txns)
    # https://github.com/JoelForamitti/agentpy/blob/master/tests/test_sequences.py
    tmp_model = BankModel()
    agents = ap.AgentList(tmp_model, 10, BankAgent)
    agents.random(5).type = 'test'
    agents.id
    for agent in agents:
        print(agent.id, agent.type)
    agents.select(agents.type == 'test').random().id

    model = BankModel(Utility.get_default_params())
    results = model.run()
    return results
'''
    
class VizUtility(object): 
#    @staticmethod
#   def format_and_viz_isolation

    @staticmethod
    def format_and_viz_tree(df_txns, title='Outlier Detection'):
        # -- this is one of two trees
        # -- Defensively clear, then set up style
        plt.rcParams.update(plt.rcParamsDefault)
        sns.reset_defaults()
        sns.set_context('notebook')
        sns.set_style('whitegrid')

        # -- Plot data
        fig, ax = plt.subplots() # TODO: fix figsize to reasonable vals
        sns.stripplot(data=df_txns, x='timestep', y='sender_type',
                      hue='y_pred',
                      edgecolor='k', linewidth=.2)
        print(df_txns.sender_type.sample())

        # --- Format nicely
        plt.title(
            'Simulated Transaction Times by Sender Type\n' \
            'Predicted vs Real Label with Decision Tree')
        plt.ylabel('True Sender Type')
        plt.xlabel('Transaction Timestep')
        #ax.legend(title='Predicted Type', labels=['normal','suspicious'])
        plt.legend(title='Predicted Type',
                   labels=['normal','suspicious'],
                  loc='center right')
        plt.tight_layout()

        fig.patch.set_facecolor('#F9F3DC')
        fig.savefig('fig2_dt.pdf') # This is just to show the figure is still generated

        return fig

    @staticmethod
    def format_and_viz_isolat(df_txns, title='Outlier Detection'):
        # -- Defensively clear, then set up style
        plt.rcParams.update(plt.rcParamsDefault)
        sns.reset_defaults()
        sns.set_context('notebook')
        sns.set_style('whitegrid')

        # -- Plot data
        fig, ax = plt.subplots()
        df_txns['normalized_y_pred'] = df_txns['y_pred'] == -1
        sns.stripplot(data=df_txns, x='timestep', y='sender_type',
                      hue='normalized_y_pred',
                      edgecolor='k', linewidth=.2)

        # --- Format nicely
        plt.title(
            'Simulated Transaction Times by Sender Type\n' \
            'Predicted vs Real Label with Isolation Forest') 
        plt.ylabel('True Sender Type')
        plt.xlabel('Transaction Timestep')
        plt.legend(title='Predicted Type', labels=['normal','suspicious'],
                   loc='center right')
        plt.tight_layout()

        fig.patch.set_facecolor('#F9F3DC')
        fig.savefig('fig1_isolat.pdf') # This is just to show the figure is still generated
        return fig

    @staticmethod
    def format_and_viz_isolat_hist(df_txns):
        # -- Defensively clear, then set up style
        plt.rcParams.update(plt.rcParamsDefault)
        sns.reset_defaults()
        sns.set_context('notebook')
        sns.set_style('whitegrid')

        fig, (ax1, ax2) = plt.subplots(2,1, sharex=True)
        #  isolation forest is 1, and -1 for outlier
        df_txns['normalized_y_pred'] = df_txns['y_pred'] == -1
        sns.histplot(data=df_txns, x='timestep', kde=True,
                     hue='normalized_y_pred', 
                     ax=ax1).set(title='Predicted')
        sns.histplot(data=df_txns, x='timestep', kde=True, hue='y_true',
                     ax=ax2).set(title='True')

        ax1.legend(title='Agent Type', labels=['suspicious','normal'])
        ax2.legend([],[], frameon=False)

        plt.suptitle('Histogram of Transactions\n'
                     '(Labelled by Isolation Forest)')
        plt.xlabel('Transaction Timestep')
        #plt.legend(title='Predicted Type', labels=['normal','suspicious'])
        plt.tight_layout()

        fig.patch.set_facecolor('#F9F3DC')
        fig.savefig('fig2_GMM.pdf') # This is just to show the figure is still generated
        return fig

    @staticmethod
    def format_and_viz_GMM(df_txns, title='Outlier Detection'):
        # -- Defensively clear, then set up style
        plt.rcParams.update(plt.rcParamsDefault)
        sns.reset_defaults()
        sns.set_context('notebook')
        sns.set_style('whitegrid')

        # -- Plot data
        fig, ax = plt.subplots()
        sns.stripplot(data=df_txns, x='timestep', y='sender_type',
                      hue='y_pred',
                      edgecolor='k', linewidth=.2)

        # --- Format nicely
        plt.title(
            'Simulated Transaction Times by Sender Type\nPredicted vs Real Label with Gaussian Mixture Model')
        plt.ylabel('True Sender Type')
        plt.xlabel('Transaction Timestep')
        plt.legend(title='Predicted Type', labels=['normal','suspicious'],
                   loc='center right')
        plt.tight_layout()

        fig.patch.set_facecolor('#F9F3DC')
        fig.savefig('fig1_GMM.pdf') # This is just to show the figure is still generated

        return fig

    @staticmethod
    def format_and_viz_GMM_hist(df_txns):
        # -- Defensively clear, then set up style
        plt.rcParams.update(plt.rcParamsDefault)
        sns.reset_defaults()
        sns.set_context('notebook')
        sns.set_style('whitegrid')

        fig, (ax1, ax2) = plt.subplots(2,1, sharex=True)
        df_txns['inverted_y_pred'] = 1 - df_txns['y_pred']
        sns.histplot(data=df_txns, x='timestep', kde=True,
                     hue='y_pred', 
                     ax=ax1).set(title='Predicted')
        sns.histplot(data=df_txns, x='timestep', kde=True, hue='y_true',
                     ax=ax2).set(title='True')

        ax2.legend([],[], frameon=False)
        ax1.legend(title='Agent Type', labels=['suspicious','normal'])

        plt.suptitle('Histogram of Transactions\n'
                     '(Labelled by Gaussian Mixture Model)')
        plt.xlabel('Transaction Timestep')
        #plt.legend(title='Predicted Type', labels=['normal','suspicious'])
        plt.tight_layout()

        fig.patch.set_facecolor('#F9F3DC')
        fig.savefig('fig2_GMM.pdf') # This is just to show the figure is still generated
        return fig


    def viz_txns_data():
        pass

    @staticmethod
    def format_fig_1():
        plt.rcParams.update(plt.rcParamsDefault)
        sns.reset_defaults()
        plt.style.use('bmh')     # switch to seaborn style
        sns.set_style('ticks')
        sns.set_context('notebook')
        #sns.set_style('whitegrid')
        fig, (ax1, ax2) = plt.subplots(1,2,
                                        figsize = (8,4),
                                        sharey=True)
        fig.patch.set_facecolor('#F9F3DC')
        return fig, (ax1, ax2)
                
    @staticmethod
    def format_fig_2():
        plt.rcParams.update(plt.rcParamsDefault)
        sns.reset_defaults()

        #plt.style.use('bmh')     # switch to seaborn style
        #sns.set_style('ticks')
        sns.set_style('whitegrid')
        sns.set_context('notebook')

        #rc('font',**{'family':'serif','serif':['CMU Serif']})
        #rc('font',**{'family':'sans serif','serif':['CMU Sans Serif']})
        #plt.rcParams.update({
            #"font.family": "serif"
        #})

        fig, axes = plt.subplots(1,3, 
                                 figsize = (8,4), sharey=True)

        fig.patch.set_facecolor('#F9F3DC')
        return fig, axes 
                
    @staticmethod
    def viz_fig_1(df1, df2, model1, model2):
        '''
        Input: df which has columns, send_txn_times, and num_txns
        Note: This fxn needed because it's intensive to draw timestamps in
        HH:MM directly
        '''
        #parameters = Utility.get_formatted_param_for_apExperiment(flatten=False)
        # TODO: this is a terrible way using lists to allow for not
        # dupcliating chart code, refactor eventually
        fig, (ax1, ax2) = VizUtility.format_fig_1()
        df = [df1, df2]
        hrs = [model1.p['mean_txn_hrs'], model2.p['mean_txn_hrs']]
        models = [model1, model2]
        axes = [ax1, ax2]

        for i, (model, ax) in enumerate( zip(models, axes) ):
            sns.lineplot(x='send_txn_times', y='num_txns', data=df[i],
                         ax=ax, markers=True,  marker='o')

            ax.set(xlabel='Time (24 Hour)', ylabel='# of Transactions',
                title=\
                   f"Mean txn time, "
                   rf"Normal: $\bf{hrs[i]['normal']}$00,"
                   f" Suspicious: {hrs[i]['suspicious']}00"
                )
            ax.set_xticklabels(ax.get_xticks(), rotation = 40)
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
            ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))
            ax.xaxis.set_major_formatter(DateFormatter("%H:%M"))

        num_agents = model1.p['num_agents_per_type']
        print('num agents', num_agents, model2.p['num_agents_per_type'])
        plt.suptitle(r"$\bf{Simulated\ Transactions\ by\ Time–of–Day}$"
            f"\n# Accounts, Normal: {num_agents['normal']}, "
            f"Suspicious: {num_agents['suspicious']}")

        plt.tight_layout()
        fig.savefig('fig_1_plot.pdf') # This is just to show the figure is still generated
        #plt.show()
        return fig

    @staticmethod
    def viz_fig_2(models, txns):
        '''
        Input: df which has columns, send_txn_times, and num_txns
        Note: Here we have to fit 3 plots side-by-side, so vary the tick
        intervals compared to fig. 1 (also the figsize)
        '''
        #parameters = Utility.get_formatted_param_for_apExperiment(flatten=False)
        # TODO: this is a terrible way using lists to allow for not
        # dupcliating chart code, refactor eventually
        fig, axes = VizUtility.format_fig_2()
        percents = [model.p['percent_sus'] for model in models]
        num_normal = models[0].p['num_agents_per_type']['normal']

        for i in range(len(models)):
            ax = axes[i]
            percent_sus = percents[i]

            sns.lineplot(x='send_txn_times', y='num_txns', data=txns[i],
                         ax=ax, markers=True,  marker='o')

            ax.set(xlabel='Time', ylabel='# of Transactions',
                title=\
                   f"{percent_sus*100}% = "
                   f"{int(percent_sus*num_normal)} Suspic. Accts."
                )
            ax.set_xticklabels(ax.get_xticks(), rotation = 40)
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))
            ax.xaxis.set_minor_locator(mdates.HourLocator(interval=2))
            ax.xaxis.set_major_formatter(DateFormatter("%H:%M"))
#            ax.tick_params(axis='both', which='both', labelbottom=True)

        plt.suptitle(
            r"$\bf{Simulated\ Transactions\ by\ Time–of–Day}$"\
            f"\n# Accounts, Normal: {num_normal}")

        plt.tight_layout()
        fig.subplots_adjust(wspace=0.05, hspace=0)
        fig.savefig('fig_2_plot.pdf') # This is just to show the figure is still generated
        #plt.show()
        return fig


'''
fig, ax = plt.subplots()
np.random.seed(123)
sns.reset_defaults()
#sns.set_theme(palette='viridis')
sns.set_context('notebook')
#sns.set_context('talk')
sns.set_style('whitegrid')
#sns.set(rc={'axes.facecolor':'palegoldenrod', 'figure.facecolor':'white'})
sns.stripplot(data=txns, x='timestep', y='sender_type', hue='y_pred', edgecolor='k', linewidth=.2)#, palette='')
fig.patch.set_facecolor('#F9F3DC')
'''

class ExportUtility():
    # -- Note: fixed filenames ! 
    @staticmethod
    def export_data(model):
        # -- collate by agent
        # -- TODO: avoid iterate through dataframe... sorry...
        all_txns = []
        for agent in model.agents:
            sends = agent.txns[agent.txns['txn_type'] == 'send']
            all_txns.append(sends)

        df_txns = pd.concat(all_txns)
        df_txns['timestep_to_time'] = df_txns['timestep'].apply(
            Utility.timestep_to_time)
        # TODO: there seems to be mismatch in len between number of agents
        # here vsin the other data files below

        # --------------------------------------------
        # -- Export normal txns data (no sender type)
        tabular_data = df_txns[[
            'timestep', 'timestep_to_time', 'sender_id', 'receiver_id', 
            'sender_type', 'amount'] ]
        tabular_data.to_csv('txns_list.csv', index=False)

        # --------------------------------------------
        # -- Export true sender identity  (sender type)
        labelled_agents = df_txns[['sender_id', 
                 'sender_type'
                ]].drop_duplicates(['sender_id', 'sender_type']
                                  )
        print('agents labels shape', labelled_agents.shape)
        labelled_agents.to_csv('agents_list.csv', index=False)
        print('vs txn agents', df_txns.sender_id.unique().shape)

        

        # --------------------------------------------
        # -- export counts txns per agent
        #counts = df_txns[['sender_type', 'receiver_type', 'sender_id']]
        #counts = counts.groupby('sender_id').value_counts()
        #counts = counts.reset_index()
        #counts = counts.rename(columns={0:'value_count'})
        
        # --------------------------------------------
        # -- Export edges (senders and receiver node ids)
        edges_list = df_txns[['sender_id', 'receiver_id']].to_numpy()
        df_edges = pd.DataFrame(edges_list,
                             columns=['nx_node_A', 'nx_node_B'])
        df_edges.to_csv('nx_edges_list.csv', index=False)

        # --------------------------------------------
        # -- Export in/out degree network data
        # -- TODO: this code is a bit wordy, can be simplified 
        G = nx.DiGraph()
        G.add_edges_from(edges_list)

        _out_deg = pd.DataFrame(G.out_degree(), 
                                  columns=['node_id', 'out_degree'])
        _in_deg = pd.DataFrame(G.in_degree(), 
                                 columns=['node_id', 'in_degree'])
        df_degs = pd.merge(_in_deg, _out_deg, on='node_id' )
        df_degs.to_csv(
            'tabular_graph_features.csv', index=False)
        # --------------------------------------------
        # -- Export in/out degree network data
        joblib.dump(G, 'graph.networkx_v2.8.4.joblib') 
        return G


class BankExpsCollection(object):
# --- define parameters
    @staticmethod
    def gen_data_for_outlier_classif(check_filename='txns_list.csv'):
        if os.path.exists(check_filename):
            print(f'file {check_filename} exists, not regenerating')
        else:
            # ------- First Run experiment  to generate data
            # set params and then ... run the classifier 
            param_changes = {
                'mean_txn_hrs': {'normal': 12, 'suspicious': 22},
                'num_agents_per_type': {'normal':1000}}
            params = Utility.get_formatted_param_for_apExperiment(
                param_changes)

            params['percent_sus'] = 1/100
            model = BankModel(params)
            model.run()

            # ------- Then export the data  
            ExportUtility.export_data(model)


    @staticmethod
    def gen_fig_3():
        # See Outlier Detection class
        pass

    @staticmethod
    def gen_fig_1():
        # run it twice, with different mean txn times
        # params A and B

        param_changes = {
            'mean_txn_hrs': {'normal': 12, 'suspicious': 22},
            'num_agents_per_type': {'normal':10000}}
        paramsA = Utility.get_formatted_param_for_apExperiment(
            param_changes)
        paramsA['percent_sus'] = 1/50 
        modelA = BankModel(paramsA)
        results = modelA.run()
        results.save()
        txns_df_a = Utility.process_data_to_datetime(modelA)
        
        # cannot modify param directly.
        # have to reformat, since we're changing a parameters that is a
        # dictionary which breaks the Experiment code...
        param_changes = {'mean_txn_hrs': {'normal': 17, 'suspicious': 22},
            'num_agents_per_type': {'normal':10000}}
        paramsB = Utility.get_formatted_param_for_apExperiment(
            param_changes)
        paramsB['percent_sus'] = 1/50
        modelB = BankModel(paramsB)
        results = modelB.run()
        txns_df_b = Utility.process_data_to_datetime(modelB)
        #fig = viz_txns_data(df) 
        fig = VizUtility.viz_fig_1(txns_df_a, txns_df_b, modelA, modelB)
        return fig

    # fig 2: txns as vary by % suspicious

    @staticmethod
    def gen_fig_2():
        # for figure 2, we hold number of agetns and mean hrs constant
        # but we vary the % suspicious
        param_changes = {
            'mean_txn_hrs': {'normal': 12, 'suspicious': 22},
            'num_agents_per_type': {'normal':10000}}
        params = Utility.get_formatted_param_for_apExperiment(
            param_changes)

        models = [] 
        txns = []
        #list_results = []
        for percent in [1/10, 1/100, 1/1000]:
            params['percent_sus'] = percent
            model = BankModel(params)
            # results = model.run()
            # results.save()
            model.run()
            models.append(model)
            #list_results.append(results)
            txns.append(Utility.process_data_to_datetime(model))

        fig = VizUtility.viz_fig_2(models, txns)
        return fig


    # ------------Experiment
    # --- Old -- I think completely replaced by run_experiment(viz=False)
    #def run_default_model(viz=False):
    #    model = BankModel(Utility.get_default_params())
    #    results = model.run()
    #    print('sanity check, agent 0s txns', model.agents[0].txns)
    #    #display.display('sanity check, agent 0s txns', model.agents[0].txns)
    #    if viz:
    #        df = Utility.process_data_to_datetime(model)
    #        fig = VizUtility.viz_txns_data(df) 
    #    return fig, model, results

    @staticmethod
    def run_experiment(viz=False):
        parameters_exp = Utility.get_formatted_param_for_apExperiment()
        parameters_exp['percent_sus'] = ap.Values(1/10, 1/100, 1/1000)

        sample = ap.Sample(parameters_exp) # grid search, each repeat 1x
        print('created sample; ', sample)

        # -- TEMP TEST 
        if False:
            model = BankModel(parameters_exp)
            # or at least try default params
            # model = BankModel(Utility.get_default_params())
            results = model.run()
            print('finished test run with default params')

        exp = ap.Experiment(BankModel, parameters_exp)
        print('created exp; ', exp)

        results = exp.run()
        print('ran exp; ', results)
        joblib.dump(results,
                    f"./results/{results.info['time_stamp'][:19]}.joblib")
        # results.save() # 'TypeError: Object of type Values is not JSON serializable'
        fig = None 
        model = None
        if viz:
            df = Utility.process_data_to_datetime(model)
            print('viz data')
            fig = VizUtility.viz_txns_data(df) 
            print('done')
        return fig, model, results


# -- Note: no data generated in this class; use BankExp class for that
class OutlierDetection():
        #true_agent_labels = pd.read_csv('agents_list.csv')
        #true_agent_labels.columns=['sender_id', 'true_sender_type']

    @staticmethod
    def create_1d_X_from_files():
        # TODO: save paremtesr of data also
        # Create data to save
        try:
            df_txns = pd.read_csv('./txns_list.csv')
        except FileNotFoundError:
            print('File (txns_list.csv) not found; have you run' \
                  'BankExpsCollection.gen_data_for_outlier_classif()' \
                  'before this?')
        # -- Rename truth values (txn labels) from string to integer 
        df_txns['y_true'] = df_txns.sender_type.apply(
            lambda x: 1 if x=='suspicious' else 0)
        # -- Format datetimes # TODO:  timestep_to_time is a column not a fxn, fix this naming
        df_txns['time'] = df_txns.timestep_to_time.apply(pd.to_datetime)
        # assert(df_txns.time.dtype == np.datetime64) 
        # -- Reshape for sklearn classifiers, which don't expect 1d data
        X = df_txns.timestep.to_numpy().reshape(-1,1)
        # Xdeg2 = np.hstack((X, X**2))
        return X, df_txns
    
    def create_4d_X_from_files():
        true_agent_labels = pd.read_csv('agents_list.csv')
        true_agent_labels.columns=['sender_id', 'true_sender_type']

        edges = pd.read_csv('nx_edges_list.csv')
        node_degs = pd.read_csv('tabular_graph_features.csv')
        #node_degs.columns=['sender_id', 'in_degree', 'out_degree']
        return edges, node_degs

    '''
    def calc_in_out_deg():
        sender_info = sender_info.merge(node_degs)
        sender_info['y_pred_type'] = sender_info.y_pred.apply(lambda x: 'normal' if x == 1 else 'suspicious')
        X = sender_info[['txn_mean_time' , 'in_degree', 'out_degree']]

        sns.pairplot(sender_info[['txn_mean_time', 'in_degree', 'out_degree']])

        '''

    # -- Unsupervised
    @staticmethod
    def gen_gaussian_mixture_figs():
        X, df_txns = OutlierDetection.create_1d_X_from_files()
        # -- Train model

        clf = mixture.GaussianMixture(n_components=2,
                                      covariance_type="full",
                                      random_state = 123)
        y_pred = clf.fit_predict(X)
        # -- use dataframe to store y_pred for ease of seaborn plotting
        df_txns['y_pred'] = y_pred
        fig1 = VizUtility.format_and_viz_GMM(df_txns)
        fig2 = VizUtility.format_and_viz_GMM_hist(df_txns)
        return clf, fig1, fig2

    def gen_tree_figs(max_depth=1):
        # -- fig 1 defined here
        X, df_txns = OutlierDetection.create_1d_X_from_files()
        print(df_txns.y_true.sample(4))
        # -- Classify with decision tree
        clf = tree.DecisionTreeClassifier(max_depth=max_depth,
                                          random_state=123,
                                          )
        clf = clf.fit(X, df_txns.y_true)
        y_pred = clf.predict(X)

        # -- Defensively clear, then set up style
        plt.rcParams.update(plt.rcParamsDefault)
        sns.reset_defaults()
        sns.set_context('notebook')
        #sns.set_style('darkgrid')

        plt.subplots(figsize=(10,10))
        # -- plot the tree
        fig1 = tree.plot_tree(clf,
                             feature_names=['Txn Timestep'],
                             label='all',
                             impurity=False,
                             class_names=['Normal', 'Suspicious'],
                             #filled=True, 
                             rounded=False)
        plt.title('Decision Tree\n(Value = # agents in each class)')
        plt.savefig('fig1_dt.pdf') # This is just to show the figure is still generated
        if True:
            plt.show() #--- OTHERWISE DOES NOT SHOW :( TODO FIX THIS

        # -- inspect jitter plot
        df_txns['y_pred'] = y_pred
        fig2 = VizUtility.format_and_viz_tree(df_txns)

        # -- print text version of the decision tree
        text_tree = export_text(clf, feature_names=['agent type'],
                        show_weights=True)
        print('\n---- Printout of decision tree in text')
        print(text_tree)

        return (fig1, fig2)


    # -- unsupervised
    def gen_isolation_figs():
        X, df_txns = OutlierDetection.create_1d_X_from_files()
        # -- Train model
        clf = IsolationForest(contamination=0.1,
                              random_state = 123).fit(X)
        y_pred = clf.fit_predict(X)
        # -- use dataframe to store y_pred for ease of seaborn plotting
        df_txns['y_pred'] = y_pred
        fig1 = VizUtility.format_and_viz_isolat(df_txns)
        fig2 = VizUtility.format_and_viz_isolat_hist(df_txns)
        return fig1, fig2

    '''
    def use_isolation_forest():
        np.random.seed(123)
        clf = IsolationForest(random_state=0, contamination=0.1).fit(Xdeg2)
        y_pred = clf.predict(Xdeg2)
#y_pred = IsolationForest(random_state=0, contamination=0.1).fit_predict(X)

        txns['y_pred'] = y_pred
        null_acc = accuracy_score(txns.y_true, np.zeros(txns.y_true.shape[0])) # ytrue, ypred # predict all 0s (majority class)
        null_acc = accuracy_score(txns.y_true, np.zeros(txns.y_true.shape[0])) # ytrue, ypred # predict all 0s (majority class)
        acc = accuracy_score(txns.y_true, txns.y_pred==-1 )  # outlier is -1, wjhich is 1 in the other labeling
        sns.stripplot(data=txns, x='timestep', y='sender_type', hue='y_pred', )
        plt.title(f'Predicted vs Real Label with Isolation Forest Model\n10% contamination, Label -1 = suspicious\n acc = {acc*100:.2f}%, null acc = {null_acc*100:.2f}%')
        #plt.show()

        sender_info['y_pred'] = y_pred
        colors = ['r' if label==0 else 'b' for label in y_pred]
        #-- plot disaggregated by true label
        plt.subplots()
        sns.stripplot(data=sender_info, x='txn_mean_time', y='true_sender_type', hue='y_pred_type', )
        plt.title('Predicted vs Real Label with Isolation Forest\n Including In/Out degree features')

        plt.show()
        '''

    def viz_decision_bounds():
        pass


        '''
        def group_txns_by_agent():
        if False:
            print(
                txns.groupby(['sender_id'])[
                            ['sender_id', 'timestep', 'y_pred', 'y_true']
                            ].value_counts()
            )

        if False:
            # I guess this is for looking at how many mismatches there were
            # for some given prediction
            # See: y_pred
            txns_by_id = txns[['sender_id' , 'timestep', 'y_pred', 'y_true']]
            txns_by_id = txns_by_id.pivot(index='sender_id',
                                          columns='timestep', values='y_true')
            txns_by_id['sum'] = txns_by_id.sum()

            txns_by_id.reset_index()[['sender_id', 'sum']].fillna(0)
            txns_by_id['agent_label'] = txns_by_id['sum'] >= 1

            pred_by_agent = txns_by_id[['agent_label']] * 1
            errors = pred_by_agent.reset_index().merge(true_agent_labels)
            errors['wrong_pred'] = errors.agent_label == errors.true_label
            errors.wrong_pred.sum() / errors.shape[0]

        # -- rename values from normal/suspicious to 0/1 respectively
        true_agent_labels['true_label'] = \
            true_agent_labels['true_sender_type'] != 'normal'
        true_agent_labels['true_label'] *= 1
        return true_agent_labels

    def per_timestep():
        sender_info['txns'] = None
        sender_info['txn_mean_time'] = None
# convert datatype to numpy array
        sender_info.txns = sender_info.txns.astype(object)
        display(sender_info.sample())


        all_my_txns = []
#for id in [1,2]:
        for id in sender_info.sender_id:
            my_txns = txns[txns.sender_id == id].timestep
            #print(my_txns.to_list())
            all_my_txns.append(my_txns.to_list())
        sender_info['txns'] = pd.Series(all_my_txns)
        display(sender_info.iloc[1])
        sender_info['txn_mean_time'] = sender_info['txns'].apply(np.mean)
        # sender_info['label_by_mean_txn_time'] = sender_info

    def viz_by_sender():
        fig, ax = plt.subplots()
        plt.set_xlim = [0,100]
        sns.histplot(sender_info['txn_mean_time'], ax=ax, kde=True , bins=50)
#sns.kdeplot(sender_info['txn_mean_time'], ax=ax)
        plt.show()

        plt.subplots()
        plt.xlim = [0,100]
        sns.histplot(txns.timestep, kde=True)
        plt.show()

        sender_info = sender_info.merge(true_agent_labels)

        sns.stripplot(sender_info['txn_mean_time'],)#, type=)
        sns.histplot(sender_info['txn_mean_time'])#, type=)

        # plot with true heu
        sns.histplot(data=sender_info, x='txn_mean_time', hue='true_sender_type')
    '''

# -------------------------------------------------------------

if __name__ == '__main__':
    BankExpsCollection.run_exp()
    model, results = Utility.run_default_model()
    df = Utility.process_data_to_datetime(model)
    print('viz data')
    VizUtility.viz_txns_data(df) 
    print('done')
