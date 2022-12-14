# Model design
import agentpy as ap
import joblib
import numpy as np
from frozendict import frozendict
import json
from matplotlib import rc

# Visualization
import seaborn as sns
import pandas as pd

import scipy.stats
import datetime
import matplotlib.pyplot as plt

from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
import networkx as nx

DEBUG = False
#from IPython import display 

ORNERY_KEYS = [ 'mean_num_txns', 
              'mean_txn_amounts',
              'agent_type_pair_probs', 'mean_txn_hrs',
              'mean_txn_amounts', 'num_agents_per_type' ]

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
            'mean_txn_amounts ': frozendict(MEAN_TXN_AMOUNTS),
            'num_agents_per_type': frozendict(NUM_AGENTS_PER_TYPE),
            'mean_txns': 4,  # avg num txns each agent makes
            'starting_balance': 100,
            'seed': 42,
            'mins_per_step': MINS_PER_STEP,  # 1 hr
            'steps': int(24 * (60/MINS_PER_STEP)),  # 24 hours * steps per hr
# hardcode, since can't give combo of options between the two
            'percent_sus': 1/100,
        }

        if param_changes!= None:
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


    def viz_and_export_network(model, viz=False):
        all_txns_2 = []
        for agent in model.agents:
            sends = agent.txns[agent.txns['txn_type'] == 'send']
            all_txns_2.append(sends)

        df_2 = pd.concat(all_txns_2)
        edges_list = df_2[['sender_id', 'receiver_id']].to_numpy()

        G=nx.DiGraph()
        G.add_edges_from(edges_list)
        print('num nodes', G.number_of_nodes())
        print('num edges', G.number_of_edges())

        counts = df_2[['sender_type', 'receiver_type', 'sender_id']]
        counts = counts.groupby('sender_id').value_counts()
        counts = counts.reset_index()
        counts = counts.rename(columns={0:'value_count'})

        counts.groupby(['sender_type', 'receiver_type']).sum().apply(np.average)
        # %% [markdown]
        # # confirm that pairs parnters are distributed correctly
        # normal-normal should be greater than normal-suspicious, etc.
        pair_cts = {}
        for s_type in counts.sender_type.unique():
            pair_cts[s_type] = {}
            for r_type in counts.receiver_type.unique():
                tmp = counts[(counts.sender_type == s_type) & (counts.receiver_type == r_type)]
                pair_cts[s_type][r_type] = np.average(tmp['value_count'])
        pair_cts = pd.DataFrame(pair_cts)

        df_2['timestep_to_time'] = df_2['timestep'].apply(Utility.timestep_to_time)

        # -- Export data!
        tabular_data = df_2[['timestep', 'timestep_to_time', 'sender_id',
                             'receiver_id', 'sender_type', 'amount'] ]
        tabular_data.to_csv('txns_list.csv', index=False)


        # -- Export more data!
        df_2[['sender_id',
              'sender_type']].drop_duplicates('sender_id').to_csv('agents_list.csv',
                                                                  index=False)

        # -- Yet more data!
        df_out_deg = pd.DataFrame(G.out_degree(), columns=['node_id', 'out_degree'])
        df_in_deg = pd.DataFrame(G.in_degree(), columns=['node_id', 'in_degree'])
        df_degs = pd.merge(df_in_deg, df_out_deg, on='node_id' )
        df_degs.to_csv(
            'tabular_graph_features.csv', index=False)

        # -- Final data export!
        df_2[['sender_id',
              'sender_type']].drop_duplicates('sender_id').to_csv('agents_list.csv',
                                                                  index=False)

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
            'mean_txn_amounts': MEAN_TXN_AMOUNTS,
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
        self.txn_amt_rng = None
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
    @staticmethod
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
    @staticmethod
    def export_data(model):
        all_txns_by_agent = []
        for agent in model.agents:
            sends = agent.txns[agent.txns['txn_type'] == 'send']
            all_txns_by_agent.append(sends)

        df = pd.concat(all_txns_by_agent)
        edges_list = df[['sender_id', 'receiver_id']].to_numpy()

        # -- Save edges to csv
        edges = pd.DataFrame(edges_list,
            columns=['nx_node_A', 'nx_node_B'])


        edges.to_csv('nx_edges_list.csv', index=False)
        #np.savetxt('nx_edges_list.csv', edges_list)

        # -- Save tabular 
        tabular_data = df[['timestep', 'timestep_to_time', 'sender_id', 'receiver_id', 'sender_type', 'amount'] ]
        tabular_data.to_csv('txns_list.csv', index=False)

        G=nx.DiGraph()
        G.add_edges_from(edges_list)
        return G


class BankExpsCollection(object):
# --- define parameters
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

            #txns_df_a = Utility.process_data_to_datetime(modelA)
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
       

# ------------Experiment
    def run_default_model(viz=False):
        model = BankModel(Utility.get_default_params())
        results = model.run()
        print('sanity check, agent 0s txns', model.agents[0].txns)
        #display.display('sanity check, agent 0s txns', model.agents[0].txns)
        if viz:
            df = Utility.process_data_to_datetime(model)
            fig = VizUtility.viz_txns_data(df) 
        return fig, model, results

# -------------------------------------------------------------

if __name__ == '__main__':
    BankExpsCollection.run_exp()
    model, results = Utility.run_default_model()
    df = Utility.process_data_to_datetime(model)
    print('viz data')
    VizUtility.viz_txns_data(df) 
    print('done')
