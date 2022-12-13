# Model design
import agentpy as ap
import numpy as np
from frozendict import frozendict

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


class Utility(object):

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
        parameters = Utility.get_default_params()
        date_and_time = datetime.datetime(2022, 10, 31, 0, 0,)
        time_elapsed = timestep * (parameters['mins_per_step'])
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
    def flatten_params(params):
        # cannot use Exp uless we take out dictionaries, 
        # due to bug in how parametes for ea. exp are stored
        # flatten for experiment; unflatten at actual model
        parameters = {
            'mean_num_txns': MEAN_NUM_TXNS,
            'mean_txn_amounts': MEAN_TXN_AMOUNTS,
            'num_agents_per_type': NUM_AGENTS_PER_TYPE,
            'agent_type_pair_probs': AGENT_TYPE_PAIR_PROBS,
            'mean_txn_hrs': MEAN_TXN_HRS,
            'mean_txn_amounts': MEAN_TXN_AMOUNTS,
            'mean_txns': 4,  # avg num txns each agent makes
            'starting_balance': 100,
            'seed': 42,
            'mins_per_step': MINS_PER_STEP,  # 1 hr
            'steps': int(24 * (60/MINS_PER_STEP)),  # 24 hours * steps per hr
        }
        # NOTE: TMEPORARY: for debugging
        #parameters['percent_sus'] = 0.01
        return f_params 

    @staticmethod
    def unflatten_params(f_params):
        pass


    @staticmethod
    def get_default_params():
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
        print(parameters)
        # NOTE: TMEPORARY: for debugging
        #parameters['percent_sus'] = 0.01
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
        self.p_txns = Utility.setup_p_txns(self.p.steps)
        print('setting up')
        print('\n\n!------ self.p')
        print(self.p)
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

        if DEBUG:
            print('here are all the agetns: ')
            print(self.agents)
            print([(agent.id, agent.type) for agent in self.agents])

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

    
def process_data(model):
    all_txns = []
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

def viz_data(df):
    fig, ax = plt.subplots()
    sns.lineplot(x='send_txn_times', y='num_txns', data=df, ax=ax,
        markers=True,  marker='o')
    # --
    # format
    parameters = Utility.get_default_params()
    ax.set(xlabel='Time (24 Hour)', ylabel='# of Transactions',
    # add title
    title=r"$\bf{Simulated\ Transactions\ by\ Time–of–Day}$"
        f"\n# Accounts, Normal: {parameters['num_agents_per_type']['normal']}, "
        f"Suspicious: {parameters['num_agents_per_type']['suspicious']}\n"
        f"Mean txn time, Normal: {parameters['mean_txn_hrs']['normal']}:00,"
        f" Suspicious: {parameters['mean_txn_hrs']['suspicious']}:00"
        )

    ax.set_xticklabels(ax.get_xticks(), rotation = 40)
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))
    ax.xaxis.set_major_formatter(DateFormatter("%H:%M"))
    plt.tight_layout()
    fig.savefig('plot.pdf') # This is just to show the figure is still generated
    #plt.show()
    return fig

def viz_network(model):
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

def network_viz(G):
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

# --- define parameters
def run_custom_exp(viz=False): 
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

    MEAN_TXN_HRS = {'normal': 14,
                    'suspicious': 22}

    MEAN_TXN_AMOUNTS = {'normal': 250,
                        'suspicious': 50}  # this shoudl actually vary...

    MEAN_NUM_TXNS = { 'normal': 4, 
                      'suspicious': 10 }
    MINS_PER_STEP = 15


    parameters_exp = {
        'mean_num_txns': MEAN_NUM_TXNS,
        'mean_txn_amounts': MEAN_TXN_AMOUNTS,
        'agent_type_pair_probs': AGENT_TYPE_PAIR_PROBS,
        'mean_txn_hrs': MEAN_TXN_HRS,
        'mean_txn_amounts ': MEAN_TXN_AMOUNTS,
        'num_agents_per_type': NUM_AGENTS_PER_TYPE,
        'num_agents_per_type': 4, 
        'mean_txns': 4,  # avg num txns each agent makes
        'starting_balance': 100,
        'seed': 42,
        'mins_per_step': MINS_PER_STEP,  # 1 hr
        'steps': int(24 * (60/MINS_PER_STEP)),  # 24 hours * steps per hr
# hardcode, since can't give combo of options between the two
        'percent_sus': 1/100,
    }

    print('pre frezze')
    print(type(parameters_exp['MEAN_TXN_HRS']))
    for param in [
            'NUM_AGENTS_PER_TYPE', 
            'AGENT_TYPE_PAIR_PROBS', 
            'MEAN_TXN_HRS', 
            'MEAN_TXN_AMOUNTS', 
            'MEAN_NUM_TXNS']:
        parameters_exp[param] = frozendict(parameters_exp[param])
    print(type(parameters_exp[MEAN_TXN_HRS]))

    #-----------------------------------------------------------
    # --- NOTE: Setting experiment here! 
    #parameters_multi['percent_sus'] = ap.Values(10, 1, 0.1)
    print('parameters sweep; ', parameters_exp,
          parameters_exp['percent_sus'])
    sample = ap.Sample(parameters_exp) # grid search, each repeat 1x
    print('created sample; ', sample)

    # -- TEMP TEST 
    #model = BankModel(parameters_multi)
    #model = BankModel(Utility.get_default_params())
    #results = model.run()
    #print('finished test run with default params')

    exp = ap.Experiment(BankModel, parameters_exp)
    #exp = ap.Experiment(BankModel, Utility.get_default_params())
    #print('created exp; ', exp)

    #return exp
    results = exp.run()
    print('ran exp; ', results)
    #results.save()

    if viz:
        df = process_data(model)
        print('viz data')
        fig = viz_data(df) 
        print('done')

    return fig, model, results
   
# ------------Experiment
def run_default_model(viz=False):
    model = BankModel(Utility.get_default_params())
    results = model.run()
    print('sanity check, agent 0s txns', model.agents[0].txns)
    #display.display('sanity check, agent 0s txns', model.agents[0].txns)
    if viz:
        df = process_data(model)
        print('viz data')
        fig = viz_data(df) 
        print('done')
    return fig, model, results

# -------------------------------------------------------------
if __name__ == '__main__':
    run_exp()
    #model, results = run_default_model()
    df = process_data(model)
    print('viz data')
    viz_data(df) 
    print('done')
