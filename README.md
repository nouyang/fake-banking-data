# fake-banking-data
simulate financial transactions with agent-based modelling to create tabular and network data

# Idea
suspicious accounts (e.g. for sex trafficking) have ATM transactions at later hours than normal accounts. 

See https://www.fincen.gov/resources/advisories/fincen-advisory-fin-2014-a008

Thus, create a synthetic dataset of ATM transactions, (then compare the
synthetic to the real dataset). Use AgentPy, a python library for agent-based
modeling.

#  Currently
I have a working agent-based simulator with two types of agents.
Each agent is provided a pre-computed table of probabilities: the probability it will make a transaction at any given timestep
The user sets the time resolution, e.g. 1 step = 15 minutes
Each agent makes transactions following a normal distribution (so, unimodal distribution). See  for implementation details
For suspicious accounts, I used a mean at 10pm. For normal accounts, mean at 2pm. 

# Implementation Details

Note: Translating "transactions should have mean at 10pm" into what each agent decides to do at each timestep required some thought. Here is what I did:
I took the probability density function (PDF) of the normal distribution (which goes to infinity) and chopped it off at +/- 5 standard deviations (std devs)
I say that this truncated PDF represents the distribution of transactions (for a given agent) over 24 hrs
Then I can equate simulator timesteps (4 steps = 1 hour) to areas under the PDF (1 hour = 2.4 std devs. Integrate PDF to get probability for 15 minutes from the mean)
Scale the probabilities if agents make multiple transactions per day
Right now, assume both types of accounts make 3 transactions a day

# Plotting

I implemented functions to translate from timesteps (e.g. 40) into physical times (e.g. 10AM) for plotting.
Here, I vary the percent of agents that are suspicious. The blips from the suspicious transactions are impossible to see by eye even though 0.1% suspicious accounts is a high percentage.

# Todo

- Create a network: Implement (random) transactions between agents. Hopefully straightforward
 - Next, vary transactions by agent type:
 - suspicious transacts with suspicious, normal with normal
- Create 2D visualizations of the transaction network  

- Export: write function to turn simulation data (time-series and network) into tabular data
- Export: to tabular network features (in- and out- degrees)

- Visually inspect data under various parameters, write related works
- Write up results

- For revision, clean up, document, and release my code
 - for revision, if time permits: use time-of-day, in- & out-degrees to create a supervised classifier
 - for revision, if time permits: create validation function (Kolmogorov-Smirnov) to compare simulation to (alternate simulation), that could be used if I find real data


