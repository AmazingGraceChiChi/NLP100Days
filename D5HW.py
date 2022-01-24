# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 14:56:43 2022

@author: User
"""

# In[]
# 若有一個人連續觀察到三天水草都是乾燥的(Dry), 則這三天的天氣機率為何？
observations = ('dry', 'dry', 'dry') #實際上觀察到的狀態為dry, dry, dry
states = ('sunny', 'rainy')
start_probability = {'sunny': 0.4, 'rainy': 0.6}
transition_probability = {'sunny':{'sunny':0.6, 'rainy':0.4},
                          'rainy': {'sunny':0.3, 'rainy':0.7}}
emission_probatility = {'sunny': {'dry':0.6, 'damp':0.3, 'soggy':0.1},
                        'rainy': {'dry':0.1, 'damp':0.4, 'soggy':0.5}}

# In[]
def viterbi(obs, states, start_p, trans_p, emit_p):
    V = [{}]
    path = {}

    # Initialize points (t == 0)
    # Only two scenarios for initialization
    # sunny: P(sunny) * P(dry|sunny)
    # rainy: P(rainy) * P(dry|rainy)
    for y in states:
        V[0][y] = start_p[y] * emit_p[y][obs[0]]
        path[y] = [y]

    # Run Viterbi for t > 0
    for t in range(1,len(obs)):
        V.append({})
        newpath = {}
        
        # cur_state is either "sunny" or "rainy"
        for cur_state in states:            
            # below is calculating the probability of next state based on previous state
            # ex: cur_state = "sunny"
            # P(sunny)*P(sunny|sunny)*P(dry|sunny) <-- pre_state is "sunny"
            # P(rainy)*P(sunny|rainy)*P(dry|sunny) <-- pre_state is "rainy"
            # keep the biggest probability choice
            (prob, state) = max([(V[t-1][pre_state] * trans_p[pre_state][cur_state] * emit_p[cur_state][obs[t]], pre_state) for pre_state in states])
            V[t][cur_state] = prob
            newpath[cur_state] = path[state] + [cur_state]
        # Don't need to remember the old paths
        path = newpath
    
    (prob, state) = max([(V[len(obs) - 1][final_state], final_state) for final_state in states])
    return (prob, path[state])

# In[]
result = viterbi(observations,
                 states,
                 start_probability,
                 transition_probability,
                 emission_probatility)












