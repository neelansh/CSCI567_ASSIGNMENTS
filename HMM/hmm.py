from __future__ import print_function
import json
import numpy as np


class HMM:

    def __init__(self, pi, A, B, obs_dict, state_dict):
        """
        - pi: (1*num_state) A numpy array of initial probabilities. pi[i] = P(Z_1 = s_i)
        - A: (num_state*num_state) A numpy array of transition probabilities. A[i, j] = P(Z_t = s_j|Z_{t-1} = s_i)
        - B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - obs_dict: A dictionary mapping each observation symbol to its index 
        - state_dict: A dictionary mapping each state to its index
        """
        self.pi = pi
        self.A = A
        self.B = B
        self.obs_dict = obs_dict
        self.state_dict = state_dict

    def forward(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - alpha: (num_state*L) A numpy array where alpha[i, t-1] = P(Z_t = s_i, X_{1:t}=x_{1:t})
                 (note that this is alpha[i, t-1] instead of alpha[i, t])
        """
        S = len(self.pi)
        L = len(Osequence)
        O = self.find_item(Osequence)
        alpha = np.zeros([S, L])
        ######################################################
        # TODO: compute and return the forward messages alpha
        ######################################################
        
        
        
        x1 = self.obs_dict[Osequence[0]]
        b_s_x1 = self.B[:, x1]
        alpha[:, 0] = self.pi * b_s_x1
        
        T = L
        for t in range(1, T):
            for s in range(0, S):
                xt = self.obs_dict[Osequence[t]]
                bsxt = self.B[s, xt]
                total_sum = 0
                for s_ in range(0, S):
                    total_sum += self.A[s_, s] * alpha[s_, t-1]
                alpha[s, t] = bsxt * total_sum

        return alpha

    def backward(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - beta: (num_state*L) A numpy array where beta[i, t-1] = P(X_{t+1:T}=x_{t+1:T} | Z_t = s_i)
                    (note that this is beta[i, t-1] instead of beta[i, t])
        """
        S = len(self.pi)
        L = len(Osequence)
        O = self.find_item(Osequence)
        beta = np.zeros([S, L])
        #######################################################
        # TODO: compute and return the backward messages beta
        #######################################################
        T = L
        beta[:, T-1] = 1

        for t in range(T-2, -1, -1):
            for s in range(S):
                beta[s, t] = 0
                for s_ in range(0, S):
                    x_tplus1 = self.obs_dict[Osequence[t+1]]
                    beta[s, t] += self.A[s, s_] * self.B[s_, x_tplus1] * beta[s_, t+1]
        
        return beta

    def sequence_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: A float number of P(X_{1:T}=x_{1:T})
        """
        
        #####################################################
        # TODO: compute and return prob = P(X_{1:T}=x_{1:T})
        #   using the forward/backward messages
        #####################################################
        
        prob = 0
        S = len(self.pi)
        alpha = self.forward(Osequence)
        beta = self.backward(Osequence)
        t = len(Osequence)-1
        for s in range(0, S):
            prob += alpha[s, t] * beta[s, t]
        
        return prob


    def posterior_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - gamma: (num_state*L) A numpy array where gamma[i, t-1] = P(Z_t = s_i | X_{1:T}=x_{1:T})
		           (note that this is gamma[i, t-1] instead of gamma[i, t])
        """
        ######################################################################
        # TODO: compute and return gamma using the forward/backward messages
        ######################################################################
        alpha = self.forward(Osequence)
        beta = self.backward(Osequence)
        total_sum = self.sequence_prob(Osequence)
        return alpha*beta/total_sum

    
    def likelihood_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: (num_state*num_state*(L-1)) A numpy array where prob[i, j, t-1] = 
                    P(Z_t = s_i, Z_{t+1} = s_j | X_{1:T}=x_{1:T})
        """
        S = len(self.pi)
        L = len(Osequence)
        prob = np.zeros([S, S, L - 1])
        #####################################################################
        # TODO: compute and return prob using the forward/backward messages
        #####################################################################
        T = L
        alpha = self.forward(Osequence)
        beta = self.backward(Osequence)
        for s in range(0, S):
            for s_ in range(0, S):
                for t in range(0, T-1):
                    x_tplus1 = self.obs_dict[Osequence[t+1]]
                    prob[s, s_, t] = alpha[s, t] * self.A[s, s_] * self.B[s_, x_tplus1] * beta[s_, t+1]
        
        return prob/self.sequence_prob(Osequence)


    def viterbi(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - path: A List of the most likely hidden states (return actual states instead of their indices;
                    you might find the given function self.find_key useful)
        """
        path = []
        ################################################################################
        # TODO: implement the Viterbi algorithm and return the most likely state path
        ################################################################################
        
        S = len(self.pi)
        T = len(Osequence)
        
        delta = np.zeros((S, T))
        delta_arg_max = np.zeros((S, T), dtype=int)
        x1 = self.obs_dict[Osequence[0]]
        delta[:, 0] = self.pi * self.B[:, x1]
        
        for t in range(1, T):
            for s in range(0, S):
                xt = self.obs_dict[Osequence[t]]
                delta[s, t] = self.B[s, xt] * np.max(self.A[:, s]*delta[:, t-1])
                delta_arg_max[s, t] = np.argmax(self.A[:, s]*delta[:, t-1])
        
        z_t_star = np.argmax(delta[:, -1])
        path.append(z_t_star)
        
        for t in range(T-1, 0, -1):
            z_tminus1 = delta_arg_max[path[-1], t]
            path.append(z_tminus1)
        
        reverse_state_dict = {v:k for (k, v) in self.state_dict.items()}
        for i, x in enumerate(path):
            path[i] = reverse_state_dict[x]
        
        return list(reversed(path))


    #DO NOT MODIFY CODE BELOW
    def find_key(self, obs_dict, idx):
        for item in obs_dict:
            if obs_dict[item] == idx:
                return item

    def find_item(self, Osequence):
        O = []
        for item in Osequence:
            O.append(self.obs_dict[item])
        return O
