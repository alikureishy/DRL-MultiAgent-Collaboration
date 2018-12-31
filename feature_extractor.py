import numpy as np

"""
Adds an identifying column to the feature tensor
so that each feature can be learned based on the
agent's perspective in the environment. In other
words, this allows for agents to learn in situations
where their perspectives are different. An example
would be at a game of tennis, where the two agents
face each other, and since the perspective is the
opposite from each side, the agents would learn
conflicting behavior (i.e, learn nothing) unless
an identifying column is provided to disambiguate
between those perspectives.
"""
class DisambiguatingFeatureExtractor(object):
    def __init__(self):
        pass

    def estimate_state_size(self, original_size):
        return original_size + 1
        # return original_size

    """
        The extractor essentially appends a column
        with monotonically increasing cell values
        that serve to disambiguate between different
        perspectives, assuming of course that each
        observation is a stack of state vectors from
        each such perspective.
    """
    def extract_states(self, observations):
        n = observations.shape[0]
        ids = np.arange(0, n).reshape((n,1))
        new_states = np.hstack((ids, observations))
        return new_states
        # return observations

    def estimate_action_size(self, original_size):
        return original_size

    def extract_actions(self, actions):
        return actions

# class JointSpaceFeatureExtractor(object):
#     def __init__(self, num_agents):
#         self.num_agents = num_agents

#     def estimate_state_size(self, original_size):
#         return self.num_agents * original_size

#     def estimate_action_size(self, original_size):


#     """
#         The extractor essentially appends a column
#         with monotonically increasing cell values
#         that serve to disambiguate between different
#         perspectives, assuming of course that each
#         observation is a stack of state vectors from
#         each such perspective.
#     """
#     def extract(self, observations):
#         n = observations.shape[0]
#         ids = np.arange(0, n).reshape((n,1))
#         new_states = np.hstack((ids, observations))
#         return new_states
