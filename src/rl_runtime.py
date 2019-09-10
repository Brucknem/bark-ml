from modules.runtime.runtime import Runtime

class RuntimeRL(Runtime):
  """Runtime wrapper for reinforcement learning
     that includes additional observers and evaluators
  
  Arguments:
      Runtime {Runtime} -- Base class that is also
                           wrapped in CPP
  """
  def __init__(self,
               action_wrapper,
               observer,
               evaluator,
               step_time,
               viewer,
               scenario_generator=None,
               render=False):
    super().__init__(step_time=step_time,
                     viewer=viewer,
                     scenario_generator=scenario_generator,
                     render=render)
    self._action_wrapper = action_wrapper
    self._observer = observer
    self._evaluator = evaluator

  def reset(self, scenario=None):
    """Resets the runtime with and its objects
    """
    super().reset(scenario=scenario)
    self._world = self._evaluator.reset(self._world,
                                        self._scenario._eval_agent_ids)
    self._world = self._action_wrapper.reset(self._world,
                                             self._scenario._eval_agent_ids)
    self._world = self._observer.reset(self._world,
                                       self._scenario._eval_agent_ids)
    return self._observer.observe(
      world=self._world,
      agents_to_observe=self._scenario._eval_agent_ids)

  def step(self, action):
    """Steps the world
    
    Arguments:
        action {any} -- will be passed to the ActionWrapper
    
    Returns:
        (next_state, reward, done, info) -- RL tuple
    """
    self._world = self._action_wrapper.action_to_behavior(world=self._world,
                                                          action=action)
    self._world.step(self._step_time)
    if self._render:
      self.render()
    return self.snapshot(
      world=self._world,
      controlled_agents=self._scenario._eval_agent_ids)

  @property
  def action_space(self):
    """Action space for agent
    """
    return self._action_wrapper.action_space

  @property
  def observation_space(self):
    """Observation space for agent
    """
    return self._observer.observation_space

  def snapshot(self, world, controlled_agents):
    """Evaluates and observes the world from the controlled-agents's
       perspective
    
    Arguments:
        world {bark.world} -- Contains all required objects for the simulation
        controlled_agents {list} -- list of ids
    
    Returns:
        (next_state, reward, done, info) -- RL tuple
    """
    next_state = self._observer.observe(
      world=self._world,
      agents_to_observe=controlled_agents)
    reward, done, info = self._evaluator.get_evaluation(world=world)
    return next_state, reward, done, info

