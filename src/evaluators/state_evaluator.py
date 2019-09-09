from bark.world.evaluation import *
from modules.runtime.commons.parameters import ParameterServer

class StateEvaluator:
    def get_evaluation(self, world):
        return # reward, done, info

    def reset(self, world, agents_to_evaluate):
        return # world


class GoalReached(StateEvaluator):
    def __init__(self, params=ParameterServer()):
        self.params = params
        self.goal_reward = params["Runtime"]["RL"]["StateEvaluator"]["GoalReward", "The reward given for goals", 0.01]
        self.collision_reward = params["Runtime"]["RL"]["StateEvaluator"]["CollisionReward", "The (negative) \
                                                     reward given for collisions", -1]
        self.max_steps = params["Runtime"]["RL"]["StateEvaluator"]["MaxSteps", "The maximum number of \
                            steps allowed to take in the environment before episode is done", 50]
        self.eval_agent = None

    def get_evaluation(self, world):
        if self.eval_agent in world.agents:
            eval_results = world.evaluate()
            collision = eval_results["collision_agents"] or eval_results["collision_driving_corridor"]
            success = eval_results["success"]
            reward = collision * self.collision_reward + success * self.goal_reward
            max_steps_reached = eval_results["step_count"] > self.max_steps
            done = success or collision or max_steps_reached
            info = {"success": success, "collision_agents": eval_results["collision_agents"], \
                    "collision_driving_corridor": eval_results["collision_driving_corridor"], \
                    "outside_map": False, "num_steps": eval_results["step_count"]}
        else:
            collision = False
            success = False
            done = True
            reward = 0
            info = {"success": success, "collision_agents": False, \
                    "collision_driving_corridor": False, "outside_map": True,
                     "num_steps": None}
        return reward, done, info
        
    def reset(self, world, agents_to_evaluate):
        if len(agents_to_evaluate) != 1:
            raise ValueError("Invalid number of agents provided for GoalReached \
                        evaluation, number= {}".format(len(agents_to_evaluate)))
        self.eval_agent = agents_to_evaluate[0]
        evaluator1 = EvaluatorGoalReached(self.eval_agent)
        evaluator2 = EvaluatorCollisionEgoAgent(self.eval_agent) #EvaluatorCollisionAgents()
        evaluator3 = EvaluatorCollisionDrivingCorridor()
        evaluator4 = EvaluatorStepCount()

        world.add_evaluator("success", evaluator1)
        world.add_evaluator("collision_agents", evaluator2)
        world.add_evaluator("collision_driving_corridor", evaluator3)
        world.add_evaluator("step_count", evaluator4)

        return world
        

