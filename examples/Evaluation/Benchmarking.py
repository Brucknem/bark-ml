##imports
import os
import matplotlib.pyplot as plt
##from IPython.display import Video

from benchmark_database.load.benchmark_database import BenchmarkDatabase
from benchmark_database.serialization.database_serializer import DatabaseSerializer
from bark.benchmark.benchmark_runner import BenchmarkRunner, BenchmarkConfig, BenchmarkResult
from bark.benchmark.benchmark_analyzer import BenchmarkAnalyzer

from bark.runtime.commons.parameters import ParameterServer

from bark.runtime.viewer.matplotlib_viewer import MPViewer
from bark.runtime.viewer.video_renderer import VideoRenderer

from bark.core.models.behavior import BehaviorIDMClassic, BehaviorConstantVelocity


##gail set up
from bark_ml.environments.blueprints import ContinuousHighwayBlueprint, \
  ContinuousMergingBlueprint, ContinuousIntersectionBlueprint, GailMergingBlueprint


"""
hab im bazel bin noch zwei dateien verändert:
benchmark_runner.py
l.212
except Exception as e:
            self.logger.error("For config-idx {}, Exception thrown in scenario.GetWorldState: {}".format(
                benchmark_config.config_idx, e))
            self._append_exception(benchmark_config, e)
            output={"config_idx": benchmark_config.config_idx,
                    "scen_set": benchmark_config.scenario_set_name,
                    "scen_idx": benchmark_config.scenario_idx,
                    "step": step,
                    "behavior": benchmark_config.behavior_name,
                    "Terminal": "exception_raised"}
            return output, scenario_history

scenario.py
nach den Imports noch hinzufügen

# Module variable to maintain map directory
__MAPFILE_DIRECTORY = None

def SetMapfileDirectory(dir):
  global __MAPFILE_DIRECTORY
  __MAPFILE_DIRECTORY = dir

def GetMapfileDirectory():
  global __MAPFILE_DIRECTORY
  return __MAPFILE_DIRECTORY
"""

###############################################################################################################################
##trying to get the needed param server from scenario
params = ParameterServer(filename="examples/example_params/gail_params.json")
# params = ParameterServer()
# changing the logging directories if not the default is used. (Which would be the same as it is in the json file.)
params["World"]["remove_agents_out_of_map"] = True

# create environment
bp = ContinuousMergingBlueprint(params,
                                number_of_senarios=500,
                                random_seed=0)
#scenario donesnt contain right info
#print(bp._scenario_generation.create_single_scenario().GetWorldState().GetParams().GetCondensedParamList())
################################################################################################################################


#benchmarking from jupyter notebook

"""
dbs = DatabaseSerializer(test_scenarios=1, test_world_steps=15, num_serialize_scenarios=1, viewer=True, visualize_tests=True, test_scenario_idxs=[1])

## download the benchmarking folder from https://github.com/bark-simulator/benchmark-database/tree/master/data and specify path correctly
path_to_dir="/home/marvin/Downloads/data/database1"

dbs.process(path_to_dir)
local_release_filename = dbs.release(version="tutorial")

print('Filename:', local_release_filename)

db = BenchmarkDatabase(database_root=local_release_filename)
scenario_generation, _ = db.get_scenario_generator(scenario_set_id=0)

for scenario_generation, _ in db:
  print('Scenario: ', scenario_generation)



evaluators = {"success" : "EvaluatorGoalReached", \
              "collision" : "EvaluatorCollisionEgoAgent", \
              "max_steps": "EvaluatorStepCount"}



terminal_when = {"collision" :lambda x: x, \
                 "max_steps": lambda x : x>100, \
                 "success" : lambda x: x}



params = ParameterServer() 
behaviors_tested = {"IDM": BehaviorIDMClassic(params), 
#"Const" : BehaviorConstantVelocity(params)
}

benchmark_runner = BenchmarkRunner(benchmark_database=db,\
                                   evaluators=evaluators,\
                                   terminal_when=terminal_when,\
                                   behaviors=behaviors_tested,\
                                   log_eval_avg_every=10)

result = benchmark_runner.run(maintain_history=True)

result.dump(os.path.join("./benchmark_results.pickle"))



result_loaded = BenchmarkResult.load(os.path.join("./benchmark_results.pickle"))

df = result_loaded.get_data_frame()

df.head()

analyzer = BenchmarkAnalyzer(benchmark_result=result_loaded)


configs_idm = analyzer.find_configs(criteria={"behavior": lambda x: x=="IDM", "success": lambda x : not x})
configs_const = analyzer.find_configs(criteria={"behavior": lambda x: x=="Const", "success": lambda x : not x})

sim_step_time=0.2

params2 = ParameterServer()

fig = plt.figure(figsize=[10, 10])
viewer = MPViewer(params=params2, y_length = 80, enforce_y_length=True, enforce_x_length=False,\
                  follow_agent_id=True, axis=fig.gca())
video_exporter = VideoRenderer(renderer=viewer, world_step_time=sim_step_time)

analyzer.visualize(viewer = video_exporter, real_time_factor = 1, configs_idx_list=configs_idm[1:3], \
                  fontsize=6)
"""         


