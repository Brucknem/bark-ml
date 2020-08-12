from pathlib import Path
from collections import defaultdict
import multiprocessing
import numpy as np
from typing import Tuple
import pandas as pd

# Bark
from bark.runtime.scenario.scenario_generation.interaction_dataset_scenario_generation import \
    InteractionDatasetScenarioGeneration
from bark.runtime.commons.parameters import ParameterServer
from bark.runtime.scenario.scenario import Scenario

from bark_ml.library_wrappers.lib_tf2rl.load_save_utils import list_files_in_dir


def get_track_files(tracks_dir: str) -> list:
  """Extracts all track file names in the given directory.

  Args:
      tracks_dir (str): The directory to search for track files

  Returns:
      list: The track files
  """
  return list_files_in_dir(tracks_dir, ".csv")


def create_parameter_servers_for_scenarios(
        map_file: str, tracks_dir: str) -> dict:
  """Generate a parameter server for every track file in the given directory.

  Args:
      map_file (str): The path of the map_file
      tracks_dir (str): The directory containing the track files

  Raises:
      ValueError: Map is not in Xodr format.

  Returns:
      dict: The parameter servers by mao and track files.
  """

  if not map_file.endswith(".xodr"):
    raise ValueError(
        f"Map file has to be in Xodr file format. Given: {map_file}")

  tracks = get_track_files(tracks_dir)

  param_servers = {}
  for track in tracks:
    df = pd.read_csv(track)
    track_ids = df.track_id.unique()
    start_ts = df.timestamp_ms.min()
    end_ts = df.timestamp_ms.max()
    num_rows = len(df.index)

    param_server = ParameterServer()
    param_server["Scenario"]["Generation"][
        "InteractionDatasetScenarioGeneration"]["MapFilename"] = map_file
    param_server["Scenario"]["Generation"][
        "InteractionDatasetScenarioGeneration"]["TrackFilename"] = track
    param_server["Scenario"]["Generation"][
        "InteractionDatasetScenarioGeneration"]["TrackIds"] = list(track_ids)
    param_server["Scenario"]["Generation"][
        "InteractionDatasetScenarioGeneration"]["StartTs"] = start_ts
    param_server["Scenario"]["Generation"][
        "InteractionDatasetScenarioGeneration"]["EndTs"] = end_ts
    param_server["Scenario"]["Generation"][
        "InteractionDatasetScenarioGeneration"]["EgoTrackId"] = track_ids[0]
    map_id = map_file.split("/")[-1].replace(".xodr", "")
    track_id = track.split("/")[-1].replace(".csv", "")
    param_servers[map_id, track_id] = param_server

  return param_servers


def create_scenario_generator(
        param_server: ParameterServer) -> InteractionDatasetScenarioGeneration:
  """Creates a bark scenario generator based on the given parameter server.

  Args:
      param_server (ParameterServer): The parameter server

  Returns:
      InteractionDatasetScenarioGeneration: The scenario generator
  """
  return InteractionDatasetScenarioGeneration(
      num_scenarios=1, random_seed=0, params=param_server)


def create_scenario(param_server: ParameterServer) -> Tuple[Scenario, float, float]:
  """Creates a bark scenario based on the given parameter server.

  Args:
      param_server (ParameterServer): The parameter server

  Returns:
      Tuple[Scenario, float, float]: The bark scenario, the start timestamp, the end timestamp
  """
  scenario_generation = create_scenario_generator(param_server)
  return (scenario_generation.get_scenario(0),
          param_server["Scenario"]["Generation"]["InteractionDatasetScenarioGeneration"]["StartTs"],
          param_server["Scenario"]["Generation"]["InteractionDatasetScenarioGeneration"]["EndTs"])
