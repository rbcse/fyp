from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import time
import optparse
import random
# import serial # <-- MODIFIED: Commented out Arduino, you can re-enable if needed
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt

# we need to import python modules from the $SUMO_HOME/tools directory
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

from sumolib import checkBinary  # noqa
import traci  # noqa

# <--- These are your specific incoming lanes from svnit.add.xml
INCOMING_LANE_MAIN = "761955902#2_1" # Lane for detector e1_2
INCOMING_LANE_SIDE = "527057142_0"   # Lane for detector e1_0
INCOMING_LANES = [INCOMING_LANE_MAIN, INCOMING_LANE_SIDE]

# <--- These are your detector IDs from svnit.add.xml
DETECTOR_MAIN = "e1_2"
DETECTOR_SIDE = "e1_0"


def get_waiting_time(lanes):
    """Gets the total waiting time for a list of lanes."""
    waiting_time = 0
    for lane in lanes:
        # getWaitingTime returns cumulative waiting time for all vehicles on that lane
        waiting_time += traci.lane.getWaitingTime(lane)
    return waiting_time


def phaseDuration(junction, phase_time, phase_state):
    """Sets the phase of a traffic light."""
    traci.trafficlight.setRedYellowGreenState(junction, phase_state)
    traci.trafficlight.setPhaseDuration(junction, phase_time)


class Model(nn.Module):
    """Neural Network for the Deep Q-Network"""
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(Model, self).__init__()
        self.lr = lr
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.linear1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.linear2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.linear3 = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.loss = nn.MSELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        actions = self.linear3(x)
        return actions


class Agent:
    """DQN Agent"""
    def __init__(
        self,
        gamma,
        epsilon,
        lr,
        input_dims,
        fc1_dims,
        fc2_dims,
        batch_size,
        n_actions,
        junctions,
        max_memory_size=100000,
        epsilon_dec=5e-4,
        epsilon_end=0.05,
    ):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.batch_size = batch_size
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.action_space = [i for i in range(n_actions)]
        self.junctions = junctions
        self.max_mem = max_memory_size
        self.epsilon_dec = epsilon_dec
        self.epsilon_end = epsilon_end
        self.mem_cntr = 0
        self.iter_cntr = 0
        self.replace_target = 100

        self.Q_eval = Model(
            self.lr, self.input_dims, self.fc1_dims, self.fc2_dims, self.n_actions
        )
        self.memory = dict()
        for junction in junctions:
            self.memory[junction] = {
                "state_memory": np.zeros(
                    (self.max_mem, self.input_dims), dtype=np.float32
                ),
                "new_state_memory": np.zeros(
                    (self.max_mem, self.input_dims), dtype=np.float32
                ),
                "reward_memory":np.zeros(self.max_mem, dtype=np.float32),
                "action_memory": np.zeros(self.max_mem, dtype=np.int32),
                "terminal_memory": np.zeros(self.max_mem, dtype=bool), # Fixed np.bool
                "mem_cntr": 0,
                "iter_cntr": 0,
            }


    def store_transition(self, state, state_, action,reward, done,junction):
        index = self.memory[junction]["mem_cntr"] % self.max_mem
        self.memory[junction]["state_memory"][index] = state
        self.memory[junction]["new_state_memory"][index] = state_
        self.memory[junction]['reward_memory'][index] = reward
        self.memory[junction]['terminal_memory'][index] = done
        self.memory[junction]["action_memory"][index] = action
        self.memory[junction]["mem_cntr"] += 1

    def choose_action(self, observation):
        state = torch.tensor([observation], dtype=torch.float).to(self.Q_eval.device)
        if np.random.random() > self.epsilon:
            actions = self.Q_eval.forward(state)
            action = torch.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        return action
    
    def reset(self,junction_numbers):
        for junction_number in junction_numbers:
            self.memory[junction_number]['mem_cntr'] = 0

    def save(self,model_name):
        if not os.path.exists('models'):
            os.makedirs('models')
        torch.save(self.Q_eval.state_dict(),f'models/{model_name}.bin')

    def learn(self, junction):
        self.Q_eval.optimizer.zero_grad()

        batch= np.arange(self.memory[junction]['mem_cntr'], dtype=np.int32)

        state_batch = torch.tensor(self.memory[junction]["state_memory"][batch]).to(
            self.Q_eval.device
        )
        new_state_batch = torch.tensor(
            self.memory[junction]["new_state_memory"][batch]
        ).to(self.Q_eval.device)
        reward_batch = torch.tensor(
            self.memory[junction]['reward_memory'][batch]).to(self.Q_eval.device)
        terminal_batch = torch.tensor(self.memory[junction]['terminal_memory'][batch]).to(self.Q_eval.device)
        action_batch = self.memory[junction]["action_memory"][batch]

        q_eval = self.Q_eval.forward(state_batch)[batch, action_batch]
        q_next = self.Q_eval.forward(new_state_batch)
        q_next[terminal_batch] = 0.0
        q_target = reward_batch + self.gamma * torch.max(q_next, dim=1)[0]
        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)

        loss.backward() # Fixed truncated line
        
        self.Q_eval.optimizer.step()

        self.iter_cntr += 1
        self.epsilon = (
            self.epsilon - self.epsilon_dec
            if self.epsilon > self.epsilon_end
            else self.epsilon_end
        )


def run(train=True,model_name="model",epochs=50,steps=500,ard=False):
    """execute the TraCI control loop"""
    epochs = epochs
    steps = steps
    best_time = np.inf
    total_time_list = list()
    # --- FIX 1: Point to YOUR config file ---
    sumo_config = "map.sumocfg" 
    
    # --- MODIFIED: Get ONLY your specific traffic light
    all_junctions = ["8797384681"] 
    
    junction_numbers = list(range(len(all_junctions)))

    brain = Agent(
        gamma=0.99,
        epsilon=0.0,
        lr=0.1,
        input_dims=2, # main_queue, side_queue
        fc1_dims=256,
        fc2_dims=256,
        batch_size=1024,
        n_actions=2, # Action 0: Main Green, Action 1: Side Green
        junctions=junction_numbers,
    )

    if not train:
        try:
            brain.Q_eval.load_state_dict(torch.load(f'models/{model_name}.bin',map_location=brain.Q_eval.device))
        except FileNotFoundError:
            print(f"Warning: Could not find model 'models/{model_name}.bin'. Starting with a new model.")
        except Exception as e:
            print(f"Error loading model: {e}. Starting with a new model.")

    print(f"Using device: {brain.Q_eval.device}")
    
    # This is the command base for starting SUMO
    sumo_cmd_base = [
        checkBinary("sumo-gui"), 
        "-c", sumo_config,
        "--tripinfo-output", "tripinfo.xml",
        "--start", # Auto-plays the simulation
        "--quit-on-end" # <-- THIS IS THE FIX. Automatically closes GUI.
    ]

    for e in range(epochs):
        
        # --- Start SUMO for this epoch ---
        if train:
            traci.start(sumo_cmd_base)
        else:
            # For testing, you might want to see the last frame
            # We can remove --quit-on-end for testing if needed, but for now, keep it
            traci.start(sumo_cmd_base)

        print(f"epoch: {e}")
        
        # [Yellow State, Green State] for each action
        select_lane = [
            ["yyrr", "GGrr"], # Action 0: Main Road Green (Your Phase 0)
            ["rryy", "rrGG"]  # Action 1: Side Road Green (Your Phase 2)
        ]

        step = 0
        total_time = 0
        min_duration = 5
        
        traffic_lights_time = dict()
        prev_wait_time = dict()
        prev_vehicles_per_lane = dict()
        prev_action = dict()
        all_lanes = list()
        
        for junction_number, junction in enumerate(all_junctions):
            prev_wait_time[junction] = 0
            prev_action[junction_number] = 0
            traffic_lights_time[junction] = 0
            prev_vehicles_per_lane[junction_number] = [0] * 2
            all_lanes.extend(list(traci.trafficlight.getControlledLanes(junction)))

        while step <= steps:
            try:
                traci.simulationStep()
            except traci.TraCIException as sim_error:
                print(f"Error during simulation step {step}: {sim_error}")
                break # Exit this epoch and try the next one

            for junction_number, junction in enumerate(all_junctions):
                
                # Get total waiting time for all incoming lanes
                waiting_time = get_waiting_time(INCOMING_LANES)
                total_time += waiting_time
                
                if traffic_lights_time[junction] == 0:
                    
                    try:
                        # Get queue (halting number) from the LANES
                        main_queued = traci.lane.getLastStepHaltingNumber(INCOMING_LANE_MAIN)
                        side_queued = traci.lane.getLastStepHaltingNumber(INCOMING_LANE_SIDE)

                    except traci.TraCIException as e:
                        print(f"Warning: Could not read lane data. {e}")
                        main_queued = 0
                        side_queued = 0
                    
                    # State is a list of [main_queue, side_queue]
                    state_ = [main_queued, side_queued] 
                    
                    #storing previous state and current state
                    reward = -1 * waiting_time # Reward is negative waiting time
                    state = prev_vehicles_per_lane[junction_number]
                    prev_vehicles_per_lane[junction_number] = state_
                    brain.store_transition(state, state_, prev_action[junction_number],reward,(step==steps),junction_number)

                    #selecting new action based on current state
                    lane = brain.choose_action(state_)
                    prev_action[junction_number] = lane
                    
                    try:
                        # Set Yellow Phase (state[0])
                        phaseDuration(junction, 6, select_lane[lane][0]) 
                        # Set Green Phase (state[1])
                        phaseDuration(junction, min_duration + 10, select_lane[lane][1]) 
                    except traci.TraCIException as e:
                        print(f"Warning: Could not set traffic light phase. {e}")

                    traffic_lights_time[junction] = min_duration + 10
                    if train:
                        brain.learn(junction_number)
                else:
                    traffic_lights_time[junction] -= 1
            step += 1
            
        print(f"epoch {e} finished. total_time: {total_time}")
        total_time_list.append(total_time)

        if total_time < best_time:
            best_time = total_time
            if train:
                print(f"New best time. Saving model to 'models/{model_name}.bin'")
                brain.save(model_name)

        traci.close() # This will now trigger the --quit-on-end flag
        sys.stdout.flush()
        if not train:
            break
            
    if train:
        if not os.path.exists('plots'):
            os.makedirs('plots')
        plt.plot(list(range(len(total_time_list))),total_time_list)
        plt.xlabel("epochs")
        plt.ylabel("total time (cumulative wait)")
        plt.title("Training Progress")
        plt.savefig(f'plots/time_vs_epoch_{model_name}.png')
        print(f"Training plot saved to 'plots/time_vs_epoch_{model_name}.png'")
        plt.show()

def get_options():
    optParser = optparse.OptionParser()
    optParser.add_option(
        "-m",
        dest='model_name',
        type='string',
        default="model",
        help="name of model",
    )
    optParser.add_option(
        "--train",
        action = 'store_true',
        default=False,
        help="training or testing",
    )
    optParser.add_option(
        "-e",
        dest='epochs',
        type='int',
        default=50,
        help="Number of epochs",
    )
    optParser.add_option(
        "-s",
        dest='steps',
        type='int',
        default=500,
        help="Number of steps per epoch",
    )
    optParser.add_option(
       "--ard",
        action='store_true',
        default=False,
        help="Connect Arduino", 
    )
    options, args = optParser.parse_args()
    return options


# this is the main entry point of this script
if __name__ == "__main__":
    options = get_options()
    model_name = options.model_name
    train = options.train
    epochs = options.epochs
    steps = options.steps
    ard = options.ard

    try:
        import torch
    except ImportError:
        print("="*80)
        print("ERROR: 'torch' (PyTorch) library not found.")
        print("Please install it first. The command is usually:")
        print("pip install torch")
        print("="*80)
        sys.exit(1)
        
    try:
        import matplotlib
    except ImportError:
        print("="*80)
        print("ERROR: 'matplotlib' library not found.")
        print("Please install it first. The command is usually:")
        print("pip install matplotlib")
        print("="*80)
        sys.exit(1)

    if not os.path.exists('models'):
        print("Creating 'models' directory...")
        os.makedirs('models')
    if not os.path.exists('plots'):
        print("Creating 'plots' directory...")
        os.makedirs('plots')

    run(train=train,model_name=model_name,epochs=epochs,steps=steps,ard=ard)