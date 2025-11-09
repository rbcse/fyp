import os
import sys
import traci

# --- CONFIGURATION: YOU MUST EDIT THIS SECTION ---

# 1. PASTE YOUR TRAFFIC LIGHT ID HERE
#    (Find this in NETEDIT in 'Traffic Light' mode)
YOUR_TRAFFIC_LIGHT_ID = "8797384681"

# 2. PASTE YOUR DETECTOR IDs HERE
#    (These are the 'id's you gave your <inductionLoop> elements in 'Additionals' mode)
#    (Matching your detectors to your green phases)
DETECTOR_MAIN_ROAD = ["e1_2"]
DETECTOR_SIDE_ROAD = ["e1_0"]

# 3. CONFIGURE TRAFFIC LIGHT LOGIC
#    (From your screenshot 'Phases' table)
#
#    Phase #0 is 'GGrr' (Main Road Green)
#    Phase #2 is 'rrGG' (Side Road Green)

PHASE_INDEX_MAIN_GREEN = 0  # This is for phase #0 (GGrr)
PHASE_INDEX_SIDE_GREEN = 2  # This is for phase #2 (rrGG)

#    How long to wait before checking traffic again (in seconds)
MIN_GREEN_TIME = 10
MAX_GREEN_TIME = 30

# --- END OF CONFIGURATION ---


# --- DO NOT EDIT BELOW THIS LINE ---

def get_sumo_binary():
    """Finds the SUMO binary (sumo-gui) - HARDCODED FIX (Plan C)"""

    # --- !! YOU MUST EDIT THIS LINE !! ---
    # Paste the FULL path to your "sumo-gui.exe" file here.
    # Find it by right-clicking your SUMO-GUI icon -> "Open file location".
    # Use r"..." (a raw string) to handle the backslashes correctly.
    
    # DEFAULT PATH (EDIT IF YOURS IS DIFFERENT):
    sumo_path = r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo-gui.exe" 
    
    # --- End of edit ---

    if not os.path.exists(sumo_path):
        sys.exit(f"ERROR: Hardcoded path to SUMO not found at '{sumo_path}'. Please find your 'sumo-gui.exe' file and paste the full path into the 'sumo_path' variable in this script.")
    
    return sumo_path

def get_total_queued(detector_ids):
    """
    Checks a list of detectors and returns the total number of
    vehicles that have been waiting (speed < 0.1 m/s)
    """
    total_queued = 0
    for det_id in detector_ids:
        try:
            # getLastStepHaltingNumber gets vehicles with speed < 0.1 m/s
            queued_count = traci.inductionloop.getLastStepHaltingNumber(det_id)
            total_queued += queued_count
        except traci.TraCIException as e:
            print(f"Warning: Could not get data for detector '{det_id}'. Error: {e}")
            print("Check that this detector ID is correct in your svnit.add.xml file.")
    return total_queued

def run_simulation():
    """Main simulation loop"""

    # --- 1. Start SUMO ---
    # This command starts sumo-gui and points it to your config file
    sumo_binary = r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo-gui.exe"
    sumo_cmd = [
        sumo_binary,
        "-c", "map.sumocfg",
        "--start",
        "--quit-on-end",
        "--remote-port", "8813"
    ]



    try:
        traci.start(sumo_cmd, port=8813)
        print("SUMO simulation started...")
    except Exception as e:
        print(f"Error starting SUMO: {e}")
        print("Is SUMO already running? Please close any SUMO-GUI windows and try again.")
        return

    # --- 2. Main Simulation Loop ---
    step = 0
    current_phase_time = 0

    # Check if config is valid before starting
    if YOUR_TRAFFIC_LIGHT_ID == "PASTE_YOUR_TRAFFIC_LIGHT_ID_HERE":
        print("\n*** ERROR: Please edit this Python file and set 'YOUR_TRAFFIC_LIGHT_ID' ***\n")
        traci.close()
        return

    print(f"Taking control of Traffic Light: {YOUR_TRAFFIC_LIGHT_ID}")

    while traci.simulation.getMinExpectedNumber() > 0:
        try:
            traci.simulationStep()
            step += 1
            current_phase_time += 1

            # --- 3. Dynamic Logic ---
            # This logic runs every second

            # Get the current phase of the traffic light
            try:
                current_phase = traci.trafficlight.getPhase(YOUR_TRAFFIC_LIGHT_ID)
            except traci.TraCIException:
                print(f"\n*** ERROR: Could not find Traffic Light '{YOUR_TRAFFIC_LIGHT_ID}' ***")
                print("Please check the ID in NETEDIT and update the Python script.")
                traci.close()
                return

            # Only check logic if the minimum green time has passed
            if current_phase_time < MIN_GREEN_TIME:
                continue # Wait...

            # --- Check MAIN ROAD Green Phase ---
            if current_phase == PHASE_INDEX_MAIN_GREEN:
                # Check if main road is empty AND side road has cars
                main_queued = get_total_queued(DETECTOR_MAIN_ROAD)
                side_queued = get_total_queued(DETECTOR_SIDE_ROAD)

                print(f"Step {step}: Main Green (Time: {current_phase_time}s) | Main Queue: {main_queued} | Side Queue: {side_queued}")

                # Decision: If main road is empty AND side road is waiting, switch
                if main_queued == 0 and side_queued > 0:
                    print(">>> Main road empty, side road waiting. Switching to Side Green.")
                    traci.trafficlight.setPhase(YOUR_TRAFFIC_LIGHT_ID, PHASE_INDEX_SIDE_GREEN)
                    current_phase_time = 0 # Reset phase timer

                # Decision: If main road has been green for too long, switch
                elif current_phase_time > MAX_GREEN_TIME:
                    print(f">>> Max green time ({MAX_GREEN_TIME}s) reached. Switching to Side Green.")
                    traci.trafficlight.setPhase(YOUR_TRAFFIC_LIGHT_ID, PHASE_INDEX_SIDE_GREEN)
                    current_phase_time = 0 # Reset phase timer

            # --- Check SIDE ROAD Green Phase ---
            elif current_phase == PHASE_INDEX_SIDE_GREEN:
                # Check if side road is empty OR main road is waiting
                main_queued = get_total_queued(DETECTOR_MAIN_ROAD)
                side_queued = get_total_queued(DETECTOR_SIDE_ROAD)

                print(f"Step {step}: Side Green (Time: {current_phase_time}s) | Main Queue: {main_queued} | Side Queue: {side_queued}")

                # Decision: If side road is empty, switch back to main
                if side_queued == 0:
                    print(">>> Side road empty. Switching to Main Green.")
                    traci.trafficlight.setPhase(YOUR_TRAFFIC_LIGHT_ID, PHASE_INDEX_MAIN_GREEN)
                    current_phase_time = 0 # Reset phase timer
                
                # Decision: If side road has been green for too long, switch back
                elif current_phase_time > MAX_GREEN_TIME:
                    print(f">>> Max green time ({MAX_GREEN_TIME}s) reached. Switching to Main Green.")
                    traci.trafficlight.setPhase(YOUR_TRAFFIC_LIGHT_ID, PHASE_INDEX_MAIN_GREEN)
                    current_phase_time = 0 # Reset phase timer

        except traci.TraCIException as e:
            print(f"Error during simulation step {step}: {e}")
            break
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            break

    # --- 4. End Simulation ---
    traci.close()
    print("Simulation ended.")

if __name__ == "__main__":

    # Check for placeholder in config
    if YOUR_TRAFFIC_LIGHT_ID == "PASTE_YOUR_TRAFFIC_LIGHT_ID_HERE":
        print("="*80)
        print("ERROR: SCRIPT NOT CONFIGURED")
        print("Please open 'dynamic_controller.py' in a text editor.")
        print("You must fill in the configuration variables at the top of the file:")
        print(" - YOUR_TRAFFIC_LIGHT_ID")
        print(" - DETECTOR_MAIN_ROAD / DETECTOR_SIDE_ROAD")
        print(" - PHASE_INDEX_MAIN_GREEN / PHASE_INDEX_SIDE_GREEN")
        print("="*80)
    else:
        # Check if Traci is installed
        try:
            import traci
        except ImportError:
            print("="*80)
            print("ERROR: 'traci' library not found.")
            print("Please install it by running: pip install traci")
            print("="*80)
            sys.exit(1)

        run_simulation()