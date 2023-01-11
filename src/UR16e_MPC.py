from isaacgym import gymapi, gymutil, gymtorch
from UR16e_sim import UR16eSim
from pytorch_mppi import mppi
import torch
import time
import math

from threading import Lock

class UR16eMPC(object):
  POS_ERROR_W = 200.0
  QUAT_ERROR_W = 0.0
  CTRL_COST_W = 0.0
  CONTACT_W = 0.1
  EPSILON = 1e-10
  
  NUM_ENVS = 297
  MPPI_NX = 25
  MPPI_NOISE_SIGMA = 15*torch.eye(2)
  MPPI_HORIZON = 10
  MPPI_LAMBDA = 0.1
  MPPI_U_MIN = torch.tensor(-25) 
  MPPI_U_MAX = torch.tensor(25)
  MPPI_SAMPLE_NULL_ACTION = False
    
  def __init__(self,
               gym,
               args,
               headless):
    self.gym = gym
    self.mppi_n_configs = 1 if brake_configs is None else brake_configs.shape[0]
    self.rs = UR16eSim(gym, self.NUM_ENVS, args, headless=headless) 
    self.goal_pos = self.rs.goal_pos[0,:]
    self.device = self.rs.device
                                 
    def mppi_dynamics(state, action):
      return self.rs.step(state, action)                         
                                  
    def mppi_running_cost(state, action):
      cost = torch.zeros(state.shape[0], dtype=torch.float32, device=self.device)    
    
    #   # Compute contact cost
    #   cost[torch.norm(self.bhs.l_tip_contact,dim=1) < self.EPSILON] += self.CONTACT_W
    #   cost[torch.norm(self.bhs.r_tip_contact,dim=1) < self.EPSILON] += self.CONTACT_W
      
      # Compute ctrl cost
      cost += self.CTRL_COST_W*torch.norm(action, dim=1)
      
      # Compute orientation cost (magnitude of x,y components of quaternion)
      cost += self.QUAT_ERROR_W*torch.norm(state[:,15:17], dim=1)
      
      return 1 # cost
    
    def mppi_terminal_state_cost(state, action):
      return self.POS_ERROR_W*torch.norm(self.goal_pos[0:2] - state[:,:,-1,12:14],dim=-1)

    self.UR16e_mppi = mppi.MPPI(dynamics=mppi_dynamics,
                             running_cost=mppi_running_cost,
                             nx=self.MPPI_NX,
                             noise_sigma=self.MPPI_NOISE_SIGMA,
                             num_samples=self.NUM_ENVS,
                             horizon=self.MPPI_HORIZON,
                             device=self.device,
                             terminal_state_cost=mppi_terminal_state_cost,
                             lambda_=self.MPPI_LAMBDA,
                             u_min=self.MPPI_U_MIN,
                             u_max=self.MPPI_U_MAX,
                             sample_null_action=self.MPPI_SAMPLE_NULL_ACTION,
                             n_configs=self.mppi_n_configs)  

  def set_goal_pose(self, goal_x, goal_y, goal_z):                             
    self.rs.set_goal_pose(goal_x, goal_y, goal_z)
    
  def compute_action(self, cur_state):
    self.rs.set_sim_state(cur_state)
    return self.UR16e_mppi.command(cur_state)                                  

  def step(self, cur_state, action):
    self.rs.set_sim_state(cur_state)
    return self.rs.step(cur_state, action[None, :], False, True)

  def get_vel_cmd(self, cur_state, prev_state, brake_config):
    if brake_config is not None:
      assert(len(brake_config.shape) == 1 and brake_config.shape[0] == 6)    
      for i in range(brake_config.shape[0]):# Filter out vel components from braked joints                
        if brake_config[i] == 1:
          prev_state[0,i] = cur_state[0,i]
    #print(cur_state[0,0:6]-prev_state[0,0:6])
    left_velocity = self.bhs.tendon_sensor.compute_tendon_velocity(self.bhs.dt,
                                                                   cur_state[0,0],
                                                                   cur_state[0,1],
                                                                   cur_state[0,2],
                                                                   prev_state[0,0],
                                                                   prev_state[0,1],
                                                                   prev_state[0,2])      
    right_velocity = self.bhs.tendon_sensor.compute_tendon_velocity(self.bhs.dt,
                                                                    cur_state[0,3],
                                                                    cur_state[0,4],
                                                                    cur_state[0,5],
                                                                    prev_state[0,3],
                                                                    prev_state[0,4],
                                                                    prev_state[0,5])
    cmd_msg = BataCmd()
    for i in range(2):
      chain_cmd_msg = BataChainCmd()
      chain_cmd_msg.enable_brake = [False]*3
      if brake_config is not None:
        for j in range(3):
          chain_cmd_msg.enable_brake[j] = (brake_config[3*i+j] == 1)
      cmd_msg.chain_cmds.append(chain_cmd_msg)
    cmd_msg.chain_cmds[0].motor_mode = 1
    cmd_msg.chain_cmds[0].motor_cmd = -1*left_velocity / 0.005 # Convert linear tendon velocity to motor angular velocity
    cmd_msg.chain_cmds[1].motor_mode = 1
    cmd_msg.chain_cmds[1].motor_cmd = -1*right_velocity / 0.005

    return cmd_msg  

  def get_pos_cmd(self, cur_state, sim_init_length, real_init_pos, brake_config):
    if brake_config is not None:
      assert(len(brake_config.shape) == 1 and brake_config.shape[0] == 6)
    left_position = self.bhs.tendon_sensor.compute_tendon_length(cur_state[0,0],
                                                                 cur_state[0,1],
                                                                 cur_state[0,2])
    right_position = self.bhs.tendon_sensor.compute_tendon_length(cur_state[0,3],
                                                                  cur_state[0,4],
                                                                  cur_state[0,5]) 
    
    cmd_msg = BataCmd()
    for i in range(2):
      chain_cmd_msg = BataChainCmd()
      chain_cmd_msg.enable_brake = [False]*3
      if brake_config is not None:
        for j in range(3):
          chain_cmd_msg.enable_brake[j] = (brake_config[3*i+j] == 1)
      cmd_msg.chain_cmds.append(chain_cmd_msg)
    cmd_msg.chain_cmds[0].motor_mode = 3
    cmd_msg.chain_cmds[0].motor_cmd = real_init_pos[0,0] + ((sim_init_length[0,0]-left_position)/0.005)
    cmd_msg.chain_cmds[1].motor_mode = 3
    cmd_msg.chain_cmds[1].motor_cmd = real_init_pos[0,1] + ((sim_init_length[0,1]-right_position)/0.005)  
    
    return cmd_msg
    
# MAKE SURE TO RUN WITH --pipeline cpu
def main():
    
  # initialize gym
  gym = gymapi.acquire_gym()

  args = gymutil.parse_arguments(
    description="BATA Hand MPC")

  if args.pipeline != 'cpu':
    print('ERROR:Please run with [--pipeline cpu], gpu pipeline has problems with setting state')
    return
 
  headless = False
  r_mpc = UR16eMPC(gym,
                    args,
                    headless)
  
  cur_state = r_mpc.rs.get_sim_state()[None, 0, :]

  cur_actions = None
  traj_weights = None

                                                                                                                                      
  while True:
    start = time.time()
    
    cur_actions, traj_weights = r_mpc.compute_action(cur_state)    

    action_idx = torch.argmax(traj_weights)
    action = cur_actions[action_idx]
    # Record action, cost
      
    cur_state = r_mpc.step(cur_state, action)[None, action_idx, :]

    goal_dist = math.sqrt((r_mpc.bhs.goal_pos[0,0] - cur_state[0,12])**2 + (r_mpc.bhs.goal_pos[0,1] - cur_state[0,13])**2)
    #print("Goal dist: %f"%(goal_dist))

    if goal_dist < 0.003:
      r_mpc.set_goal_pose(0.165, -1 * r_mpc.bhs.goal_pos[0,1])
      #bh_mpc.set_goal_pose(bh_mpc.bhs.goal_pos[0,0], -1*bh_mpc.bhs.goal_pos[0,1])
      
    print("Time: %f, action_idx: %d"%(time.time()-start, torch.argmax(traj_weights)))
    
if __name__ == '__main__':
  main()