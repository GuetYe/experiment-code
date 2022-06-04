""""
function:Memory
date    :2021/10/16
author  :HLQ
"""
import os,sys
PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PATH)
import time
import torch
from env import Env
from ppo_agent import Agent
from drawTools import *
from utils import PathOperator
from tensorboardX import SummaryWriter

#=======================================#
#       Train PPO algorithm offline     #
#=======================================#
class RLDQN:
    def __init__(self):
        # --- train/test params --- #
        print(" ..... LOAD PPO MAIN ......")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.batch_size = 64
        self.hidden_size = 256
        self.gamma = 0.99
        self.lr_actor = 0.0003
        self.lr_critic = 0.001
        self.lr_rnn = 0.001
        self.K_epochs = 40
        self.eps_clip = 0.2
        # self.lr = 0.001

        # --- larger size --- #
        # self.frequency_step = 100
        # self.frequency_reward = 500
        # self.frequency_ddpg_network = 1000
        # self.frequency_rnn_network = 1000
        # self.frequency_rnn_loss = 500
        # self.save_reward_count = 10000
        # self.total_train_episodes = 100000

        # --- small size --- #
        self.frequency_step = 4                 # update target network
        self.frequency_reward = 10
        self.frequency_ddpg_network = 200       # save policy network
        self.frequency_rnn_network = 200
        self.frequency_rnn_loss = 100
        self.save_reward_count = 500
        # self.max_ep_len = 100                 # max timesteps in one episode
        self.update_timestep = 128              # update policy every n timesteps
        # max_training_timesteps = int(1e5)     # break training loop if timeteps > max_training_timesteps
        self.total_train_episodes = 4000

        # --- generate obj --- #
        self.env = Env()
        self.agent = Agent(self.env.action_dim, self.lr_actor, self.lr_critic, self.lr_rnn, self.gamma, self.K_epochs, self.eps_clip, device)
        self.file_obj = self.env.operator_file_obj                # file obj
        self.path_obj = PathOperator()                            # path obj
        self.draw_obj = draw()
        self.train_reward = []
        self.train_rnn_loss = []
        self.ma_train_reward = []

        self.use_rnn_network = True

    def output_related_parameters(self):
        """
            Output DRL's params
        :return
        """
        print("-----------------------------------")
        print("gamma :", self.gamma)
        print("lr actor|critic|rnn :", self.lr_actor, self.lr_critic, self.lr_rnn)
        print("batch :", self.batch_size)
        print("use rnn network :", self.use_rnn_network)
        print("weight factor :", self.env.weight_factors)
        print("beta factor :", self.env.beta_factor)
        print("device :", torch.cuda.current_device(), torch.cuda.device_count())
        print("-----------------------------------")


    def breakpoint_train(self):
        """
            Determine whether to continue breakpoint training
        :return:
        """
        data = self.path_obj.load_breakpoint_pth()   # idx and pths
        if data:
            self.agent.policy_net.load_state_dict(torch.load(data[1]))
            return data[0]
        else:
            return 0


    def train_model(self):
        """
            Train model
        :return:
        """
        torch.manual_seed(1)
        torch.cuda.manual_seed_all(1)
        # breakpoint execution
        start_t = self.breakpoint_train()  # get location of the breakpoint
        if start_t != 0:
            print("Continue execution T : ", start_t)
            self.path_obj.dqn_idx = start_t
        # load gru model
        if self.use_rnn_network:
            self.agent.gru.load_state_dict(torch.load(self.path_obj.load_gru_model_pth()))
        decay = start_t * self.frequency_ddpg_network + 1
        time_step = 0
        for t in range(start_t * self.frequency_ddpg_network + 1, self.total_train_episodes):
            episode_reward = 0.
            # save_reward = 0.
            state = self.env.reset()
            while True:
                action = self.agent.action(state)
                next_state, reward, done, _ = self.env.step(action)
                episode_reward += reward
                self.agent.buffer.rewards.append(reward)
                self.agent.buffer.is_terminals.append(done)
                # store experience pools
                # self.agent.memory.store_experience(state, action, reward, next_state, done)
                time_step += 1
                # command rnn network #
                if self.use_rnn_network:
                    # shape seq, 8, 14 * 14
                    seq_previous_traffic = self.agent.rnn_obj.generate_predict_traffic(self.env)
                    if seq_previous_traffic is not None:
                        predict_state = self.agent.get_predict_state(seq_previous_traffic)        # numpy shape, 3, 196
                        predict_state = predict_state.reshape((self.env.k_paths, self.env.nodes, -1))  # numpy shape, 3, 14, 14
                        action = self.agent.action(predict_state)
                        reward, done, _ = self.env.predict_step(action)
                        self.agent.buffer.rewards.append(reward)
                        self.agent.buffer.is_terminals.append(done)
                        time_step += 1
                state = next_state
                # update policy network #
                if time_step % self.update_timestep  == 0:
                    self.agent.update()
                # exit loop
                if done:
                    break
            # save reward & ma_reward
            self.train_reward.append(episode_reward)
            if self.ma_train_reward:
                self.ma_train_reward.append(0.9 * self.ma_train_reward[-1] + 0.1 * episode_reward)
            else:
                self.ma_train_reward.append(episode_reward)
            # print reward value
            if (t + 1) % self.frequency_reward == 0:
                print("T:%s REWARD:%s " % (t + 1, episode_reward))
                episode_reward = 0.

            # update target network params
            if (t + 1) % self.frequency_ddpg_network == 0:
                torch.save(self.agent.policy_net.state_dict(), self.path_obj.get_save_path(type="PPO") + "ppo.pth")
                print("\033[33;1m POLICY NETWORK HAS BEEN SAVED in [%s]\033[0m" % self.wacth_time())

        print("Train done !!!", self.wacth_time())
        print("Train done !!!", self.wacth_time())
        try:
            with open(
                    os.path.abspath(os.path.dirname(self.path_obj.get_save_path(type="REWARD"))) + "/check_reward.txt",
                    "w") as f:
                np.savetxt(f, np.array(self.train_reward), fmt="%f", delimiter=" ")
            with open(os.path.abspath(
                    os.path.dirname(self.path_obj.get_save_path(type="REWARD"))) + "/check_ma_reward.txt", "w") as f:
                np.savetxt(f, np.array(self.ma_train_reward), fmt="%f", delimiter=" ")
            with open(os.path.abspath(os.path.dirname(os.path.dirname(self.path_obj.get_save_path(type="REWARD")))) + "/explain.txt",
                      "w") as f:
                title = "***The influence of GRU K=8 PPO weight on the experiment was measured***\n"
                traffic_weight = "traffic: [0.6, 0.3, 0.1]  --->   " + str(self.env.weight_factors) + "\n"
                reward_weight = "reward: [1.0, -1.0, -1.0, -1.0, -1.0, -1.0]  --->   " + str(
                    self.env.beta_factor) + "\n"
                infos = title + traffic_weight + reward_weight
                f.write(infos)
        except Exception as e:
            pass

        with open("check_reward.txt", "w") as f:
            np.savetxt(f, np.array(self.train_reward), fmt="%f", delimiter=" ")
        with open("check_ma_reward.txt", "w") as f:
            np.savetxt(f, np.array(self.ma_train_reward), fmt="%f", delimiter=" ")

        self.draw_obj.draw_reward_curve(self.path_obj, self.train_reward, type="REWARD")
        self.draw_obj.draw_ma_reward_curve(self.path_obj, self.ma_train_reward, type="REWARD")
        self.draw_obj.draw_reward_curve(self.path_obj, self.train_reward, type="LOSS")

    def wacth_time(self):
        """
            time counter.
        :return:
        """
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))


if __name__ == '__main__':
    rldqn = RLDQN()
    rldqn.output_related_parameters()
    rldqn.train_model()


