import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pykan.kan import KAN, create_dataset
import gymnasium as gym
import os

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

# Define the Agent class
class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        # self.critic = KAN(width=[np.array(envs.observation_space.shape).prod()] + [16] * 2 + [1], grid=3, k=3, grid_range=[-1, 1], device='cpu')
        # self.actor_mean = KAN(width=[np.array(envs.observation_space.shape).prod()] + [16] * 2 + [np.prod(envs.action_space.shape)], grid=3, k=3, grid_range=[-1, 1], device='cpu')

        # self.critic = KAN(width=[np.array(envs.observation_space.shape).prod()] + [12] * 2 + [1], grid=8, k=3, grid_range=[-1, 1], device='cpu')

        # self.pre_actor = nn.Sequential(
        #     layer_init(nn.Linear(np.array(envs.observation_space.shape).prod(), np.array(envs.observation_space.shape).prod())),
        #     nn.Tanh(),
        # )

        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = KAN(width=[np.array(envs.observation_space.shape).prod()] + [np.prod(envs.action_space.shape)], grid=3, k=2, grid_range=[-1, 1], device='cpu')

        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.action_space.shape)))

    # def get_value(self, x):
    #     return self.critic(x)

    # def get_action_and_value(self, x, action=None):
    #     action_mean = self.actor_mean(x)
    #     action_logstd = self.actor_logstd.expand_as(action_mean)
    #     action_std = torch.exp(action_logstd)
    #     probs = torch.distributions.Normal(action_mean, action_std)
    #     if action is None:
    #         action = probs.sample()
    #     return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

# Function to load a pre-trained model
def load_pretrained_model(filepath, envs):
    # Initialize the agent with the given environment
    agent = Agent(envs)

    # Load the state_dict from the file
    agent.load_state_dict(torch.load(filepath))

    return agent

def plot_kan_model(model, folder="./figures", num_samples=100, in_vars=None, out_vars=None, title='KAN Model'):
    if not os.path.exists(folder):
        os.makedirs(folder)
    print("---------------------------------------------------")
    print("Model Widths:", model.width)
    print("---------------------------------------------------")
    input_data = torch.normal(0, 1, size=(num_samples, model.width[0]))  # Example input data
    outputs = model(input_data)  # Perform a forward pass to obtain outputs
    
    # Assuming your model has a 'plot' method or similar; update this as per your model's API
    if in_vars is not None and hasattr(model, 'plot'):
        model.plot(folder=folder, beta=100, in_vars=in_vars, out_vars=out_vars, title=title)
    else:    
        # model.plot(folder=folder, beta=100, in_vars=[r'$\alpha$', 'x'], out_vars=['y'], title='My KAN')
        model.plot(folder=folder, beta=100, scale=1.0)



if __name__ == "__main__":


    # # create a KAN: 2D inputs, 1D output, and 5 hidden neurons. cubic spline (k=3), 5 grid intervals (grid=5).
    # model = KAN(width=[2,5,1], grid=5, k=3, seed=0)

    # # create dataset f(x,y) = exp(sin(pi*x)+y^2)
    # f = lambda x: torch.exp(torch.sin(torch.pi*x[:,[0]]) + x[:,[1]]**2)
    # dataset = create_dataset(f, n_var=2)
    # # plot KAN at initialization
    # model(dataset['train_input'])
    # model.plot(beta=100)
    # print("done")


    # Create your environment
    envs = gym.make('Swimmer-v4', render_mode="rgb_array")
    # env = gym.wrappers.RecordVideo(envs, f"videos/teste")

    # Path to the pre-trained model
    model_filepath = 'Swimmerk2g3.cleanrl_model'

    # Load the pre-trained model
    agent = load_pretrained_model(model_filepath, envs)

    # Debug print to ensure the agent is correctly loaded
    print("Loaded agent:", agent)

    # Plot the critic model
    # plot_kan_model(agent.critic, folder="./figures/critic", num_samples=1000)
    # in_vars = ['Tip Angle', 'Rotor 1 Angle', 'Rotor 2 Angle', 'X-Velocity', 'Y-Velocity', 'Tip Ang. Vel.', 'Rotor 1 Ang. Vel.', 'Rotor 2 Ang. Vel.', 'X-Position', 'Y-Position']
    # out_vars = ['Torque 1', 'Torque 2']

    # Ajuste da chamada de plotagem para incluir os novos rótulos
    plot_kan_model(agent.actor_mean, folder="./figures/Swimmerk2g3novo", num_samples=100)

    # Save the figure as a PDF file
    pdf_path = 'Swimmerk2g3novo.pdf'
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')

    # # Show plots
    plt.show()



    # # Create your environment
    # envs = gym.make('InvertedPendulum-v4', render_mode="rgb_array")
    # # env = gym.wrappers.RecordVideo(envs, f"videos/teste")

    # # Path to the pre-trained model
    # model_filepath = 'InvertedPendulum.cleanrl_model'

    # # Load the pre-trained model
    # agent = load_pretrained_model(model_filepath, envs)

    # # Debug print to ensure the agent is correctly loaded
    # print("Loaded agent:", agent)

    # # Plot the critic model
    # # plot_kan_model(agent.critic, folder="./figures/critic", num_samples=1000)

    # # Plot the actor model
    # plot_kan_model(agent.actor_mean, folder="./figures/actorInvertedPendulum", num_samples=100)

    # # # Show plots
    # plt.show()

    
    # # Create your environment
    # envs = gym.make('InvertedPendulum-v4', render_mode="rgb_array")
    # # env = gym.wrappers.RecordVideo(envs, f"videos/teste")

    # # Path to the pre-trained model
    # model_filepath = 'InvertedPendulum.cleanrl_model'

    # # Load the pre-trained model
    # agent = load_pretrained_model(model_filepath, envs)

    # # Debug print to ensure the agent is correctly loaded
    # print("Loaded agent:", agent)

    # # Plot the critic model
    # # plot_kan_model(agent.critic, folder="./figures/critic", num_samples=1000)

    # # Plot the actor model
    # plot_kan_model(agent.actor_mean, folder="./figures/actorInvertedPendulum", num_samples=100)

    # # # Show plots
    # plt.show()

    
    # # Create your environment
    # envs = gym.make('InvertedPendulum-v4', render_mode="rgb_array")
    # # env = gym.wrappers.RecordVideo(envs, f"videos/teste")

    # # Path to the pre-trained model
    # model_filepath = 'InvertedPendulum.cleanrl_model'

    # # Load the pre-trained model
    # agent = load_pretrained_model(model_filepath, envs)

    # # Debug print to ensure the agent is correctly loaded
    # print("Loaded agent:", agent)

    # # Plot the critic model
    # # plot_kan_model(agent.critic, folder="./figures/critic", num_samples=1000)

    # # Plot the actor model
    # plot_kan_model(agent.actor_mean, folder="./figures/actorInvertedPendulum", num_samples=100)

    # # # Show plots
    # plt.show()

    
    # # Create your environment
    # envs = gym.make('Hopper-v3', render_mode="rgb_array")
    # env = gym.wrappers.RecordVideo(envs, f"videos/teste")

    # # Path to the pre-trained model
    # model_filepath = 'model.cleanrl_model'

    # # Load the pre-trained model
    # agent = load_pretrained_model(model_filepath, envs)

    # # Debug print to ensure the agent is correctly loaded
    # print("Loaded agent:", agent)

    # # Plot the critic model
    # # plot_kan_model(agent.critic, folder="./figures/critic", num_samples=1000)

    # # Plot the actor model
    # plot_kan_model(agent.actor_mean, folder="./figures/actor", num_samples=100)

    # # # Show plots
    # plt.show()



# if __name__ == "__main__":

#     model = KAN(width=[2, 5, 1], grid=5, k=3, seed=0)

#     # create dataset f(x,y) = exp(sin(pi*x)+y^2)
#     f = lambda x: torch.exp(torch.sin(torch.pi * x[:, [0]]) + x[:, [1]] ** 2)
#     dataset = create_dataset(f, n_var=2)
#     dataset['train_input'].shape, dataset['train_label'].shape

#     # plot KAN at initialization
#     model(dataset['train_input']);
#     model.plot(beta=100, in_vars=[r'$\alpha$', 'x'], out_vars=['y'], title = 'My KAN')

#     # Create your environment
#     envs = gym.make('Humanoid-v4', render_mode="rgb_array")
#     env = gym.wrappers.RecordVideo(envs, f"videos/teste")

#     # Path to the pre-trained model
#     model_filepath = 'model.cleanrl_model'

#     # Load the pre-trained model
#     agent = load_pretrained_model(model_filepath, envs)

#     next_obs, _ = envs.reset(seed=0)
#     #for step in range(0, 1000):
#     next_obs = torch.Tensor(next_obs).to("cpu")

#     with torch.no_grad():
#         action, logprob, entropy, value = agent.get_action_and_value(next_obs.unsqueeze(0))
#         value = value.flatten()

#     # TRY NOT TO MODIFY: execute the game and log data.
#     action = action.squeeze().cpu().numpy()
#     print("Action:", action)
#     next_obs, reward, terminations, truncations, infos = envs.step(action)
#     print("Reward:", reward)

#     # print(f"Action: {action.item()}, Log Probability: {logprob.item()}, Entropy: {entropy.item()}, Value: {value.item()}")

#     envs.render()
#     # agent.actor_mean.prune()
#     agent.actor_mean.plot(beta=3)
