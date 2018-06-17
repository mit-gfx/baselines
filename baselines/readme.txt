To run our script, first navigate to basellines/baselines/, the call this command:
python ./run_pareto_ppo1.py --model-path <whatever> --env <InvertedPendulum-v2|InvertedDoublePendulum-v2|Hopper-v2> --num-timesteps <some integer you like>
For example:
python ./run_pareto_ppo1.py --model-path ./pen --env InvertedPendulum-v2 --num-timesteps 10000

This command will train the neural network controller and save it into a file with prefix = <whatever>
To test the controller, add '--play' at the end of the previous command. This will simulate a rollout using the trained controller.

An overview of the code:
- run_pareto_ppo1.py is the main script. Find the main() function, and you will see a train() function in line 92. This is where all the magic happens.
- Now you enter the train() function in line 58. The inner function 'def policy_fn()' defines your controller. It is a standard neural network.
  * TODO: you can play with hid_size, num_hid_layer to change the structure of your NN.
  * TODO: I currently hard-coded the file_path, you should refactor it as an input argument.
- The policy_fn defines a ParetoMlpPolicy, which we have gone through this afternoon, so I will skip it here.
- Next in the train() function you see make_pareto_mujoco_env in line 67, this is where you define/modify your environment.
- Enter that function in line 18, you will see a list of if/else statements that describes multiple environments.
- For example, line 27 defines an inverted pendulum. In line 28, the original inverted pendulum environment in openai-gym is created. It is built on top
  of mujoco but it is not hard to understand.
- I wrote a 'wrapper' environment for InvertedPendulum in line 29. This wrapper class (InvertedPendulumParetoWrapper) allows you to redefine the reward and action.
- For example, open baselines/pareto_env_wrapper.py, in line 31 you will see the definition of this class. The most important function is step() in line 41.
- In this function, self.env is the original InvertedPendulum environment in openai-gym. For now I just forward whatever I get from env.step to the caller.
- You can unpack ob (observations) and recdefine reward in this function. You can also clamp action before you call self.env.step().
- In general, just think of self.env.step as a Mujoco-based physics simulator that takes your action and returns the observation. You can redfine everything else.
